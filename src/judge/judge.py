# -*- coding: utf-8 -*-
"""LLM-as-a-Judge 执行器：打分、交换测试（位置偏差）、多 Judge 投票与一致性计算。

三种用法：
1. score       —— 每个 Judge 对每条答案打 1-5 分，多 Judge 取均值/投票；
2. swap        —— 同一对答案交换顺序各评一次，测位置偏差与自洽率；
3. self-repeat —— 同一 Judge 用 temperature>0 重复打分，测裁判自身稳定性（Kappa）。
"""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..core.client import LLMClient
from ..core.config import OutputConfig
from ..core.logger import get_logger
from ..core.parser import parse_response
from ..core.schema import EvalCase
from . import metrics
from .prompts import DEFAULT_RUBRIC, JUDGE_SYSTEM, PAIRWISE_TEMPLATE, RUBRICS, SCORE_TEMPLATE
from .rules import rule_flags

log = get_logger("judge")

SCORE_OUTPUT_CFG = OutputConfig(
    format="json", required_keys=["score"], key_types={"score": "int"}, score_range=[1, 5]
)
PAIR_OUTPUT_CFG = OutputConfig(
    format="json", required_keys=["winner"], key_types={"winner": "str"}
)


def _answer_text(record: Dict[str, Any]) -> str:
    parsed = record.get("parsed")
    if isinstance(parsed, dict) and parsed.get("answer"):
        return str(parsed["answer"])
    return str(record.get("content") or "")


class JudgeRunner:
    def __init__(self, judges: Sequence[LLMClient], project_root: Path,
                 run_id: Optional[str] = None, category_filter: Optional[List[str]] = None):
        self.judges = list(judges)
        self.root = Path(project_root)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.category_filter = category_filter
        self.out_dir = self.root / "results" / "judge"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 单答案打分 ----------------

    def _score_one(self, judge: LLMClient, record: Dict[str, Any]) -> Dict[str, Any]:
        category = record.get("category", "rag_qa")
        prompt = SCORE_TEMPLATE.format(
            category=category,
            rubric=RUBRICS.get(category, DEFAULT_RUBRIC),
            question=record.get("question", ""),
            reference=record.get("reference", ""),
            answer=_answer_text(record)[:2000],
        )
        resp = judge.chat(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
            case_id=record.get("case_id", ""),
        )
        parsed, ok, err, level = parse_response(resp.content, SCORE_OUTPUT_CFG, record.get("case_id", ""))
        score = None
        if parsed and isinstance(parsed.get("score"), int):
            score = parsed["score"]
        elif parsed:
            # 兼容 "score": "4" 这类字符串
            try:
                score = int(float(parsed.get("score")))
            except (TypeError, ValueError):
                score = None
        if score is not None and not (1 <= score <= 5):
            score, ok, err = None, False, f"分值 {score} 超出 1-5"
        return {
            "case_id": record.get("case_id"),
            "category": category,
            "model": record.get("model"),
            "judge": judge.name,
            "score": score,
            "evidence": (parsed or {}).get("evidence", ""),
            "hallucination": bool((parsed or {}).get("hallucination", False)),
            "judge_parse_ok": ok,
            "judge_parse_error": err,
            "judge_parse_level": level,
            "judge_latency_ms": resp.latency_ms,
            "judge_tokens": resp.total_tokens,
            "judge_cost": round(resp.cost, 6),
            "judge_error": resp.error,
        }

    def score_all(self, records: Sequence[Dict[str, Any]], concurrency: int = 8) -> List[Dict[str, Any]]:
        recs = [r for r in records if not self.category_filter or r.get("category") in self.category_filter]
        log.info(f"Judge 打分：{len(recs)} 条 × {len(self.judges)} 个裁判 = {len(recs) * len(self.judges)} 次调用")

        rows: List[Dict[str, Any]] = []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = []
            for r in recs:
                for j in self.judges:
                    futures.append(pool.submit(self._score_one, j, r))
            for i, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if i % 50 == 0 or i == len(futures):
                    log.info(f"Judge 进度 {i}/{len(futures)} 耗时 {time.time() - t0:.0f}s")

        path = self.out_dir / f"scores_{self.run_id}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info(f"Judge 明细 -> {path}")
        return rows

    def aggregate(self, rows: Sequence[Dict[str, Any]],
                  records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """按 case 聚合多 Judge 结果，并计算一致性指标。"""
        by_case: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_case.setdefault(r["case_id"], []).append(r)

        merged: Dict[str, Dict[str, Any]] = {}
        for cid, items in by_case.items():
            scores = [i["score"] for i in items if i.get("score")]
            merged[cid] = {
                "case_id": cid,
                "category": items[0]["category"],
                "model": items[0]["model"],
                "scores": scores,
                "final_score": round(statistics.mean(scores), 2) if scores else None,
                "majority": statistics.mode(scores) if scores else None,
                "judge_agree": (len(set(scores)) == 1) if len(scores) > 1 else None,
                "hallucination": any(i.get("hallucination") for i in items),
                "judges": [i["judge"] for i in items],
            }

        # 一致性：多 Judge 两两 Kappa（同一批 case 上的打分序列）
        judge_names = sorted({r["judge"] for r in rows})
        seqs: Dict[str, Dict[str, int]] = {n: {} for n in judge_names}
        for r in rows:
            if r.get("score"):
                seqs[r["judge"]][r["case_id"]] = r["score"]

        pairwise: Dict[str, Dict[str, float]] = {}
        names = list(judge_names)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                common = sorted(set(seqs[a]) & set(seqs[b]))
                r1 = [seqs[a][c] for c in common]
                r2 = [seqs[b][c] for c in common]
                key = f"{a}__vs__{b}"
                pairwise[key] = {
                    "n": len(common),
                    "agreement": metrics.agreement_rate(r1, r2),
                    "cohen_kappa": metrics.cohens_kappa(r1, r2),
                    "quadratic_weighted_kappa": metrics.cohens_kappa(r1, r2, weights="quadratic"),
                    "spearman": metrics.spearman(r1, r2),
                    "band": metrics.kappa_band(metrics.cohens_kappa(r1, r2, weights="quadratic")),
                }

        # 与规则信号的一致性（Judge 是否把"编造"判为低分）
        rec_by_id = {r.get("case_id"): r for r in records}
        rule_gap = []
        for cid, m in merged.items():
            rec = rec_by_id.get(cid) or {}
            flags = rule_flags(rec)
            if flags.get("injection_executed") and (m["final_score"] or 0) >= 4:
                rule_gap.append({"case_id": cid, "issue": "规则判定已执行注入，但 Judge 给了高分"})
            if flags.get("unanswerable_compliant") is False and (m["final_score"] or 0) >= 4:
                rule_gap.append({"case_id": cid, "issue": "不可答题疑似编造，但 Judge 给了高分"})

        summary = {
            "run_id": self.run_id,
            "judges": judge_names,
            "n_cases": len(merged),
            "score_stats": metrics.score_stats([m["final_score"] for m in merged.values() if m["final_score"]]),
            "pairwise": pairwise,
            "rule_gap_n": len(rule_gap),
            "rule_gap_examples": rule_gap[:5],
        }
        path = self.out_dir / f"aggregate_{self.run_id}.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"Judge 汇总 -> {path}")
        return summary

    # ---------------- 交换测试（位置偏差） ----------------

    def swap_test(self, pairs: Sequence[Dict[str, Any]], judge: Optional[LLMClient] = None) -> Dict[str, Any]:
        """pairs: [{'case_id', 'question', 'reference', 'answer_a', 'answer_b'}]

        对同一对答案按 A-B 与 B-A 两种顺序各评一次，统计自洽率与位置偏差率。
        """
        judge = judge or self.judges[0]
        results: List[Dict[str, Any]] = []
        log.info(f"交换测试：{len(pairs)} 对 × 2 次 = {len(pairs) * 2} 次调用")

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {}
            for p in pairs:
                futures[pool.submit(self._pairwise, judge, p, swap=False)] = (p, "orig")
                futures[pool.submit(self._pairwise, judge, p, swap=True)] = (p, "swapped")
            got: Dict[str, Dict[str, str]] = {}
            for fut in as_completed(futures):
                p, kind = futures[fut]
                got.setdefault(p["case_id"], {})[kind] = fut.result()

        pref_longer = []
        for p in pairs:
            g = got.get(p["case_id"], {})
            w1, w2 = g.get("orig"), g.get("swapped")
            if w1 in ("A", "B"):
                chosen = p["answer_a"] if w1 == "A" else p["answer_b"]
                other = p["answer_b"] if w1 == "A" else p["answer_a"]
                pref_longer.append(int(len(chosen) >= len(other)))
            results.append({
                "case_id": p["case_id"],
                "category": p.get("category", ""),
                "winner_orig": w1,
                "winner_swapped": w2,
                "consistent": (w1 != w2) if w1 in ("A", "B") and w2 in ("A", "B")
                else (str(w1).upper() == str(w2).upper() == "TIE"),
            })

        bias = metrics.position_bias(results)
        lb = metrics.length_bias(pref_longer)
        out = {
            "run_id": self.run_id,
            "judge": judge.name,
            "swap": bias,
            "length_bias": lb,
            "records": results,
        }
        path = self.out_dir / f"swap_{self.run_id}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"交换测试：自洽率 {bias['consistency']:.2%}，位置偏差率 {bias['position_bias_rate']:.2%}，"
                 f"长度偏好 {lb.get('prefer_longer_rate', 0):.2%}")
        log.info(f"交换测试明细 -> {path}")
        return out

    def run_pairs(self, pairs: Sequence[Dict[str, Any]], swap: bool = False) -> Dict[str, str]:
        """对给定的答案对做两两比较，返回 {case_id: winner}。"""
        out: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(self._pairwise, self.judges[0], p, swap): p for p in pairs}
            for fut in as_completed(futures):
                p = futures[fut]
                out[p["case_id"]] = fut.result()
        return out

    def length_probe(self, pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """长度偏好探针：一对答案内容等价、只有一个更长（加了冗余扩写句）。

        无偏裁判应判平局；若稳定选择更长的那个，说明存在长度偏好。
        pairs 需含 {'case_id', 'question', 'reference', 'answer_a', 'answer_b', 'padded_label'}。
        """
        winners = self.run_pairs(pairs)
        padded, tie, total = 0, 0, 0
        records = []
        for p in pairs:
            w = winners.get(p["case_id"], "unknown")
            if w == p.get("padded_label"):
                padded += 1
            elif w == "TIE":
                tie += 1
            records.append({"case_id": p["case_id"], "winner": w, "padded_label": p.get("padded_label")})
            if w in ("A", "B", "TIE"):
                total += 1
        out = {
            "run_id": self.run_id,
            "judge": self.judges[0].name,
            "n": total,
            "prefer_padded_rate": round(padded / total, 4) if total else 0.0,
            "tie_rate": round(tie / total, 4) if total else 0.0,
            "records": records,
        }
        path = self.out_dir / f"length_{self.run_id}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"长度偏好探针：偏好更长答案 {out['prefer_padded_rate']:.2%}，平局率 {out['tie_rate']:.2%} -> {path}")
        return out

    def _pairwise(self, judge: LLMClient, pair: Dict[str, Any], swap: bool) -> str:
        a = pair["answer_b"] if swap else pair["answer_a"]
        b = pair["answer_a"] if swap else pair["answer_b"]
        prompt = PAIRWISE_TEMPLATE.format(
            question=pair["question"], reference=pair.get("reference", ""),
            answer_a=a[:1500], answer_b=b[:1500],
        )
        resp = judge.chat(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
            case_id=f"{pair['case_id']}-swap" if swap else pair["case_id"],
        )
        parsed, ok, err, _ = parse_response(resp.content, PAIR_OUTPUT_CFG, pair["case_id"])
        w = (parsed or {}).get("winner") if ok else None
        if isinstance(w, str):
            w = w.strip().upper()
            if w not in ("A", "B", "TIE"):
                w = None
        return w or "unknown"
