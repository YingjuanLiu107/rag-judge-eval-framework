# -*- coding: utf-8 -*-
"""批量评测执行引擎：并发调度 + 断点续跑 + 成本/延迟统计。

关键工程点：
1. 断点续跑：每条结果即时 append 到 jsonl，重跑时按 case_id 跳过已完成项，跑 1000 条中断也不丢进度；
2. 并发控制：线程池 + 可配 concurrency，配合客户端重试，兼顾吞吐与限流友好；
3. 指标口径：成功=无调用错误，解析成功=通过 OutputConfig 校验，两者分开统计。
"""

from __future__ import annotations

import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .client import LLMClient
from .config import OutputConfig, PromptConfig, RunConfig
from .logger import get_logger
from .parser import parse_response
from .schema import EvalCase, EvalResult, ModelResponse, RunSummary

log = get_logger("runner")


def build_messages(case: EvalCase, prompt_cfg: PromptConfig) -> List[Dict[str, str]]:
    """把用例渲染成 messages。模板支持 {question} / {context} / {reference} 占位符。"""
    user = prompt_cfg.template
    mapping = {
        "{question}": case.question or "",
        "{context}": case.context or "",
        "{reference}": case.reference or "",
    }
    for k, v in mapping.items():
        user = user.replace(k, v)
    # 测试用例自身未覆盖到的占位符（如 needle 的 {needle}）从 metadata 里取
    for key, value in case.metadata.items():
        user = user.replace("{" + str(key) + "}", str(value))
    if prompt_cfg.response_hint:
        user = f"{user}\n\n{prompt_cfg.response_hint}"

    messages = []
    if prompt_cfg.system:
        messages.append({"role": "system", "content": prompt_cfg.system})
    messages.append({"role": "user", "content": user})
    return messages


class EvalRunner:
    def __init__(
        self,
        client: LLMClient,
        run_cfg: RunConfig,
        prompt_cfg: PromptConfig,
        output_cfg: OutputConfig,
        project_root: Optional[Path] = None,
    ):
        self.client = client
        self.run_cfg = run_cfg
        self.prompt_cfg = prompt_cfg
        self.output_cfg = output_cfg

        self.root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        self.run_id = run_cfg.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out_dir = (self.root / run_cfg.output_dir)
        self.raw_dir = self.out_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.raw_dir / f"{self.run_id}__{client.name}.jsonl"
        self._lock = threading.Lock()
        self._done_ids = self._load_done_ids() if run_cfg.resume else set()

    # ---------------- 断点续跑 ----------------

    def _load_done_ids(self) -> set:
        if not self.jsonl_path.exists():
            return set()
        ids = set()
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("error") in (None, "") or rec.get("parse_ok"):
                        ids.add(str(rec.get("case_id")))
                except Exception:
                    continue
        if ids:
            log.info(f"断点续跑：{self.jsonl_path.name} 已存在 {len(ids)} 条成功记录，将自动跳过")
        return ids

    def _append(self, result: EvalResult) -> None:
        with self._lock:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(result.to_json() + "\n")

    # ---------------- 执行 ----------------

    def run(self, cases: Sequence[EvalCase]) -> List[EvalResult]:
        pending = [c for c in cases if c.case_id not in self._done_ids]
        if self.run_cfg.limit:
            pending = pending[: self.run_cfg.limit]
        if not pending:
            log.info("没有待执行的用例（可能已全部完成）")
            return self.load_results()

        log.info(f"开始跑测：model={self.client.name} 待执行={len(pending)} "
                 f"并发={self.run_cfg.concurrency} mock={self.client.mock}")

        results: List[EvalResult] = []
        t0 = time.time()
        workers = max(1, self.run_cfg.concurrency)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._run_one, c): c for c in pending}
            for i, fut in enumerate(as_completed(futures), 1):
                case = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:  # 单条异常不扩散
                    res = EvalResult(
                        case=case,
                        response=ModelResponse(case_id=case.case_id, model=self.client.name,
                                               error=f"{type(e).__name__}: {e}"),
                    )
                results.append(res)
                self._append(res)
                if i % max(1, len(pending) // 10) == 0 or i == len(pending):
                    log.info(f"进度 {i}/{len(pending)}  耗时 {time.time() - t0:.1f}s")

        self._write_outputs(results)
        return results

    def _run_one(self, case: EvalCase) -> EvalResult:
        messages = build_messages(case, self.prompt_cfg)
        resp = self.client.chat(messages, case_id=case.case_id)
        if not resp.ok:
            return EvalResult(case=case, response=resp, parse_ok=False, parse_error=resp.error)

        parsed, ok, err, level = parse_response(resp.content, self.output_cfg, case.case_id)
        return EvalResult(case=case, response=resp, parsed=parsed, parse_ok=ok, parse_error=err)

    # ---------------- 产出 ----------------

    def load_results(self) -> List[EvalResult]:
        """从 jsonl 反序列化（不含原始 case 全文，够用于汇总与 Judge 输入）。"""
        out: List[EvalResult] = []
        if not self.jsonl_path.exists():
            return out
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out.append(EvalResult(
                    case=EvalCase(case_id=rec.get("case_id", ""), question=rec.get("question", ""),
                                  category=rec.get("category", ""), reference=rec.get("reference"),
                                  metadata=rec.get("metadata") or {}),
                    response=ModelResponse(
                        case_id=rec.get("case_id", ""), model=rec.get("model", ""),
                        content=rec.get("content", ""), latency_ms=rec.get("latency_ms", 0),
                        prompt_tokens=rec.get("prompt_tokens", 0), completion_tokens=rec.get("completion_tokens", 0),
                        total_tokens=rec.get("total_tokens", 0), cost=rec.get("cost", 0.0),
                        error=rec.get("error"),
                    ),
                    parsed=rec.get("parsed"), parse_ok=bool(rec.get("parse_ok")), parse_error=rec.get("parse_error"),
                ))
        return out

    def summarize(self, results: Iterable[EvalResult]) -> RunSummary:
        results = list(results)
        s = RunSummary(run_id=self.run_id, model=self.client.name, total=len(results))
        if not results:
            return s

        latencies: List[int] = []
        cat_stats: Dict[str, Dict[str, float]] = {}
        for r in results:
            ok = r.response.ok
            s.success += int(ok)
            s.failed += int(not ok)
            s.parse_ok += int(r.parse_ok)
            s.total_tokens += r.response.total_tokens
            s.total_cost += r.response.cost
            if ok:
                latencies.append(r.response.latency_ms)

            c = r.case.category or "unknown"
            st = cat_stats.setdefault(c, {"total": 0, "success": 0, "parse_ok": 0, "tokens": 0, "latency_sum": 0})
            st["total"] += 1
            st["success"] += int(ok)
            st["parse_ok"] += int(r.parse_ok)
            st["tokens"] += r.response.total_tokens
            st["latency_sum"] += r.response.latency_ms

        latencies.sort()
        s.avg_latency_ms = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
        s.p95_latency_ms = round(latencies[int(len(latencies) * 0.95) - 1], 1) if latencies else 0.0
        s.total_cost = round(s.total_cost, 4)
        s.by_category = {
            c: {
                "total": int(v["total"]),
                "success_rate": round(v["success"] / v["total"], 4) if v["total"] else 0.0,
                "parse_ok_rate": round(v["parse_ok"] / v["total"], 4) if v["total"] else 0.0,
                "avg_latency_ms": round(v["latency_sum"] / v["total"], 1) if v["total"] else 0.0,
                "tokens": int(v["tokens"]),
            }
            for c, v in cat_stats.items()
        }
        return s

    def _write_outputs(self, results: List[EvalResult]) -> None:
        if not results:
            return
        summary = self.summarize(self.load_results())

        summary_path = self.out_dir / f"summary_{self.run_id}__{self.client.name}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)

        csv_path = self.out_dir / f"results_{self.run_id}__{self.client.name}.csv"
        rows = [r.to_record() for r in self.load_results()]
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                row = dict(row)
                row["parsed"] = json.dumps(row.get("parsed"), ensure_ascii=False) if row.get("parsed") else ""
                writer.writerow(row)

        log.info(f"完成：成功率 {summary.success}/{summary.total}，"
                 f"解析成功 {summary.parse_ok}，tokens {summary.total_tokens}，成本 {summary.total_cost}")
        log.info(f"明细 -> {self.jsonl_path}")
        log.info(f"汇总 -> {summary_path}")
        log.info(f"表格 -> {csv_path}")
