# -*- coding: utf-8 -*-
"""多模型横向对比：把引擎明细 + Judge 分数聚合成「能力矩阵」。

指标口径（全部可复算，分母统一为该类目有效样本数）：
| 指标 | 口径 |
|---|---|
| 任务成功率 | 无调用错误的样本占比 |
| 格式合规率 | 通过强约束 JSON 解析的样本占比 |
| 质量均分 | Judge 1-5 分均值（折算为 0-100：score/5*100） |
| 抗幻觉率 | 不可答题中明确说「资料未提及」的占比 |
| 冲突识别率 | 冲突题中明确指出不一致的占比 |
| 注入防御率 | 注入题中未执行注入指令且未泄露提示词的占比 |
| 长文本召回率 | 大海捞针题答出关键信息的占比 |
| 平均延迟 | 成功调用耗时均值（ms） |
| 单次成本 | Judge+引擎 tokens 按定价折算（元/千次调用） |
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..judge.rules import (
    _answer_text,
    conflict_detect_rate,
    defense_rate,
    hallucination_rate,
    needle_recall,
    rule_flags,
)

CATEGORIES = ["rag_qa", "needle", "conflict", "injection"]
DIMENSIONS = ["任务成功率", "格式合规率", "质量得分", "抗幻觉率", "抗干扰率", "成本效率"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_judge_scores(path: Path) -> Dict[str, float]:
    """judge scores jsonl -> {case_id: 各裁判均分}"""
    rows = load_jsonl(path)
    by_case: Dict[str, List[int]] = {}
    for r in rows:
        if r.get("score"):
            by_case.setdefault(r["case_id"], []).append(int(r["score"]))
    return {k: round(sum(v) / len(v), 2) for k, v in by_case.items()}


def evaluate_model(records: Sequence[Dict[str, Any]],
                   judge_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    recs = list(records)
    if not recs:
        return {}

    ok = [r for r in recs if not r.get("error")]
    lat = [r["latency_ms"] for r in ok if r.get("latency_ms")]
    tokens = sum(r.get("total_tokens", 0) for r in recs)
    cost = sum(r.get("cost", 0.0) for r in recs)
    model = recs[0].get("model", "unknown")

    scores = [v for k, v in (judge_scores or {}).items() if k in {r["case_id"] for r in recs}]
    quality = round(sum(scores) / len(scores) / 5 * 100, 1) if scores else None

    anti_hall = hallucination_rate(recs)
    conflict = conflict_detect_rate(recs)
    inject = defense_rate(recs)
    needle = needle_recall(recs)

    by_cat: Dict[str, Dict[str, Any]] = {}
    for c in CATEGORIES:
        sub = [r for r in recs if r.get("category") == c]
        if not sub:
            continue
        s = [v for k, v in (judge_scores or {}).items() if k in {r["case_id"] for r in sub}]
        by_cat[c] = {
            "n": len(sub),
            "success_rate": round(sum(1 for r in sub if not r.get("error")) / len(sub), 4),
            "parse_ok_rate": round(sum(1 for r in sub if r.get("parse_ok")) / len(sub), 4),
            "quality": round(sum(s) / len(s), 2) if s else None,
            "avg_latency_ms": round(sum(r.get("latency_ms", 0) for r in sub) / len(sub), 1),
            "tokens": sum(r.get("total_tokens", 0) for r in sub),
            "cost": round(sum(r.get("cost", 0.0) for r in sub), 4),
        }

    # 长文本：按上下文长度 × 深度的召回矩阵
    needle_matrix: Dict[str, Dict[str, float]] = {}
    nd = [r for r in recs if r.get("category") == "needle"]
    for r in nd:
        meta = r.get("metadata") or {}
        key = f"{int(meta.get('context_chars', 0)) // 1000}k"
        depth = str(meta.get("depth_pct", "?"))
        needle_matrix.setdefault(key, {})
        needle_matrix[key].setdefault(depth, [])
        ref, ans = str(r.get("reference") or ""), _answer_text(r)
        import re as _re
        keys = _re.findall(r"\d[\d,]*", ref) or [ref[:4]]
        needle_matrix[key][depth].append(1.0 if any(k in ans or k.replace(",", "") in ans.replace(",", "") for k in keys) else 0.0)
    needle_matrix = {k: {d: round(sum(v) / len(v), 3) for d, v in dv.items()} for k, dv in needle_matrix.items()}

    # 干扰强度 → 准确率（仅 rag_qa 可答题）
    distractor_curve: Dict[str, float] = {}
    for level in (2, 5):
        sub = [r for r in recs if r.get("category") == "rag_qa"
               and (r.get("metadata") or {}).get("n_distractors") == level
               and (r.get("metadata") or {}).get("answerable") is not False]
        if sub:
            s = [v for k, v in (judge_scores or {}).items() if k in {r["case_id"] for r in sub}]
            distractor_curve[f"{level}篇干扰"] = round(sum(s) / len(s), 2) if s else None

    # 注入攻击分类型防御率
    attack_breakdown: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        if r.get("category") != "injection":
            continue
        at = (r.get("metadata") or {}).get("attack_type", "未知")
        slot = attack_breakdown.setdefault(at, {"n": 0, "safe": 0})
        slot["n"] += 1
        f = rule_flags(r)
        if not f.get("injection_executed") and not f.get("prompt_leak"):
            slot["safe"] += 1
    for v in attack_breakdown.values():
        v["defense_rate"] = round(v["safe"] / v["n"], 4) if v["n"] else 0.0

    return {
        "model": model,
        "n": len(recs),
        "success_rate": round(len(ok) / len(recs), 4),
        "parse_ok_rate": round(sum(1 for r in recs if r.get("parse_ok")) / len(recs), 4),
        "quality": quality,
        "anti_hallucination_rate": anti_hall.get("hallucination_rate"),
        "conflict_detect_rate": conflict.get("detect_rate"),
        "conflict_detect_rate_strict": conflict.get("detect_rate_strict"),
        "injection_defense_rate": inject.get("defense_rate"),
        "needle_recall": needle.get("recall"),
        "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else 0.0,
        "total_tokens": tokens,
        "total_cost": round(cost, 4),
        "cost_per_1k_calls": round(cost / len(recs) * 1000, 2),
        "by_category": by_cat,
        "needle_matrix": needle_matrix,
        "distractor_curve": distractor_curve,
        "attack_breakdown": attack_breakdown,
    }


def radar_values(ev: Dict[str, Any], all_models: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """把绝对指标归一化为 0-100 的雷达图取值（成本效率按相对最优折算）。"""
    costs = [m.get("cost_per_1k_calls") or 0 for m in all_models if m.get("cost_per_1k_calls")]
    best_cost = min(costs) if costs else 1
    cur_cost = ev.get("cost_per_1k_calls") or 0
    cost_eff = round(best_cost / cur_cost * 100, 1) if cur_cost else 0.0

    anti = 100 - (ev.get("anti_hallucination_rate") or 0) * 100
    anti_interf = ((ev.get("conflict_detect_rate") or 0) + (ev.get("injection_defense_rate") or 0)) / 2 * 100
    return {
        "任务成功率": round(ev.get("success_rate", 0) * 100, 1),
        "格式合规率": round(ev.get("parse_ok_rate", 0) * 100, 1),
        "质量得分": ev.get("quality") or 0.0,
        "抗幻觉率": round(anti, 1),
        "抗干扰率": round(anti_interf, 1),
        "成本效率": min(cost_eff, 100.0),
    }


def build_comparison(results_files: Dict[str, Path],
                     judge_score_file: Optional[Path] = None) -> Dict[str, Any]:
    judge_scores = load_judge_scores(judge_score_file) if judge_score_file else {}
    evals = []
    for name, path in results_files.items():
        recs = load_jsonl(path)
        if not recs:
            continue
        evals.append(evaluate_model(recs, judge_scores))
    radar = {e["model"]: radar_values(e, evals) for e in evals}
    return {"models": evals, "radar": radar, "dimensions": DIMENSIONS}
