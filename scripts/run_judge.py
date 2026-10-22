# -*- coding: utf-8 -*-
"""Judge 入口：对引擎产出的答案打分，并做偏差校准与一致性评估。

用法：
    # 1) 打分（多裁判）→ 输出 scores / aggregate（含 Cohen's Kappa）
    python scripts/run_judge.py --results results/raw/exp1__qwen-max.jsonl --run-id exp1

    # 2) 交换测试（位置偏差 / 长度偏好）：模型答案 vs 参考答案
    python scripts/run_judge.py --results results/raw/exp1__qwen-max.jsonl --mode swap --limit 40

    # 3) 只评某一类 / 限制条数 / 裁判重复采样测自洽
    python scripts/run_judge.py --results xxx.jsonl --category injection --limit 30
    python scripts/run_judge.py --results xxx.jsonl --repeat 2 --judge-temperature 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.client import LLMClient  # noqa: E402
from src.core.config import EvalConfig, ModelConfig, load_yaml  # noqa: E402
from src.core.logger import get_logger  # noqa: E402
from src.judge.judge import JudgeRunner  # noqa: E402
from src.judge.rules import rule_flags  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
log = get_logger("run_judge")


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        raise SystemExit(f"结果文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_judges(models_path: Path, names: List[str] | None, repeat: int,
                 temperature: float | None, mock: bool) -> List[LLMClient]:
    raw = load_yaml(models_path)
    defaults = raw.get("defaults", {}) or {}
    judge_cfgs = raw.get("judges") or [raw.get("judge")]
    judge_cfgs = [c for c in judge_cfgs if c]
    if names:
        judge_cfgs = [c for c in judge_cfgs if c.get("name") in names]

    clients: List[LLMClient] = []
    for c in judge_cfgs:
        for r in range(repeat):
            mc = ModelConfig.from_dict(c, defaults)
            kwargs = mc.to_client_kwargs()
            if temperature is not None:
                kwargs["temperature"] = temperature
            if repeat > 1:
                kwargs["name"] = f"{mc.name}#r{r + 1}"
            clients.append(LLMClient(**kwargs, mock=mock))
    return clients


def _answer_text(rec: Dict[str, Any]) -> str:
    parsed = rec.get("parsed")
    if isinstance(parsed, dict) and parsed.get("answer"):
        return str(parsed["answer"])
    return str(rec.get("content") or "")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="LLM-as-a-Judge 打分与偏差校准")
    p.add_argument("--results", required=True, help="引擎产出的 results/raw/*.jsonl")
    p.add_argument("--models-config", default=str(ROOT / "configs" / "models.yaml"))
    p.add_argument("--eval-config", default=str(ROOT / "configs" / "eval_config.yaml"))
    p.add_argument("--mode", default="score", choices=["score", "swap", "length"])
    p.add_argument("--judge", nargs="*", default=None, help="指定裁判名，缺省用配置里全部")
    p.add_argument("--judge-scores", default=None,
                   help="swap 模式下用于筛选非满分样本的 scores_*.jsonl（缺省自动取最新）")
    p.add_argument("--swap-max-score", type=float, default=4.5,
                   help="swap 模式只保留 Judge 均分 <= 该值的样本（避免大量平局）")
    p.add_argument("--repeat", type=int, default=1, help="同一裁判重复采样次数（测自洽）")
    p.add_argument("--judge-temperature", type=float, default=None)
    p.add_argument("--category", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--run-id", default=None)
    p.add_argument("--mock", action="store_true")
    args = p.parse_args(argv)

    EvalConfig.load(args.models_config, args.eval_config, env_path=ROOT / ".env")
    judges = build_judges(Path(args.models_config), args.judge, args.repeat,
                          args.judge_temperature, args.mock)
    if not judges:
        raise SystemExit("没有可用裁判，请检查 configs/models.yaml 的 judges 配置")
    log.info(f"裁判：{[j.name for j in judges]}")

    records = load_records(args.results)
    if args.category:
        records = [r for r in records if r.get("category") in args.category]
    records = [r for r in records if not r.get("error")]
    log.info(f"待评答案 {len(records)} 条")

    runner = JudgeRunner(judges, ROOT, run_id=args.run_id, category_filter=args.category)

    if args.mode == "score":
        rows = runner.score_all(records, concurrency=args.concurrency)
        agg = runner.aggregate(rows, records)
        print("\n===== Judge 汇总 =====")
        print(f"裁判: {agg['judges']}")
        print(f"样本: {agg['n_cases']}  均分: {agg['score_stats'].get('mean')}  "
              f"标准差: {agg['score_stats'].get('std')}")
        for k, v in agg["pairwise"].items():
            print(f"{k}: n={v['n']} 一致率={v['agreement']:.3f} "
                  f"Kappa={v['cohen_kappa']:.3f} 加权Kappa={v['quadratic_weighted_kappa']:.3f} "
                  f"Spearman={v['spearman']:.3f} → {v['band']}")
        print(f"与规则信号冲突: {agg['rule_gap_n']} 条")
    elif args.mode == "swap":
        # 优先挑 Judge 判为"非满分"的样本做对比，否则大量平局会让交换测试失去判别力
        judge_scores_file = Path(args.judge_scores) if args.judge_scores else None
        if judge_scores_file is None:
            cand = sorted((ROOT / "results" / "judge").glob("scores_*.jsonl"),
                          key=lambda p: p.stat().st_mtime)
            judge_scores_file = cand[-1] if cand else None
        if judge_scores_file and Path(judge_scores_file).exists():
            from src.analysis.compare import load_judge_scores
            js = load_judge_scores(Path(judge_scores_file))
            scored = [r for r in records if js.get(r["case_id"], 5) <= args.swap_max_score]
            if scored:
                log.info(f"交换测试优先选用非满分样本：{len(scored)}/{len(records)} 条")
                records = scored
        if args.limit:
            records = records[: args.limit]
        pairs = [{
            "case_id": r["case_id"],
            "category": r.get("category", ""),
            "question": r.get("question", ""),
            "reference": r.get("reference", ""),
            "answer_a": _answer_text(r),
            "answer_b": str(r.get("reference") or ""),  # B 固定为参考答案（ground truth）
        } for r in records]
        out = runner.swap_test(pairs)
        # 金标准偏好率：裁判是否稳定选中参考答案
        gold = []
        for rec in out["records"]:
            if rec["winner_orig"] == "B" or rec["winner_swapped"] == "A":
                gold.append(1)
            elif rec["winner_orig"] in ("A", "B"):
                gold.append(0)
        pref = sum(gold) / len(gold) if gold else 0.0
        out["gold_preference_rate"] = round(pref, 4)
        (runner.out_dir / f"swap_{runner.run_id}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n===== 交换测试（位置偏差 / 长度偏好）=====")
        print(f"样本 {out['swap']['n']} 对  自洽率 {out['swap']['consistency']:.2%}  "
              f"位置偏差率 {out['swap']['position_bias_rate']:.2%}  "
              f"平局率 {out['swap'].get('tie_rate', 0):.2%}")
        print(f"长度偏好（偏好更长答案）{out['length_bias'].get('prefer_longer_rate', 0):.2%}  "
              f"金标准偏好率 {pref:.2%}（有效判定 {len(gold)} 条）")
        print("注：本交换测试中参考答案通常更完整，长度偏好指标与正确性混淆，仅供交叉参考；"
              "干净的长度偏好见 length 模式探针。")

    elif args.mode == "length":
        PAD = ("补充：以上结论依据公司现行人力资源制度，相关条款以书面文本为准，"
               "如需进一步确认请联系人力资源部核实后再行处理。")
        pairs = []
        for i, r in enumerate(records):
            ans = _answer_text(r)
            if not ans or len(ans) < 5:
                continue
            padded = ans + PAD
            if i % 2 == 0:
                a, b, pl = ans, padded, "B"
            else:
                a, b, pl = padded, ans, "A"
            pairs.append({
                "case_id": r["case_id"], "category": r.get("category", ""),
                "question": r.get("question", ""), "reference": r.get("reference", ""),
                "answer_a": a[:1500], "answer_b": b[:1500], "padded_label": pl,
            })
        out = runner.length_probe(pairs)
        print("\n===== 长度偏好探针（冗余扩写对比）=====")
        print(f"样本 {out['n']} 对  偏好更长(冗余)答案 {out['prefer_padded_rate']:.2%}  "
              f"平局率 {out['tie_rate']:.2%}")
        print("判定：平局率越高越无长度偏好；偏好冗余长答案比例越高，长度偏好越明显。")

    # 规则信号
    from src.judge.rules import conflict_detect_rate, defense_rate, hallucination_rate, needle_recall
    print("\n===== 规则校验 =====")
    print("注入防御率:", defense_rate(records))
    print("不可答幻觉率:", hallucination_rate(records))
    print("冲突识别率:", conflict_detect_rate(records))
    print("大海捞针召回:", needle_recall(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
