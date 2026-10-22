# -*- coding: utf-8 -*-
"""生成最终分析报告：聚合引擎结果 + Judge 分数 → 图表 + Markdown 报告。

用法：
    python scripts/run_report.py
    python scripts/run_report.py --results qwen-max=results/raw/full__qwen-max.jsonl \
        --judge-scores results/judge/scores_full.jsonl --out docs/04-竞品横向对比分析报告.md
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.compare import build_comparison, load_jsonl, load_judge_scores  # noqa: E402
from src.analysis.visualize import (  # noqa: E402
    plot_category_bars,
    plot_cost_performance,
    plot_needle_heatmap,
    plot_radar,
    plot_score_distribution,
)
from src.core.logger import get_logger  # noqa: E402
from src.judge.metrics import kappa_band  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
log = get_logger("report")


def _latest(pattern: str) -> Optional[Path]:
    files = sorted((ROOT / "results" / "judge").glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="生成评测分析报告")
    p.add_argument("--results", nargs="*", default=None,
                   help="name=path 形式；缺省自动扫描 results/raw/*.jsonl")
    p.add_argument("--judge-scores", default=None)
    p.add_argument("--aggregate", default=None)
    p.add_argument("--swap", default=None)
    p.add_argument("--length", default=None)
    p.add_argument("--figures", default=str(ROOT / "results" / "figures"))
    p.add_argument("--out", default=str(ROOT / "docs" / f"04-竞品横向对比分析报告-{date.today():%Y%m%d}.md"))
    args = p.parse_args(argv)

    if args.results:
        files: Dict[str, Path] = {}
        for item in args.results:
            name, path = item.split("=", 1)
            files[name] = Path(path)
    else:
        # 每个模型只保留用例数最多的那份（避免 stale 小样本/测试文件干扰）
        best: Dict[str, Any] = {}
        for p in sorted((ROOT / "results" / "raw").glob("*.jsonl")):
            name = p.name.split("__")[-1].replace(".jsonl", "")
            if "mock" in name:
                continue
            n = sum(1 for _ in open(p, encoding="utf-8"))
            if name not in best or n > best[name][1]:
                best[name] = (p, n)
        files = {k: v[0] for k, v in best.items()}
    if not files:
        log.error("results/raw 下没有跑测结果，请先运行 scripts/run_eval.py")
        return 1

    judge_scores_file = Path(args.judge_scores) if args.judge_scores else _latest("scores_*.jsonl")
    agg_file = Path(args.aggregate) if args.aggregate else _latest("aggregate_*.json")
    swap_file = Path(args.swap) if args.swap else _latest("swap_*.json")
    length_file = Path(args.length) if getattr(args, "length", None) else _latest("length_*.json")

    cmp_data = build_comparison(files, judge_scores_file)
    models = cmp_data["models"]
    radar = cmp_data["radar"]

    fig_dir = Path(args.figures)
    figures: Dict[str, Path] = {}
    figures["radar"] = plot_radar(radar, cmp_data["dimensions"], fig_dir / "radar.png")
    figures["cost"] = plot_cost_performance(models, fig_dir / "cost_performance.png")
    figures["category"] = plot_category_bars(models, fig_dir / "category_quality.png")

    for m in models:
        if m.get("needle_matrix"):
            figures[f"needle_{m['model']}"] = plot_needle_heatmap(
                m["needle_matrix"], fig_dir / f"needle_{m['model']}.png", m["model"])

    if judge_scores_file:
        scores = list(load_judge_scores(judge_scores_file).values())
        figures["dist"] = plot_score_distribution(scores, fig_dir / "score_distribution.png")

    agg = json.loads(agg_file.read_text(encoding="utf-8")) if agg_file and Path(agg_file).exists() else {}
    swap = json.loads(swap_file.read_text(encoding="utf-8")) if swap_file and Path(swap_file).exists() else {}
    length = json.loads(length_file.read_text(encoding="utf-8")) if length_file and Path(length_file).exists() else {}

    md = render_markdown(models, radar, cmp_data["dimensions"], figures, agg, swap, files, length)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    log.info(f"报告 -> {out}")
    for k, v in figures.items():
        log.info(f"图表 {k} -> {v}")
    return 0


def render_markdown(models, radar, dims, figures, agg, swap, files, len_probe=None) -> str:
    def rel(p: Path) -> str:
        try:
            return str(Path(p).relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            return str(p)

    lines: List[str] = []
    lines.append(f"# AI智能体评测与治理-RAG与长文本抗干扰评测分析报告-{date.today():%Y%m%d}\n")
    lines.append("> 数据口径：全部指标由 `scripts/run_eval.py` 跑测与 `scripts/run_judge.py` 打分自动产出，"
                 "可一键复现；成本按 `configs/models.yaml` 的 pricing 折算（示例价，需按官方最新公示价核对）。\n")

    # 1 结论
    lines.append("## 一、结论先行\n")
    best = max(models, key=lambda m: m.get("quality") or 0)
    lines.append(f"- 本轮共评测 **{len(models)}** 个模型、**{sum(m['n'] for m in models)}** 条用例，"
                 f"覆盖 RAG 问答 / 长文本大海捞针 / 多文档冲突 / 指令注入四类场景。")
    lines.append(f"- 综合质量得分最高的是 **{best['model']}**（{best.get('quality')}/100），"
                 f"任务成功率 {_pct(best.get('success_rate'))}，格式合规率 {_pct(best.get('parse_ok_rate'))}。")
    if models[0].get("injection_defense_rate") is not None:
        lines.append(f"- 指令注入防御率 {_pct(models[0].get('injection_defense_rate'))}，"
                     f"多文档冲突识别率 {_pct(models[0].get('conflict_detect_rate'))}，"
                     f"不可答题幻觉率 {_pct(models[0].get('anti_hallucination_rate'))}。")
    if agg.get("pairwise"):
        k = list(agg["pairwise"].values())[0]
        lines.append(f"- Judge 一致性：二次加权 Cohen's Kappa = **{k['quadratic_weighted_kappa']}**"
                     f"（{k['band']}），一致率 {_pct(k['agreement'])}。")
    if swap.get("swap"):
        lines.append(f"- 交换测试：自洽率 {_pct(swap['swap'].get('consistency'))}，"
                     f"位置偏差率 {_pct(swap['swap'].get('position_bias_rate'))}，"
                     f"长度偏好 {_pct(swap['length_bias'].get('prefer_longer_rate'))}，"
                     f"金标准偏好率 {_pct(swap.get('gold_preference_rate'))}。")
    lines.append("")

    # 2 总体
    lines.append("## 二、总体表现\n")
    lines.append("| 模型 | 样本 | 任务成功率 | 格式合规率 | 质量得分 | 不可答幻觉率↓ | 冲突识别率(严/宽) | 注入防御率 | 长文本召回 | 平均延迟 | 总成本(元) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in models:
        lines.append(
            f"| {m['model']} | {m['n']} | {_pct(m.get('success_rate'))} | {_pct(m.get('parse_ok_rate'))} | "
            f"{m.get('quality')} | {_pct(m.get('anti_hallucination_rate'))} | "
            f"{_pct(m.get('conflict_detect_rate_strict'))} / {_pct(m.get('conflict_detect_rate'))} | "
            f"{_pct(m.get('injection_defense_rate'))} | {_pct(m.get('needle_recall'))} | "
            f"{m.get('avg_latency_ms')}ms | {m.get('total_cost')} |")
    lines.append("")
    lines.append(f"![能力雷达图]({rel(figures['radar'])})\n")
    lines.append(f"![成本-性能]({rel(figures['cost'])})\n")

    # 3 分类别
    lines.append("## 三、分类别表现\n")
    lines.append("| 模型 | 类别 | 样本 | 成功率 | 格式合规率 | Judge均分 | 平均延迟 | tokens | 成本(元) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    cat_cn = {"rag_qa": "RAG问答", "needle": "长文本大海捞针", "conflict": "多文档冲突", "injection": "指令注入"}
    for m in models:
        for c, v in (m.get("by_category") or {}).items():
            lines.append(f"| {m['model']} | {cat_cn.get(c, c)} | {v['n']} | {_pct(v['success_rate'])} | "
                         f"{_pct(v['parse_ok_rate'])} | {v.get('quality') or '—'} | {v['avg_latency_ms']}ms | "
                         f"{v['tokens']} | {v['cost']} |")
    lines.append("")
    lines.append(f"![分类别质量]({rel(figures['category'])})\n")

    # 4 长文本
    lines.append("## 四、长文本召回（大海捞针）\n")
    for m in models:
        if not m.get("needle_matrix"):
            continue
        lines.append(f"**{m['model']}**（召回率）\n")
        depths = sorted({d for v in m["needle_matrix"].values() for d in v},
                        key=lambda x: int(x) if str(x).isdigit() else 0)
        lines.append("| 上下文长度 | " + " | ".join(f"{d}%" for d in depths) + " |")
        lines.append("|---|" + "---|" * len(depths))
        for length in sorted(m["needle_matrix"].keys(), key=lambda s: int(s.rstrip("k") or 0)):
            row = m["needle_matrix"][length]
            lines.append(f"| {length} | " + " | ".join(_pct(row.get(d)) for d in depths) + " |")
        lines.append("")
        if f"needle_{m['model']}" in figures:
            lines.append(f"![大海捞针热力图-{m['model']}]({rel(figures[f'needle_{m["model"]}'])})\n")

    # 5 注入攻击分类型
    for m in models:
        if m.get("attack_breakdown"):
            lines.append("## 五、指令注入：分攻击类型防御率\n")
            lines.append("| 攻击类型 | 样本 | 防御成功 | 防御率 |")
            lines.append("|---|---:|---:|---:|")
            for at, v in sorted(m["attack_breakdown"].items(), key=lambda kv: kv[1]["defense_rate"]):
                lines.append(f"| {at} | {v['n']} | {v['safe']} | {_pct(v['defense_rate'])} |")
            lines.append("")
            weak = [at for at, v in m["attack_breakdown"].items() if v["defense_rate"] < 1.0]
            if weak:
                lines.append(f"> 未完全防住的攻击类型：**{'、'.join(weak)}**——这是护栏加固的第一优先级。\n")

    # 6 干扰强度
    for m in models:
        if m.get("distractor_curve"):
            lines.append("## 六、干扰强度对准确率的影响\n")
            lines.append("| 干扰文档数量 | Judge 均分 |")
            lines.append("|---|---|")
            for k, v in m["distractor_curve"].items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

    # 6 Judge 校准
    lines.append("## Judge 偏差校准与一致性\n")
    if agg:
        lines.append(f"- 裁判：{', '.join(agg.get('judges', []))}；样本 {agg.get('n_cases')} 条；"
                     f"均分 {agg.get('score_stats', {}).get('mean')}，标准差 {agg.get('score_stats', {}).get('std')}")
        lines.append("")
        lines.append("| 裁判对 | 样本 | 一致率 | Cohen's Kappa | 二次加权Kappa | Spearman | 判定 |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for k, v in agg.get("pairwise", {}).items():
            lines.append(f"| {k} | {v['n']} | {_pct(v['agreement'])} | {v['cohen_kappa']} | "
                         f"{v['quadratic_weighted_kappa']} | {v['spearman']} | {v['band']} |")
        lines.append("")
        lines.append(f"- 与规则信号冲突的样本：**{agg.get('rule_gap_n')}** 条"
                     f"（Judge 与确定性规则判定不一致，是 Prompt 迭代的重点）")
    if swap:
        lines.append("")
        lines.append("| 偏差类型 | 指标 | 值 |")
        lines.append("|---|---|---:|")
        lines.append(f"| 位置偏差 | 交换测试自洽率 | {_pct(swap['swap'].get('consistency'))} |")
        lines.append(f"| 位置偏差 | 位置偏差率 | {_pct(swap['swap'].get('position_bias_rate'))} |")
        lines.append(f"| 准确性（金标准偏好率） | 稳定选中参考答案 | {_pct(swap.get('gold_preference_rate'))} |")
        lines.append("")
        lines.append("> 说明：交换测试里参考答案通常更完整，其「偏好更长答案」指标与正确性混淆，"
                     "仅作交叉参考；干净的长度偏好见下方独立探针。")
    if len_probe:
        lines.append("")
        lines.append("| 长度偏好探针 | 指标 | 值 |")
        lines.append("|---|---|---:|")
        lines.append(f"| 冗余扩写对 | 样本数 | {len_probe.get('n')} |")
        lines.append(f"| 偏好更长(冗余)答案 | 比例 | {_pct(len_probe.get('prefer_padded_rate'))} |")
        lines.append(f"| 判为平局 | 比例 | {_pct(len_probe.get('tie_rate'))} |")
        lines.append("")
        lines.append(f"> 判定：平局率 {_pct(len_probe.get('tie_rate'))} 越高越无长度偏好；"
                     f"本次偏好「冗余更长答案」仅 {_pct(len_probe.get('prefer_padded_rate'))}，"
                     f"说明该裁判基本不因答案更长而给更高评价。")
    if "dist" in figures:
        lines.append("")
        lines.append(f"![Judge分值分布]({rel(figures['dist'])})\n")

    # 7 局限
    lines.append("## 结论边界与下一步\n")
    lines.append("- **待补**：本轮仅接入了有 Key 的模型；未接入模型的对比数据需在补齐 API Key 后重跑。")
    lines.append("- **局限**：本轮两个裁判来自同一厂商（同源），Kappa 可能被高估；建议下一步接入 GPT-4o 等"
                 "不同源裁判做交叉验证。")
    lines.append("- **定价**：成本按 `configs/models.yaml` 的示例单价折算，需按厂商官网最新公示价核对后再对外引用。")
    lines.append("- **测试集**：为可自动校验，知识库为虚构公司制度；真实业务评测需替换为业务文档并人工标注参考答案。")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
