# -*- coding: utf-8 -*-
"""可视化：能力雷达图、成本-性能图、长文本召回热力图、分类别柱状图。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Windows 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = ["#C0392B", "#2980B9", "#27AE60", "#8E44AD", "#E67E22"]


def _ensure(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_radar(radar: Dict[str, Dict[str, float]], dimensions: Sequence[str],
               out_path: Path, title: str = "大模型能力雷达图") -> Path:
    """radar: {model_name: {dim: value}}，取值 0-100。"""
    n = len(dimensions)
    angles = [i * 2 * math.pi / n for i in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for idx, (model, values) in enumerate(radar.items()):
        vals = [float(values.get(d, 0)) for d in dimensions]
        vals += vals[:1]
        color = COLORS[idx % len(COLORS)]
        ax.plot(angles, vals, linewidth=2, label=model, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(dimensions), fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=9)
    ax.set_title(title, fontsize=14, pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=10)
    fig.tight_layout()
    out = _ensure(Path(out_path))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cost_performance(models: Sequence[Dict], out_path: Path,
                          title: str = "成本-性能权衡（每千次调用成本 vs 质量得分）") -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(models):
        cost = m.get("cost_per_1k_calls") or 0
        quality = m.get("quality") or 0
        color = COLORS[i % len(COLORS)]
        ax.scatter(cost, quality, s=140, color=color, zorder=3, label=m.get("model"))
        ax.annotate(m.get("model"), (cost, quality), textcoords="offset points",
                    xytext=(8, 6), fontsize=10, color=color)
    ax.set_xlabel("每千次调用成本（元）", fontsize=11)
    ax.set_ylabel("质量得分（0-100）", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    out = _ensure(Path(out_path))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_needle_heatmap(matrix: Dict[str, Dict[str, float]], out_path: Path,
                        model_name: str = "") -> Path:
    """matrix: {长度档: {深度: 召回率}}"""
    lengths = sorted(matrix.keys(), key=lambda s: int(s.rstrip("k") or 0))
    depths = sorted({d for v in matrix.values() for d in v},
                    key=lambda x: int(x) if str(x).isdigit() else 0)
    data = [[matrix[l].get(d, float("nan")) for d in depths] for l in lengths]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([f"{d}%" for d in depths])
    ax.set_yticks(range(len(lengths)))
    ax.set_yticklabels(lengths)
    ax.set_xlabel("针的插入深度", fontsize=11)
    ax.set_ylabel("上下文长度", fontsize=11)
    ax.set_title(f"大海捞针召回热力图 {('· ' + model_name) if model_name else ''}", fontsize=13)
    for i in range(len(lengths)):
        for j in range(len(depths)):
            v = data[i][j]
            if not (v != v):  # 非 NaN
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=10,
                        color="black" if 0.3 < v < 0.85 else "white")
    fig.colorbar(im, ax=ax, label="召回率")
    fig.tight_layout()
    out = _ensure(Path(out_path))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_category_bars(models: Sequence[Dict], out_path: Path,
                       metric: str = "quality", title: str = "分类别质量得分") -> Path:
    cats = ["rag_qa", "needle", "conflict", "injection"]
    labels = {"rag_qa": "RAG问答", "needle": "长文本", "conflict": "文档冲突", "injection": "注入防御"}
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(cats))
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        vals = [(m.get("by_category", {}).get(c, {}).get(metric) or 0) for c in cats]
        ax.bar([p + i * width for p in x], vals, width=width, label=m.get("model"),
               color=COLORS[i % len(COLORS)])
    ax.set_xticks([p + width * (len(models) - 1) / 2 for p in x])
    ax.set_xticklabels([labels[c] for c in cats], fontsize=11)
    ax.set_ylabel("Judge 均分（1-5）" if metric == "quality" else metric, fontsize=11)
    ax.set_ylim(0, 5.5 if metric == "quality" else 1.05)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend()
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    out = _ensure(Path(out_path))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_score_distribution(scores: Sequence[float], out_path: Path, model_name: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color="#2980B9", alpha=0.85, rwidth=0.8)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Judge 分值", fontsize=11)
    ax.set_ylabel("样本数", fontsize=11)
    ax.set_title(f"Judge 分值分布 {('· ' + model_name) if model_name else ''}", fontsize=13)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    out = _ensure(Path(out_path))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
