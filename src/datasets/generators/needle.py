# -*- coding: utf-8 -*-
"""「大海捞针」（Needle in a Haystack）长文本测试集。

构造方式：用知识库原文拼出指定长度的填充文本（haystack），把一条关键事实（needle）
插到指定深度位置，然后问这条事实。可控变量写进 metadata：
- context_chars：上下文长度（2k / 8k 字符）
- depth_pct：针所在的深度百分比（0 / 25 / 50 / 75 / 100）
用于绘制「长度 × 深度」热力图，定位模型长文本召回的失效边界。
"""

from __future__ import annotations

from typing import List, Sequence

from ...core.schema import EvalCase
from .base import rng
from .kb import DOCS

# (needle_id, 针文本, 问题, 参考答案)
NEEDLES = [
    ("n1", "补充记录：2026 年星海科技年会确定在成都举办，举办日期为 4 月 18 日。",
     "2026 年公司年会在哪个城市举办？", "成都"),
    ("n2", "补充记录：代号「灯塔项目」的年度预算总额为 3,700,000 元，由战略部归口管理。",
     "「灯塔项目」的年度预算总额是多少？", "3,700,000 元（370 万元）"),
    ("n3", "补充记录：客户「远洋集团」的合同续签截止日期为 2026 年 11 月 5 日，逾期视为自动终止。",
     "客户「远洋集团」的合同续签截止日期是什么时候？", "2026 年 11 月 5 日"),
]

_FILLER_SENTENCES = [d.text for d in DOCS]


def _build_haystack(target_chars: int, seed: int) -> str:
    r = rng(seed)
    pool = list(_FILLER_SENTENCES)
    r.shuffle(pool)
    chunks: List[str] = []
    size = 0
    i = 0
    while size < target_chars:
        chunk = f"（第{i // len(pool) + 1}轮摘录）{pool[i % len(pool)]}"
        chunks.append(chunk)
        size += len(chunk)
        i += 1
    return "\n\n".join(chunks)


def _insert_needle(haystack: str, needle: str, depth_pct: int) -> str:
    """按字符位置插入到指定深度（0=开头，100=末尾）。"""
    pos = int(len(haystack) * depth_pct / 100)
    # 对齐到最近的段落边界，避免把针插进句子中间导致语义破碎
    boundary = haystack.find("\n\n", pos)
    if boundary == -1:
        boundary = len(haystack)
    return haystack[:boundary] + "\n\n" + needle + "\n\n" + haystack[boundary:]


def build_needle(sizes: Sequence[int] = (2000, 8000),
                 depths: Sequence[int] = (0, 25, 50, 75, 100),
                 seed: int = 7) -> List[EvalCase]:
    cases: List[EvalCase] = []
    for si, size in enumerate(sizes):
        hay = _build_haystack(size, seed + si)
        for ni, (nid, needle_text, q, a) in enumerate(NEEDLES):
            for depth in depths:
                ctx = _insert_needle(hay, needle_text, depth)
                cases.append(EvalCase(
                    case_id=f"needle-{nid}-{size//1000}k-d{depth}",
                    category="needle",
                    question=q,
                    context=ctx,
                    reference=a,
                    metadata={
                        "needle_id": nid,
                        "context_chars": len(ctx),
                        "depth_pct": depth,
                        "answerable": True,
                        "source": "synthetic_kb",
                    },
                ))
    return cases
