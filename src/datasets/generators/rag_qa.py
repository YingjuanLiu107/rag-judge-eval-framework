# -*- coding: utf-8 -*-
"""RAG 场景基础问答集：黄金文档 + N 篇干扰文档。

两个可变维度（都写进 metadata，方便后面做归因分析）：
- n_distractors：干扰文档数量（2 / 5），用来测「干扰强度 → 准确率」曲线；
- answerable：可答 / 不可答，不可答题用于测「资料没写时是否老实说不知道」（幻觉检测）。
"""

from __future__ import annotations

from typing import List

from ...core.schema import EvalCase
from .base import pick_distractors, render_context, rng
from .kb import DOCS, UNANSWERABLE_QA, DOC_BY_ID


def build_rag_qa(distractor_levels=(2, 5), seed: int = 42) -> List[EvalCase]:
    cases: List[EvalCase] = []
    r = rng(seed)

    for level in distractor_levels:
        for doc in DOCS:
            for qi, (q, a) in enumerate(doc.qa):
                distractors = pick_distractors(doc, level, seed + qi + hash(doc.doc_id) % 997)
                docs = [doc] + distractors
                context = render_context(docs, seed=seed + qi)
                gold_pos = context.find(f"[{doc.doc_id}]")
                cases.append(EvalCase(
                    case_id=f"ragqa-{doc.doc_id.lower()}-q{qi}-d{level}",
                    category="rag_qa",
                    question=q,
                    context=context,
                    reference=a,
                    metadata={
                        "gold_doc_id": doc.doc_id,
                        "section": doc.section,
                        "n_distractors": level,
                        "gold_char_pos": gold_pos,
                        "answerable": True,
                        "source": "synthetic_kb",
                    },
                ))

    # 不可答题（陷阱）：资料里根本没有答案，正确行为是明确说"资料中未提及"
    for i, (q, a, doc_ids) in enumerate(UNANSWERABLE_QA):
        docs = [DOC_BY_ID[d] for d in doc_ids if d in DOC_BY_ID]
        extra = [d for d in DOCS if d.doc_id not in doc_ids]
        r.shuffle(extra)
        docs = docs + extra[:2]
        cases.append(EvalCase(
            case_id=f"ragqa-unans-{i:02d}",
            category="rag_qa",
            question=q,
            context=render_context(docs, seed=seed + i),
            reference=a,
            metadata={
                "gold_doc_id": None,
                "n_distractors": len(docs) - 1,
                "answerable": False,
                "trap": "unanswerable",
                "source": "synthetic_kb",
            },
        ))

    return cases
