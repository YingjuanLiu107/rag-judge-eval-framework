# -*- coding: utf-8 -*-
"""生成器公共工具：确定性随机、上下文拼装、落盘。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ...core.schema import EvalCase
from .kb import DOCS, Doc


def rng(seed: int) -> random.Random:
    """固定种子，保证每次生成的测试集完全一致（可复现是评测集的底线）。"""
    return random.Random(seed)


def render_doc(doc: Doc) -> str:
    return f"[{doc.doc_id}] 《{doc.title}》\n{doc.text}"


def render_context(docs: Sequence[Doc], seed: int, shuffle: bool = True) -> str:
    items = list(docs)
    if shuffle:
        rng(seed).shuffle(items)
    return "\n\n".join(render_doc(d) for d in items)


def pick_distractors(gold: Doc, k: int, seed: int) -> List[Doc]:
    """优先挑同领域附近的文档做干扰项（更难），不足时跨领域补齐。"""
    r = rng(seed)
    pool = [d for d in DOCS if d.doc_id != gold.doc_id]
    same = [d for d in pool if d.section == gold.section]
    other = [d for d in pool if d.section != gold.section]
    r.shuffle(same)
    r.shuffle(other)
    picked = (same + other)[:k]
    return picked


def save_jsonl(cases: Sequence[EvalCase], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    return p


def stats(cases: Sequence[EvalCase]) -> Dict[str, Any]:
    by_cat: Dict[str, int] = {}
    for c in cases:
        by_cat[c.category] = by_cat.get(c.category, 0) + 1
    return {"total": len(cases), "by_category": by_cat}
