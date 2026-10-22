# -*- coding: utf-8 -*-
"""测试集生成器：RAG 问答 / 大海捞针 / 多文档冲突 / 指令注入。"""

from ...core.schema import EvalCase
from .base import save_jsonl, stats
from .conflict import build_conflict
from .injection import build_injection
from .needle import build_needle
from .rag_qa import build_rag_qa

BUILDERS = {
    "rag_qa": build_rag_qa,
    "needle": build_needle,
    "conflict": build_conflict,
    "injection": build_injection,
}

__all__ = ["build_rag_qa", "build_needle", "build_conflict", "build_injection",
           "save_jsonl", "stats", "BUILDERS", "EvalCase"]
