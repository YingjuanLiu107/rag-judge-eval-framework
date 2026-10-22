# -*- coding: utf-8 -*-
"""Judge 模块：LLM-as-a-Judge 打分、偏差校准与一致性评估。"""

from . import metrics, prompts, rules
from .judge import JudgeRunner, PAIR_OUTPUT_CFG, SCORE_OUTPUT_CFG
from .metrics import cohens_kappa, kappa_band, position_bias, length_bias, spearman
from .rules import defense_rate, hallucination_rate, conflict_detect_rate, needle_recall, rule_flags

__all__ = [
    "JudgeRunner", "SCORE_OUTPUT_CFG", "PAIR_OUTPUT_CFG",
    "metrics", "prompts", "rules",
    "cohens_kappa", "kappa_band", "position_bias", "length_bias", "spearman",
    "defense_rate", "hallucination_rate", "conflict_detect_rate", "needle_recall", "rule_flags",
]
