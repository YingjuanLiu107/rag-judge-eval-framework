# -*- coding: utf-8 -*-
"""评测流水线中的核心数据结构。

设计目标：
1. 所有中间产物都是可序列化的纯数据（dataclass -> dict -> jsonl），便于断点续跑与复现；
2. 字段覆盖「成本 - 延迟 - 内容 - 解析」四要素，后续 Judge 打分与校准直接消费 EvalResult。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class EvalCase:
    """一条测试用例。"""

    case_id: str                       # 全局唯一 ID，断点续跑靠它去重
    question: str                      # 用户问题
    category: str = "rag_qa"           # rag_qa / needle / conflict / injection ...
    context: Optional[str] = None      # RAG 检索上下文（长文本场景可放 10k+ tokens）
    reference: Optional[str] = None    # 参考答案（可为空，交给 Judge 模型）
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EvalCase":
        known = {"case_id", "question", "category", "context", "reference", "metadata"}
        return EvalCase(
            case_id=str(d.get("case_id") or d.get("id") or ""),
            question=d.get("question", ""),
            category=d.get("category", "rag_qa"),
            context=d.get("context"),
            reference=d.get("reference"),
            metadata=d.get("metadata") or {k: v for k, v in d.items() if k not in known},
        )


@dataclass
class ModelResponse:
    """模型一次调用的原始返回（不含业务解析结果）。"""

    case_id: str
    model: str
    content: str = ""                  # 模型输出的文本（可能是 JSON 字符串）
    latency_ms: int = 0                # 端到端延迟
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0                  # 单次调用成本（按 pricing 配置折算）
    error: Optional[str] = None        # 不为空表示本次调用失败
    raw: Dict[str, Any] = field(default_factory=dict)  # 原始响应体（排错用，可裁剪）

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class EvalResult:
    """用例 + 模型返回 + 解析结果，落盘到 jsonl 的最终记录。"""

    case: EvalCase
    response: ModelResponse
    parsed: Optional[Dict[str, Any]] = None   # 强约束解析后的结构化输出
    parse_ok: bool = False
    parse_error: Optional[str] = None

    def to_record(self) -> Dict[str, Any]:
        return {
            "case_id": self.case.case_id,
            "category": self.case.category,
            "question": self.case.question,
            "reference": self.case.reference,
            "model": self.response.model,
            "content": self.response.content,
            "parsed": self.parsed,
            "parse_ok": self.parse_ok,
            "parse_error": self.parse_error,
            "latency_ms": self.response.latency_ms,
            "prompt_tokens": self.response.prompt_tokens,
            "completion_tokens": self.response.completion_tokens,
            "total_tokens": self.response.total_tokens,
            "cost": round(self.response.cost, 6),
            "error": self.response.error,
            "metadata": self.case.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_record(), ensure_ascii=False)


@dataclass
class RunSummary:
    """一次跑测的汇总指标。"""

    run_id: str
    model: str
    total: int = 0
    success: int = 0
    failed: int = 0
    parse_ok: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["success_rate"] = round(self.success / self.total, 4) if self.total else 0.0
        d["parse_ok_rate"] = round(self.parse_ok / self.total, 4) if self.total else 0.0
        return d


def results_to_rows(results: List[EvalResult]) -> List[Dict[str, Any]]:
    return [r.to_record() for r in results]
