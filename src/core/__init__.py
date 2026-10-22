# -*- coding: utf-8 -*-
"""核心引擎：配置 / 客户端 / 解析 / 执行。"""

from .client import LLMClient, LLMError, build_client_from_model_cfg
from .config import (
    ConfigError,
    EvalConfig,
    ModelConfig,
    OutputConfig,
    PromptConfig,
    RunConfig,
    load_yaml,
)
from .logger import get_logger
from .parser import extract_json, parse_response
from .runner import EvalRunner, build_messages
from .schema import EvalCase, EvalResult, ModelResponse, RunSummary

__all__ = [
    "LLMClient", "LLMError", "build_client_from_model_cfg",
    "ConfigError", "EvalConfig", "ModelConfig", "OutputConfig", "PromptConfig", "RunConfig", "load_yaml",
    "get_logger",
    "extract_json", "parse_response",
    "EvalRunner", "build_messages",
    "EvalCase", "EvalResult", "ModelResponse", "RunSummary",
]
