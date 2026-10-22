# -*- coding: utf-8 -*-
"""配置加载：YAML 配置 + 环境变量注入，产出可直接使用的强类型配置对象。

约定：
- 配置里所有 `${ENV_VAR}` 占位符会在加载时替换为环境变量，API Key 只存 env，不进仓库；
- 配置文件中出现的 `pricing` 均为「待按官方最新定价核对」的示例值，跑正式评测前须校正。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


class ConfigError(Exception):
    pass


def load_dotenv(path: str | Path = ".env", override: bool = False) -> bool:
    """把 .env 里的键值对注入 os.environ（不覆盖已有环境变量，除非 override=True）。

    优先使用 python-dotenv；未安装时用内置解析兜底，保证零额外依赖也能跑。
    """
    p = Path(path)
    if not p.is_file():
        return False
    try:
        from dotenv import load_dotenv as _ld  # type: ignore
        _ld(str(p), override=override)
        return True
    except ImportError:
        pass
    except Exception:
        pass

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and (override or k not in os.environ):
                os.environ[k] = v
    return True


def _expand_env(node: Any) -> Any:
    """递归替换 ${VAR} / ${VAR:default}。"""
    if isinstance(node, str):
        def repl(m: "re.Match[str]") -> str:
            name, default = m.group(1), m.group(2)
            return os.getenv(name, default if default is not None else "")
        return _ENV_PATTERN.sub(repl, node)
    if isinstance(node, list):
        return [_expand_env(x) for x in node]
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    return node


def load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise ConfigError("未安装 pyyaml，请先 pip install -r requirements.txt")
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"配置文件不存在: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _expand_env(data)


@dataclass
class ModelConfig:
    name: str
    model_id: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 60
    max_retries: int = 3
    pricing: Dict[str, float] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> "ModelConfig":
        dd = defaults or {}
        return ModelConfig(
            name=d.get("name") or d.get("model_id") or "unnamed",
            model_id=d.get("model_id", ""),
            base_url=d.get("base_url", dd.get("base_url", "https://api.openai.com/v1")),
            api_key_env=d.get("api_key_env", "OPENAI_API_KEY"),
            api_key=d.get("api_key", "") or "",
            temperature=d.get("temperature", dd.get("temperature", 0.0)),
            max_tokens=d.get("max_tokens", dd.get("max_tokens", 1024)),
            timeout=d.get("timeout", dd.get("timeout", 60)),
            max_retries=d.get("max_retries", dd.get("max_retries", 3)),
            pricing=d.get("pricing", {}) or {},
            headers=d.get("headers", {}) or {},
        )

    def to_client_kwargs(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "api_key": os.getenv(self.api_key_env, self.api_key) or "",
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "pricing": self.pricing,
            "extra_headers": self.headers,
        }


@dataclass
class OutputConfig:
    """强约束输出解析配置。"""
    format: str = "json"                       # json / text
    required_keys: List[str] = field(default_factory=list)  # 必须出现的字段
    key_types: Dict[str, str] = field(default_factory=dict)  # 字段 -> str/int/float/list/dict
    score_range: Optional[List[int]] = None    # 打分题的分值域，如 [1, 5]
    strict: bool = False                       # True 时不走正则兜底，只接受严格 JSON


@dataclass
class PromptConfig:
    system: str = "你是一个严谨、诚实的问答助手。"
    template: str = "{question}"
    response_hint: str = ""                    # 附加到用户消息末尾的输出格式要求


@dataclass
class RunConfig:
    output_dir: str = "results"
    run_id: Optional[str] = None
    resume: bool = True
    concurrency: int = 8
    limit: Optional[int] = None
    mock: bool = False


@dataclass
class EvalConfig:
    run: RunConfig
    prompt: PromptConfig
    output: OutputConfig
    models: List[ModelConfig] = field(default_factory=list)
    datasets: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(models_path: str | Path, eval_path: str | Path, env_path: str | Path = ".env") -> "EvalConfig":
        load_dotenv(env_path)
        mcfg = load_yaml(models_path)
        ecfg = load_yaml(eval_path)
        defaults = mcfg.get("defaults", {}) or {}
        models = [ModelConfig.from_dict(m, defaults) for m in mcfg.get("models", [])]

        run_raw = ecfg.get("run", {}) or {}
        prompt_raw = ecfg.get("prompt", {}) or {}
        output_raw = ecfg.get("output", {}) or {}

        return EvalConfig(
            run=RunConfig(
                output_dir=run_raw.get("output_dir", "results"),
                run_id=run_raw.get("run_id"),
                resume=bool(run_raw.get("resume", True)),
                concurrency=int(run_raw.get("concurrency", defaults.get("concurrency", 8))),
                limit=run_raw.get("limit"),
                mock=bool(run_raw.get("mock", False)),
            ),
            prompt=PromptConfig(
                system=prompt_raw.get("system", "你是一个严谨、诚实的问答助手。"),
                template=prompt_raw.get("template", "{question}"),
                response_hint=prompt_raw.get("response_hint", ""),
            ),
            output=OutputConfig(
                format=output_raw.get("format", "json"),
                required_keys=output_raw.get("required_keys", []) or [],
                key_types=output_raw.get("key_types", {}) or {},
                score_range=output_raw.get("score_range"),
                strict=bool(output_raw.get("strict", False)),
            ),
            models=models,
            datasets=ecfg.get("datasets", []) or [],
            raw=ecfg,
        )

    def get_model(self, name: str) -> ModelConfig:
        for m in self.models:
            if m.name == name:
                return m
        raise ConfigError(f"未在 configs/models.yaml 中找到模型: {name}（可选: {[m.name for m in self.models]}）")
