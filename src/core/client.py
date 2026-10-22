# -*- coding: utf-8 -*-
"""LLM API 客户端：统一封装 OpenAI 兼容接口。

能力：
1. 兼容 OpenAI / Qwen(dashscope compatible-mode) / GLM(paas v4) 等 OpenAI 风格端点，
   直接走 REST（/chat/completions），不绑定任何厂商 SDK；
2. 指数退避重试：只对可恢复错误（超时、连接错误、429、5xx）重试，4xx 参数错误快速失败；
3. Token 与成本统计：优先用响应体 usage，缺失时用启发式估算，保证成本核算不中断；
4. Mock 模式：无 Key 时也能跑通全链路，用于离线冒烟测试与 CI。
"""

from __future__ import annotations

import json
import time
import hashlib
from typing import Any, Dict, List, Optional

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    requests = None  # 允许在未安装 requests 时导入本模块（配合 mock 模式）

from .logger import get_logger
from .schema import ModelResponse

log = get_logger("client")

# 可重试的错误关键词（网络抖动 / 限流 / 服务端错误）
_RETRYABLE_HINTS = ("timeout", "timed out", "connection", "429", "500", "502", "503", "504")


class LLMError(Exception):
    """LLM 调用失败的基类。"""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class LLMClient:
    """单个模型端点的调用封装（线程安全，可被线程池并发调用）。"""

    def __init__(
        self,
        name: str,
        model_id: str,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 60,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        pricing: Optional[Dict[str, float]] = None,
        mock: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.backoff_base = backoff_base
        self.pricing = pricing or {}
        self.mock = mock or not api_key
        self.extra_headers = extra_headers or {}

        if self.mock:
            log.warn(f"[{self.name}] 未配置 API Key 或显式开启 mock，进入离线模拟模式（不产生真实调用）")

    # ---------------- 对外主入口 ----------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        case_id: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> ModelResponse:
        """发起一次对话请求，带重试；失败返回 error 字段而非抛异常，保证批量任务不中断。"""
        temp = self.temperature if temperature is None else temperature
        mtok = self.max_tokens if max_tokens is None else max_tokens

        last_err = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.mock:
                    return self._mock_chat(messages, case_id)
                return self._http_chat(messages, case_id, temp, mtok, response_format)
            except LLMError as e:
                last_err = str(e)
                if not e.retryable or attempt == self.max_retries:
                    break
                sleep_s = self.backoff_base ** (attempt - 1)
                log.warn(f"[{self.name}] {case_id} 第 {attempt} 次调用失败（{last_err}），{sleep_s:.1f}s 后重试")
                time.sleep(sleep_s)
            except Exception as e:  # 兜底：任何未知异常都不应打断整批任务
                last_err = f"{type(e).__name__}: {e}"
                if attempt == self.max_retries:
                    break
                time.sleep(self.backoff_base ** (attempt - 1))

        return ModelResponse(case_id=case_id, model=self.name, error=last_err or "unknown error")

    # ---------------- 内部实现 ----------------

    def _http_chat(self, messages, case_id, temperature, max_tokens, response_format) -> ModelResponse:
        if requests is None:
            raise LLMError("未安装 requests，请先 pip install -r requirements.txt", retryable=False)

        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        t0 = time.time()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        except Exception as e:
            raise LLMError(f"{type(e).__name__}: {e}", retryable=True)

        latency_ms = int((time.time() - t0) * 1000)

        if resp.status_code != 200:
            body = resp.text[:300]
            retryable = resp.status_code == 429 or resp.status_code >= 500
            raise LLMError(f"HTTP {resp.status_code}: {body}", retryable=retryable)

        try:
            data = resp.json()
        except Exception:
            raise LLMError("响应体不是合法 JSON", retryable=True)

        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"响应结构异常: {json.dumps(data, ensure_ascii=False)[:300]}", retryable=False)

        usage = data.get("usage") or {}
        p_tok = int(usage.get("prompt_tokens") or 0)
        c_tok = int(usage.get("completion_tokens") or 0)
        if p_tok == 0:  # 部分网关不返回 usage，回退到启发式估算
            p_tok = self.estimate_tokens("".join(m.get("content", "") for m in messages))
        if c_tok == 0:
            c_tok = self.estimate_tokens(content)

        return ModelResponse(
            case_id=case_id,
            model=self.name,
            content=content.strip(),
            latency_ms=latency_ms,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=p_tok + c_tok,
            cost=self.compute_cost(p_tok, c_tok),
            raw={"id": data.get("id"), "finish_reason": _finish_reason(data)},
        )

    def _mock_chat(self, messages, case_id) -> ModelResponse:
        """离线模拟返回一个合法 JSON，用于跑通解析与落盘链路。"""
        prompt = "".join(m.get("content", "") for m in messages)
        seed = int(hashlib.md5((case_id + prompt).encode("utf-8")).hexdigest()[:8], 16)
        time.sleep(0.01)
        answer = f"mock-answer-{seed % 1000}"
        content = json.dumps(
            {"answer": answer, "confidence": round((seed % 100) / 100, 2), "citations": ["doc-1"]},
            ensure_ascii=False,
        )
        if seed % 7 == 0:  # 故意掺入 1/7 的脏输出，验证解析器的容错能力
            content = f"好的，我的回答如下：\n```json\n{content}\n```\n希望有帮助。"

        p_tok = self.estimate_tokens(prompt)
        c_tok = self.estimate_tokens(content)
        return ModelResponse(
            case_id=case_id,
            model=self.name,
            content=content,
            latency_ms=seed % 800 + 120,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=p_tok + c_tok,
            cost=self.compute_cost(p_tok, c_tok),
            raw={"mock": True},
        )

    # ---------------- 工具方法 ----------------

    def compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """按 pricing 配置计算成本（单位与配置一致，默认人民币）。"""
        pin = float(self.pricing.get("prompt_per_1k", 0.0))
        pout = float(self.pricing.get("completion_per_1k", 0.0))
        return (prompt_tokens / 1000.0) * pin + (completion_tokens / 1000.0) * pout

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """无 usage 时的启发式估算：中日韩字符约 1 token/字，其余约 4 字符/token。"""
        if not text:
            return 0
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        others = len(text) - cjk
        return int(cjk * 1.0 + others / 4.0) + 1


def _finish_reason(data: Dict[str, Any]) -> str:
    try:
        return data["choices"][0].get("finish_reason", "")
    except (KeyError, IndexError, TypeError):
        return ""


def build_client_from_model_cfg(cfg: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None,
                                mock: bool = False) -> LLMClient:
    """从 configs/models.yaml 的模型条目构造客户端。"""
    d = defaults or {}
    import os
    api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env, cfg.get("api_key", "") or "")
    return LLMClient(
        name=cfg.get("name", cfg.get("model_id", "unnamed")),
        model_id=cfg.get("model_id", ""),
        api_key=api_key,
        base_url=cfg.get("base_url", d.get("base_url", "https://api.openai.com/v1")),
        temperature=cfg.get("temperature", d.get("temperature", 0.0)),
        max_tokens=cfg.get("max_tokens", d.get("max_tokens", 1024)),
        timeout=cfg.get("timeout", d.get("timeout", 60)),
        max_retries=cfg.get("max_retries", d.get("max_retries", 3)),
        pricing=cfg.get("pricing", {}),
        mock=mock,
        extra_headers=cfg.get("headers", {}),
    )
