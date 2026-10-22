# -*- coding: utf-8 -*-
"""强约束输出解析器：把模型自由文本稳稳落成结构化 JSON。

四级解析策略（逐级降级的兜底设计，保证解析率可量化）：
L1 直接 json.loads
L2 剥离 ```json 代码围栏 / 前后废话后 json.loads
L3 花括号配对扫描，提取第一个完整的 JSON 对象
L4 正则兜底抽取关键字段（如 "score": 4），并标记 parsed_by=regex

解析结果统一附带 `_meta`，记录解析层级，便于后续分析「哪类 case 容易诱导模型输出失控」。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

from .config import OutputConfig
from .logger import get_logger

log = get_logger("parser")

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_KV_RE = re.compile(r"""["']?([A-Za-z_\u4e00-\u9fff][\w\u4e00-\u9fff]*)["']?\s*[:：]\s*(".*?"|'.*?'|-?\d+\.?\d*|true|false|null)""")


class ParseError(Exception):
    pass


def extract_json(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """返回 (解析出的字典或 None, 解析层级标记)。"""
    if not text or not text.strip():
        return None, "empty"

    s = text.strip()

    # L1：直接解析
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, "json"
    except Exception:
        pass

    # L2：剥离代码围栏
    m = _FENCE_RE.search(s)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                return obj, "fence"
        except Exception:
            pass

    # L3：花括号配对扫描
    obj = _scan_balanced_object(s)
    if obj is not None:
        return obj, "scan"

    # L4：正则兜底
    obj = _regex_kv(s)
    if obj:
        return obj, "regex"

    return None, "failed"


def _scan_balanced_object(s: str) -> Optional[Dict[str, Any]]:
    """扫描第一个 '{' 起的完整 JSON 对象（正确处理字符串内的花括号与转义）。"""
    starts = [i for i, ch in enumerate(s) if ch == "{"]
    for start in starts:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = s[start:i + 1]
                    try:
                        obj = json.loads(cand)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
        # 该起点失败，尝试下一个 '{'
    return None


def _regex_kv(s: str) -> Dict[str, Any]:
    """最后兜底：抽出 key: value 形式的字段（字符串去引号，数字转数值）。"""
    out: Dict[str, Any] = {}
    for m in _KV_RE.finditer(s):
        key, raw = m.group(1), m.group(2)
        val: Any = raw
        if raw[:1] in ('"', "'") and raw[-1:] == raw[:1] and len(raw) >= 2:
            val = raw[1:-1]
        elif re.fullmatch(r"-?\d+", raw):
            val = int(raw)
        elif re.fullmatch(r"-?\d+\.\d+", raw):
            val = float(raw)
        elif raw in ("true", "false"):
            val = raw == "true"
        elif raw == "null":
            val = None
        out.setdefault(key, val)
    return out


# ---------------- 结构校验 ----------------

_TYPE_MAP = {
    "str": str, "string": str,
    "int": int, "float": float, "number": (int, float),
    "list": list, "dict": dict, "bool": bool,
}


def validate(obj: Dict[str, Any], cfg: OutputConfig) -> Tuple[bool, Optional[str]]:
    """按 OutputConfig 校验字段完整性与类型，并检查分值域。"""
    if cfg.format != "json":
        return True, None

    missing = [k for k in cfg.required_keys if k not in obj]
    if missing:
        return False, f"缺少字段: {missing}"

    for key, tname in cfg.key_types.items():
        if key not in obj:
            continue
        expected = _TYPE_MAP.get(str(tname).lower())
        if expected and not isinstance(obj[key], expected):
            return False, f"字段 {key} 类型应为 {tname}，实际为 {type(obj[key]).__name__}"

    if cfg.score_range and len(cfg.score_range) == 2:
        lo, hi = cfg.score_range
        score = _first_numeric(obj)
        if score is None:
            return False, "未找到可校验的分值字段"
        if not (lo <= score <= hi):
            return False, f"分值 {score} 超出允许区间 [{lo}, {hi}]"

    return True, None


def _first_numeric(obj: Dict[str, Any]) -> Optional[float]:
    for k in ("score", "rating", "分数", "得分"):
        v = obj.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            m = re.search(r"-?\d+\.?\d*", v)
            if m:
                return float(m.group())
    return None


def parse_response(text: str, cfg: Optional[OutputConfig] = None,
                   case_id: str = "") -> Tuple[Optional[Dict[str, Any]], bool, Optional[str], str]:
    """统一入口：解析 + 校验。返回 (parsed, ok, error, level)。"""
    cfg = cfg or OutputConfig()
    obj, level = extract_json(text)

    if obj is None:
        if cfg.strict:
            return None, False, "strict 模式下未解析到 JSON", level
        return None, False, "未解析到结构化输出", level

    ok, err = validate(obj, cfg)
    if not ok:
        return obj, False, err, level

    obj["_meta"] = {"parsed_by": level, "case_id": case_id}
    if level in ("regex", "scan", "fence"):
        log.debug(f"[{case_id}] 非标准 JSON 输出，已通过 {level} 策略修复")
    return obj, True, None, level
