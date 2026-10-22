# -*- coding: utf-8 -*-
"""测试集加载：支持 jsonl / json / csv，统一归一化为 EvalCase。

第二步（多维度测试集构建）的生成器会把数据落到 datasets/ 下，这里提供统一读取入口，
保证引擎与数据集解耦：引擎只认 EvalCase。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from ..core.schema import EvalCase

SUPPORTED_SUFFIX = {".jsonl", ".json", ".csv"}


class DatasetError(Exception):
    pass


def load_cases(path: Union[str, Path], category: str = "") -> List[EvalCase]:
    """加载测试集；category 为空时使用文件内的 category 字段，否则统一覆盖。"""
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"测试集不存在: {p}")
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIX:
        raise DatasetError(f"不支持的格式: {suffix}（支持 {SUPPORTED_SUFFIX}）")

    if suffix == ".jsonl":
        raw = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        raw = data if isinstance(data, list) else data.get("cases", [])
    else:
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            raw = list(csv.DictReader(f))

    cases: List[EvalCase] = []
    for i, d in enumerate(raw):
        case = EvalCase.from_dict(d)
        if not case.case_id:
            case.case_id = f"{p.stem}-{i:04d}"
        if category:
            case.category = category
        cases.append(case)
    return cases


def load_from_config(datasets_cfg: List[Dict[str, Any]], root: Union[str, Path]) -> List[EvalCase]:
    """按 eval_config.yaml 的 datasets 列表批量加载。

    配置示例：
      datasets:
        - path: datasets/rag_qa.jsonl
          category: rag_qa      # 可选，缺省用文件内字段
          limit: 50             # 可选
    """
    root = Path(root)
    all_cases: List[EvalCase] = []
    for item in datasets_cfg:
        path = root / item["path"]
        cases = load_cases(path, category=item.get("category", ""))
        limit = item.get("limit")
        if limit:
            cases = cases[: int(limit)]
        all_cases.extend(cases)

    # case_id 去重（跨文件可能重名）
    seen: Dict[str, int] = {}
    for c in all_cases:
        seen[c.case_id] = seen.get(c.case_id, 0) + 1
        if seen[c.case_id] > 1:
            c.case_id = f"{c.case_id}#{seen[c.case_id]}"
    return all_cases
