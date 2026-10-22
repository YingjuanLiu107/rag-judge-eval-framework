# -*- coding: utf-8 -*-
"""测试集模块：第二步将在此扩展「大海捞针 / 多文档冲突 / 指令注入」生成器。"""

from .loader import DatasetError, load_cases, load_from_config

__all__ = ["load_cases", "load_from_config", "DatasetError"]
