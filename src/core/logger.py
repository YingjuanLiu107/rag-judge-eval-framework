# -*- coding: utf-8 -*-
"""轻量日志：控制台 + 文件双输出，避免引入重型日志框架。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}


class Logger:
    def __init__(self, name: str = "rag-eval", level: str = "INFO",
                 log_file: Optional[Path] = None):
        self.name = name
        self.level = _LEVELS.get(level.upper(), 20)
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _emit(self, level: str, msg: str) -> None:
        if _LEVELS[level] < self.level:
            return
        line = f"[{datetime.now():%H:%M:%S}] {level:<5} {self.name} | {msg}"
        stream = sys.stderr if level in ("WARN", "ERROR") else sys.stdout
        print(line, file=stream)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {level:<5} {msg}\n")

    def debug(self, msg: str) -> None:
        self._emit("DEBUG", msg)

    def info(self, msg: str) -> None:
        self._emit("INFO", msg)

    def warn(self, msg: str) -> None:
        self._emit("WARN", msg)

    def error(self, msg: str) -> None:
        self._emit("ERROR", msg)


_DEFAULT = Logger()


def get_logger(name: str = "rag-eval", level: str = "INFO",
               log_file: Optional[Path] = None) -> Logger:
    """默认返回全局 logger；传入 log_file 时返回带文件输出的独立实例。"""
    if log_file is None and name == "rag-eval":
        _DEFAULT.level = _LEVELS.get(level.upper(), 20)
        return _DEFAULT
    return Logger(name=name, level=level, log_file=log_file)
