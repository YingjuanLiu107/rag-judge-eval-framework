# -*- coding: utf-8 -*-
"""生成全部测试集 -> datasets/*.jsonl

用法：
    python scripts/build_datasets.py              # 全部生成
    python scripts/build_datasets.py --only rag_qa needle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.generators import BUILDERS, save_jsonl, stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "datasets"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="生成多维度测试集")
    p.add_argument("--only", nargs="*", default=None, help="只生成指定类别")
    p.add_argument("--out", default=str(OUT_DIR))
    args = p.parse_args(argv)

    names = args.only or list(BUILDERS)
    OUT = Path(args.out)
    total = 0
    print(f"{'类别':<12}{'条数':>6}  输出文件")
    print("-" * 60)
    for name in names:
        if name not in BUILDERS:
            print(f"[WARN] 未知类别: {name}")
            continue
        cases = BUILDERS[name]()
        path = save_jsonl(cases, OUT / f"{name}.jsonl")
        s = stats(cases)
        total += s["total"]
        print(f"{name:<12}{s['total']:>6}  {path}")

    manifest = {n: stats(BUILDERS[n]())["total"] for n in names}
    manifest["_total"] = total
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-" * 60)
    print(f"合计 {total} 条，清单已写入 {OUT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
