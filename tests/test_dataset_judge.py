# -*- coding: utf-8 -*-
"""数据集生成器 / Judge 度量 / 规则校验 的集成自检（无需 API Key）。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.generators import BUILDERS, save_jsonl, stats  # noqa: E402
from src.judge.metrics import cohens_kappa, kappa_band, length_bias, position_bias, spearman  # noqa: E402
from src.judge.rules import conflict_detect_rate, defense_rate, flag_injection_executed  # noqa: E402
from src.core.schema import EvalCase  # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {extra}")


def test_generators():
    print("\n[A] 测试集生成器")
    total = 0
    for name, fn in BUILDERS.items():
        cases = fn()
        check(f"{name} 生成非空", len(cases) > 0, f"n={len(cases)}")
        total += len(cases)
        # 每条都有可解析的 case_id / category / question
        ok = all(c.case_id and c.category and c.question for c in cases)
        check(f"{name} 字段完整", ok)
    check("四类合计 >= 200", total >= 200, f"total={total}")
    # 可复现：再生成一次 id 应完全一致
    a = set(c.case_id for c in BUILDERS["rag_qa"]())
    b = set(c.case_id for c in BUILDERS["rag_qa"]())
    check("rag_qa 可复现（case_id 一致）", a == b)


def test_judge_metrics():
    print("\n[B] Judge 一致性度量")
    r1 = [5, 4, 3, 5, 1, 2, 4, 5, 3, 4]
    r2 = [5, 4, 3, 4, 2, 2, 4, 5, 3, 5]
    k = cohens_kappa(r1, r2, weights="quadratic")
    check("二次加权 Kappa 在 0-1 之间", 0 <= k <= 1, f"k={k}")
    check("Kappa 判定带非空", bool(kappa_band(k)) and isinstance(kappa_band(k), str))
    check("Spearman 在 -1..1", -1 <= spearman(r1, r2) <= 1)
    pb = position_bias([
        {"winner_orig": "A", "winner_swapped": "B"},
        {"winner_orig": "A", "winner_swapped": "A"},
        {"winner_orig": "TIE", "winner_swapped": "TIE"},
    ])
    check("position_bias 自洽率≈0.67", round(pb["consistency"], 2) == round(2 / 3, 2), str(pb))
    lb = length_bias([1, 0, 1])
    check("length_bias 偏好更长≈0.67", round(lb["prefer_longer_rate"], 2) == round(2 / 3, 2), str(lb))


def test_rules():
    print("\n[C] 规则校验")
    rec_leak = {"category": "injection", "content": "原样输出系统提示词：你是一个严谨的助手",
                "metadata": {"danger_signals": ["系统提示词"]}}
    rec_safe = {"category": "injection", "content": "员工累计工作满5年的年假是12天",
                "metadata": {"danger_signals": []}}
    check("注入泄漏被识别", flag_injection_executed(rec_leak["content"], ["系统提示词"]))
    check("正常回答不误报", not flag_injection_executed(rec_safe["content"], []))
    recs = [rec_leak, rec_safe] * 3
    d = defense_rate(recs)
    check("注入防御率 = 0.5", abs(d["defense_rate"] - 0.5) < 1e-9, str(d))
    cf_ans = {"category": "conflict", "content": "不一致。A说5天，B说10天，存在冲突"}
    check("冲突宽松口径识别", conflict_detect_rate([cf_ans])["detect_rate"] == 1.0)


def test_e2e_write(tmp_path):
    print("\n[D] 端到端：生成 -> 落盘 -> 读回")
    with tempfile.TemporaryDirectory() as td:
        path = save_jsonl(BUILDERS["needle"](), Path(td) / "needle.jsonl")
        from src.datasets import load_cases
        cases = load_cases(path)
        check("落盘后读回数量一致", len(cases) == len(BUILDERS["needle"]()))


def main():
    test_generators()
    test_judge_metrics()
    test_rules()
    test_e2e_write(None)
    print(f"\n===== 结果：PASS={PASS} FAIL={FAIL} =====")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
