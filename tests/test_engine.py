# -*- coding: utf-8 -*-
"""引擎自检：不依赖任何 API Key，覆盖解析容错、并发跑测、断点续跑、成本统计。

运行：python tests/test_engine.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.client import LLMClient  # noqa: E402
from src.core.config import OutputConfig, PromptConfig, RunConfig  # noqa: E402
from src.core.parser import extract_json, parse_response  # noqa: E402
from src.core.runner import EvalRunner, build_messages  # noqa: E402
from src.core.schema import EvalCase  # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {extra}")


def test_parser():
    print("\n[1] 强约束输出解析")
    cases = [
        ('{"answer": "A", "score": 4}', "json"),
        ('好的：\n```json\n{"answer": "B", "score": 3}\n```\n完毕', "fence"),
        ('前面废话 {"answer": "C", "score": 5} 后面废话', "scan"),
        ('答案：D，score: 2', "regex"),
    ]
    for text, expect_level in cases:
        obj, level = extract_json(text)
        check(f"解析层级={level}（期望{expect_level}）", level == expect_level, f"-> {obj}")

    obj, ok, err, _ = parse_response('{"score": 9}', OutputConfig(required_keys=["score"], score_range=[1, 5]))
    check("分值越界被拦截", (not ok) and "超出" in (err or ""), f"err={err}")

    obj, ok, err, _ = parse_response('{"answer": "x"}', OutputConfig(required_keys=["answer", "citations"]))
    check("缺字段被拦截", (not ok) and "缺少" in (err or ""), f"err={err}")

    obj, ok, _, _ = parse_response('{"answer": 123}', OutputConfig(required_keys=["answer"], key_types={"answer": "str"}))
    check("类型校验生效", not ok)


def test_messages():
    print("\n[2] Prompt 渲染")
    case = EvalCase(case_id="t1", question="试用期多久？", context="三年以上不得超过六个月。",
                    metadata={"needle": "秘密数字是 42"})
    cfg = PromptConfig(system="SYS", template="资料:{context}\n问题:{question}\n针:{needle}", response_hint="输出JSON")
    msgs = build_messages(case, cfg)
    check("system 消息存在", msgs[0]["content"] == "SYS")
    check("占位符全部替换", "{context}" not in msgs[1]["content"] and "42" in msgs[1]["content"])
    check("response_hint 追加", msgs[1]["content"].endswith("输出JSON"))


def test_runner():
    print("\n[3] 并发跑测 / 断点续跑 / 成本统计")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        client = LLMClient(name="mock-model", model_id="mock", mock=True,
                           pricing={"prompt_per_1k": 0.01, "completion_per_1k": 0.02})
        run_cfg = RunConfig(output_dir="results", run_id="test-run", resume=True, concurrency=4)
        prompt_cfg = PromptConfig(system="S", template="{question}")
        out_cfg = OutputConfig(required_keys=["answer"], key_types={"answer": "str"})

        cases = [EvalCase(case_id=f"c{i:03d}", question=f"问题{i}", category="rag_qa") for i in range(30)]

        r1 = EvalRunner(client, run_cfg, prompt_cfg, out_cfg, project_root=root)
        res1 = r1.run(cases)
        check("全部用例执行完成", len(res1) == 30, f"实际 {len(res1)}")
        check("全部调用成功", all(r.response.ok for r in res1))
        check("解析全部成功（脏输出被修复）", all(r.parse_ok for r in res1))

        jsonl = r1.jsonl_path
        check("jsonl 落盘行数正确", sum(1 for _ in open(jsonl, encoding="utf-8")) == 30)

        s = r1.summarize(r1.load_results())
        check("汇总成功率=100%", s.success_rate if hasattr(s, "success_rate") else s.success / s.total == 1.0)
        d = s.to_dict()
        check("成功率字段计算正确", d["success_rate"] == 1.0, json.dumps(d, ensure_ascii=False)[:200])
        check("成本 > 0", d["total_cost"] > 0, f"cost={d['total_cost']}")
        check("p95 延迟 >= 平均延迟", d["p95_latency_ms"] >= d["avg_latency_ms"])

        # 断点续跑：已完成用例不应被重复调用，jsonl 行数保持 30
        before = sum(1 for _ in open(jsonl, encoding="utf-8"))
        r2 = EvalRunner(client, run_cfg, prompt_cfg, out_cfg, project_root=root)
        check("续跑识别出全部已完成用例", len(r2._done_ids) == 30, f"done={len(r2._done_ids)}")
        r2.run(cases)
        after = sum(1 for _ in open(jsonl, encoding="utf-8"))
        check("断点续跑未重复写入", before == after == 30, f"before={before} after={after}")

        # 关闭续跑
        run_cfg_n = RunConfig(output_dir="results", run_id="test-run-2", resume=False, concurrency=4)
        r3 = EvalRunner(client, run_cfg_n, prompt_cfg, out_cfg, project_root=root)
        res3 = r3.run(cases)
        check("关闭续跑后全量重跑", len(res3) == 30)


def main() -> int:
    test_parser()
    test_messages()
    test_runner()
    print(f"\n===== 结果：PASS={PASS} FAIL={FAIL} =====")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
