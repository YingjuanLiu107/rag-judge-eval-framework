# -*- coding: utf-8 -*-
"""评测主入口：加载配置 -> 加载测试集 -> 并发调用 -> 落盘明细与汇总。

用法示例：
    # 离线冒烟（不消耗 token）
    python scripts/run_eval.py --mock

    # 真实跑测
    python scripts/run_eval.py --model qwen-max --concurrency 8

    # 多模型对比（第四步竞品雷达图的数据来源）
    python scripts/run_eval.py --model qwen-max glm-4 gpt-4o-mini

    # 只跑前 20 条 / 指定 run_id
    python scripts/run_eval.py --model qwen-max --limit 20 --run-id exp1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.client import LLMClient  # noqa: E402
from src.core.config import EvalConfig, load_dotenv  # noqa: E402
from src.core.logger import get_logger  # noqa: E402
from src.core.runner import EvalRunner  # noqa: E402
from src.datasets import load_from_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
log = get_logger("run_eval")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RAG / 长文本自动化评测引擎")
    p.add_argument("--models-config", default=str(ROOT / "configs" / "models.yaml"))
    p.add_argument("--eval-config", default=str(ROOT / "configs" / "eval_config.yaml"))
    p.add_argument("--model", nargs="+", default=None, help="模型名，可多个；缺省用配置里的第一个")
    p.add_argument("--dataset", default=None, help="覆盖配置中的测试集路径（单文件）")
    p.add_argument("--category", default=None, help="配合 --dataset 指定类别")
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--no-resume", action="store_true", help="忽略已有进度，全部重跑")
    p.add_argument("--mock", action="store_true", help="离线模拟，不发起真实请求")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.verbose:
        log.level = 10

    # 优先加载项目根目录 .env 中的 API Key（.env 不进仓库）
    load_dotenv(ROOT / ".env")
    cfg = EvalConfig.load(args.models_config, args.eval_config, env_path=ROOT / ".env")
    if args.concurrency:
        cfg.run.concurrency = args.concurrency
    if args.limit:
        cfg.run.limit = args.limit
    if args.run_id:
        cfg.run.run_id = args.run_id
    if args.no_resume:
        cfg.run.resume = False
    if args.mock:
        cfg.run.mock = True

    # 测试集
    if args.dataset:
        from src.datasets import load_cases
        cases = load_cases(args.dataset, category=args.category or "custom")
    else:
        cases = load_from_config(cfg.datasets, ROOT)
    if not cases:
        log.error("没有加载到任何测试用例，请检查 configs/eval_config.yaml 的 datasets 配置")
        return 1
    log.info(f"加载测试用例 {len(cases)} 条，类别分布：{_dist(cases)}")

    model_names = args.model or ([cfg.models[0].name] if cfg.models else [])
    if not model_names:
        log.error("configs/models.yaml 中没有配置任何模型")
        return 1

    summaries = []
    for name in model_names:
        mcfg = cfg.get_model(name)
        if cfg.run.mock:
            mcfg.api_key = ""
        client = LLMClient(**mcfg.to_client_kwargs(), mock=cfg.run.mock)
        runner = EvalRunner(client, cfg.run, cfg.prompt, cfg.output, project_root=ROOT)
        runner.run(cases)
        summaries.append(runner.summarize(runner.load_results()).to_dict())

    print("\n===== 本轮跑测汇总 =====")
    for s in summaries:
        print(f"model={s['model']:<16} 成功率={s['success_rate']:.2%} "
              f"解析成功={s['parse_ok_rate']:.2%} 平均延迟={s['avg_latency_ms']:.0f}ms "
              f"tokens={s['total_tokens']} 成本={s['total_cost']}")
    return 0


def _dist(cases) -> str:
    d: dict = {}
    for c in cases:
        d[c.category] = d.get(c.category, 0) + 1
    return ", ".join(f"{k}:{v}" for k, v in d.items())


if __name__ == "__main__":
    raise SystemExit(main())
