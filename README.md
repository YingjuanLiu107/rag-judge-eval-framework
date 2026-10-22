# 基于 LLM-as-a-Judge 的 RAG 检索增强与长文本抗干扰自动化评测框架

独立开发的自动化压测引擎，用于评估 LLM 在 RAG 场景与长文本复杂上下文下的鲁棒性。

## 功能模块

1. **自动化评测引擎（Python & API）**
   - 封装 OpenAI / Qwen 等 API 接口
   - Prompt 批量注入、并发调用与断点重试机制
   - 基于 JSON + 正则的强约束格式化输出解析（如模型返回的 JSON 打分）

2. **多维度测试集构建（Dataset & OOD）**
   - RAG 场景 100+ 测试集
   - "大海捞针"（Needle in a Haystack）长文本测试集
   - "多文档冲突"陷阱题
   - 指令注入攻击（Prompt Injection）防御测试

3. **Judge 模型偏差校准与一致性评估（Data Analysis）**
   - 强模型（GPT-4 / Qwen-Max）作为裁判（LLM-as-a-Judge）
   - 定位裁判模型的"位置偏差"与"长度偏好"
   - 交换测试法（Swap Test）与多 Judge 投票机制
   - Cohen's Kappa 一致性系数校准至 0.85 以上

4. **竞品横向对比与能力雷达（Insights）**
   - 跑测对比 3 款主流大模型（如 GPT-3.5, Qwen-Max, GLM-4）
   - 自动生成能力雷达图与"成本-性能（Token 消耗 vs 准确率）"折线图
   - RAG 场景结构化分析报告

## 目录结构

```
rag-judge-eval-framework/
├── src/          # 核心源码（引擎、Judge、解析、可视化）
├── configs/      # API 配置与评测方案配置
├── datasets/     # 测试集（基础问答 / 大海捞针 / 冲突陷阱 / 注入攻击）
├── results/      # 评测结果与生成的图表报告
├── scripts/      # 运行入口脚本
└── docs/         # 设计文档与分析报告
```

## 快速开始

```bash
pip install -r requirements.txt
# 在 configs/ 下配置 API Key 后
python scripts/run_eval.py --config configs/eval_config.yaml
```

## License

MIT
