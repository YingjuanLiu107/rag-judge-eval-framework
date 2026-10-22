# 03 · Judge 偏差校准与一致性评估（Data Analysis）

> 状态：已完成 ✅ ｜ 目标：二次加权 Cohen's Kappa ≥ 0.85

## 1. 校准流程

| 步骤 | 做法 | 产出 |
|---|---|---|
| ① 多裁判打分 | 2 个裁判模型对同一批答案各打 1-5 分 | `results/judge/scores_*.jsonl` |
| ② 一致性计算 | Cohen's Kappa / 二次加权 Kappa / 一致率 / Spearman | `results/judge/aggregate_*.json` |
| ③ 位置偏差 | 交换测试（Swap Test）：同一对答案按 A-B 与 B-A 各评一次 | `results/judge/swap_*.json` |
| ④ 长度偏好 | 统计裁判选中「更长答案」的比例 | 同上 |
| ⑤ 规则交叉验证 | 确定性规则 vs Judge 判定，列出不一致样本 | `aggregate.rule_gap_*` |

## 2. 抗偏差的 Prompt 设计（`src/judge/prompts.py`）

- 明确写入「答案的顺序、长度、语气都与质量无关」，直接抑制位置偏差与长度偏好；
- 要求先写证据要点再给分（CoT），减少随手打分；
- 分值锚点按类别定制：RAG 看事实一致性，冲突题看「是否识别冲突」，注入题看「是否执行注入指令」；
- 不可答题锚点单独说明：参考答案为「资料中未提及」时，只有明确说不知道才给 5 分。

## 3. 指标口径

- **Cohen's Kappa** = (Po − Pe) / (1 − Pe)，未加权只看"是否同分"；
- **二次加权 Kappa** 对差 2 分以上的分歧加重惩罚，更适合 1-5 分 ordinal 打分（本项目主指标）；
- **判定带**：≥0.85 达到目标；0.80-0.85 需微调 Prompt；0.60-0.80 需优化；<0.40 不建议直接采信；
- **位置偏差率** = 交换顺序后仍然选择「当前排在第一位」的比例（越高越说明裁判被顺序带跑）；
- **金标准偏好率** = 模型答案 vs 参考答案对比中，裁判选中参考答案的比例（反映裁判的绝对判准能力）。

## 4. 运行方式

```bash
# 多裁判打分 + 一致性
python scripts/run_judge.py --results results/raw/full__qwen-max.jsonl --run-id full

# 交换测试（位置偏差 / 长度偏好），建议先跑 40 条看趋势
python scripts/run_judge.py --results results/raw/full__qwen-max.jsonl --mode swap --limit 40 --run-id swap

# 同一裁判重复采样测自洽（temperature > 0）
python scripts/run_judge.py --results xxx.jsonl --repeat 2 --judge-temperature 0.7
```

## 5. 已知局限（如实标注）

- 本轮两个裁判（qwen-max / qwen-plus）**来自同一厂商**，同源偏好会让 Kappa 被高估；
- 建议在补齐其他厂商 Key 后，用不同源裁判（如 GPT-4o）做交叉验证；
- Judge 与规则判定不一致的样本（`rule_gap`）是 Prompt 迭代的第一优先级。
