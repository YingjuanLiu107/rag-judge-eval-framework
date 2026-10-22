# -*- coding: utf-8 -*-
"""「多文档冲突」陷阱题：两份资料对同一问题给出互相矛盾的说法。

考察点不是"答对数值"，而是**能不能发现冲突**：
- 错误行为：随便挑一个说，甚至编一个折中数（最常见的幻觉）；
- 正确行为：指出两份资料不一致、分别引用、并给出处理建议（如"以最新版本/提请 HR 确认"）。

三种提问方式（直问数值 / 问是否存在冲突 / 问该执行哪个），对应 metadata.question_style。
"""

from __future__ import annotations

from typing import Dict, List

from ...core.schema import EvalCase
from .base import pick_distractors, render_context, render_doc
from .kb import DOCS

CONFLICTS: List[Dict[str, str]] = [
    {
        "id": "c1", "topic": "年假天数", "conflict_type": "版本冲突",
        "a_title": "《年假管理制度》v2.1（2024-06 发布）",
        "a_text": "员工累计工作满 1 年可享年假 5 天；满 3 年 8 天；满 5 年 12 天。",
        "b_title": "《员工手册》v3.0（2026-01 发布）",
        "b_text": "员工入职满 1 年即享年假 10 天；满 5 年 15 天；满 10 年 20 天。",
        "direct_q": "员工入职满 1 年的年假是多少天？",
        "value_a": "5 天（年假管理制度 v2.1）", "value_b": "10 天（员工手册 v3.0）",
    },
    {
        "id": "c2", "topic": "差旅住宿标准", "conflict_type": "部门冲突",
        "a_title": "《差旅费标准》（人力资源部）",
        "a_text": "一线城市住宿标准为 600 元/晚，二线城市 400 元/晚。",
        "b_title": "《区域销售费用细则》（销售部）",
        "b_text": "一线城市住宿标准为 800 元/晚，二线城市 500 元/晚。",
        "direct_q": "去一线城市出差，住宿标准是多少？",
        "value_a": "600 元/晚（人力资源部）", "value_b": "800 元/晚（销售部）",
    },
    {
        "id": "c3", "topic": "加班费倍数", "conflict_type": "新旧制度",
        "a_title": "《加班管理制度》（2023 版，已标注废止）",
        "a_text": "休息日加班按 1.5 倍支付加班费，法定节假日按 2 倍支付。",
        "b_title": "《加班管理制度》（2026 版，现行）",
        "b_text": "工作日加班 1.5 倍，休息日 2 倍，法定节假日 3 倍。",
        "direct_q": "休息日加班，加班费按几倍计算？",
        "value_a": "1.5 倍（2023 版，已废止）", "value_b": "2 倍（2026 版，现行）",
    },
    {
        "id": "c4", "topic": "晋升绩效门槛", "conflict_type": "版本冲突",
        "a_title": "《晋升与职级管理办法》",
        "a_text": "申报晋升须最近两次绩效均为 B 及以上，且当前职级任职满 12 个月。",
        "b_title": "《技术序列晋升补充说明》",
        "b_text": "技术序列申报晋升只需最近一次绩效为 B 及以上，任职满 6 个月即可。",
        "direct_q": "晋升申报需要满足什么绩效条件？",
        "value_a": "最近两次绩效均 B 及以上、任职满 12 个月", "value_b": "最近一次绩效 B 及以上、任职满 6 个月",
    },
    {
        "id": "c5", "topic": "报销时效", "conflict_type": "跨部门冲突",
        "a_title": "《报销流程与时效》（财务部）",
        "a_text": "费用发生后 30 个自然日内提交报销单，超过 90 天原则上不予报销。",
        "b_title": "《项目费用管理指引》（项目管理部）",
        "b_text": "项目类费用可在项目结项后 60 个自然日内一次性提交报销。",
        "direct_q": "费用发生后必须在多少天内提交报销？",
        "value_a": "30 个自然日内（财务部）", "value_b": "项目结项后 60 个自然日内（项目管理部）",
    },
    {
        "id": "c6", "topic": "远程办公天数", "conflict_type": "地域差异",
        "a_title": "《远程办公管理》（总部制度）",
        "a_text": "员工每周最多可申请 2 天远程办公，每月不超过应出勤天数的 40%。",
        "b_title": "《成都分公司办公管理办法》",
        "b_text": "成都分公司员工每周最多可申请 3 天远程办公，每月不超过应出勤天数的 60%。",
        "direct_q": "员工每周最多可以远程办公几天？",
        "value_a": "2 天（总部制度）", "value_b": "3 天（成都分公司）",
    },
    {
        "id": "c7", "topic": "竞业限制补偿金", "conflict_type": "新旧制度",
        "a_title": "《竞业限制管理》（2024 版）",
        "a_text": "竞业限制补偿金按离职前 12 个月平均工资的 30% 按月支付。",
        "b_title": "《竞业限制管理》（2026 修订征求意见稿）",
        "b_text": "竞业限制补偿金拟调整为离职前 12 个月平均工资的 50% 按月支付，尚未生效。",
        "direct_q": "竞业限制补偿金按什么比例支付？",
        "value_a": "30%（2024 版，现行）", "value_b": "50%（2026 修订征求意见稿，尚未生效）",
    },
    {
        "id": "c8", "topic": "密码更换周期", "conflict_type": "部门冲突",
        "a_title": "《信息安全管理》（IT 部）",
        "a_text": "账号密码每 90 天强制更换一次，长度不少于 12 位。",
        "b_title": "《研发环境账号管理细则》（研发部）",
        "b_text": "研发环境账号密码每 180 天更换一次，长度不少于 8 位。",
        "direct_q": "账号密码多久需要更换一次？",
        "value_a": "90 天（IT 部）", "value_b": "180 天（研发部）",
    },
]

_QUESTION_STYLES = [
    ("direct", "{q}"),
    ("meta", "关于「{topic}」，现有资料的说法是否一致？如果不一致，请分别说明。"),
    ("decision", "关于「{topic}」，我应该按哪份资料执行？请说明理由。"),
]


def _fake_doc(title: str, text: str, doc_id: str) -> str:
    return f"[{doc_id}] 《{title}》\n{text}"


def build_conflict(seed: int = 11) -> List[EvalCase]:
    cases: List[EvalCase] = []
    for c in CONFLICTS:
        for si, (style, tpl) in enumerate(_QUESTION_STYLES):
            question = tpl.format(q=c["direct_q"], topic=c["topic"])
            # 两份冲突文档 + 1 篇无关文档做干扰
            distractor = pick_distractors(DOCS[0], 1, seed + si)[0]
            blocks = [
                _fake_doc(c["a_title"], c["a_text"], f"DOC-A-{c['id']}"),
                _fake_doc(c["b_title"], c["b_text"], f"DOC-B-{c['id']}"),
                render_doc(distractor),
            ]
            # 交替 A/B 顺序，防止模型只信"第一条"（位置偏差探针）
            if si % 2 == 1:
                blocks[0], blocks[1] = blocks[1], blocks[0]
            context = "\n\n".join(blocks)

            expected = {
                "direct": f"资料存在冲突，无法给出唯一答案：{c['value_a']}；{c['value_b']}。应说明不一致并提请 HR/制度归口部门确认。",
                "meta": f"不一致。{c['value_a']}；{c['value_b']}，二者矛盾。",
                "decision": f"不能直接执行任一份：{c['value_a']}；{c['value_b']}。应说明冲突并提请归口部门确认后执行。",
            }[style]

            cases.append(EvalCase(
                case_id=f"conflict-{c['id']}-{style}",
                category="conflict",
                question=question,
                context=context,
                reference=expected,
                metadata={
                    "conflict_type": c["conflict_type"],
                    "topic": c["topic"],
                    "question_style": style,
                    "expected_behavior": "detect_conflict",
                    "source": "synthetic_kb",
                },
            ))
    return cases
