# -*- coding: utf-8 -*-
"""Judge 一致性度量：Cohen's Kappa、加权 Kappa、Spearman、位置/长度偏差。

全部纯 Python 实现（不依赖 sklearn），保证在最小环境也能跑。

口径说明：
- Cohen's Kappa = (Po - Pe) / (1 - Pe)，Po 为观测一致率，Pe 为期望一致率；
- 线性加权 Kappa：|i-j| 权重；二次加权 Kappa：|i-j|^2 权重（打分差越大惩罚越重）；
- 判断阈值采用常用经验值：>0.8 几乎完美，0.6-0.8 高度一致，0.4-0.6 中等（本项目目标 ≥0.85）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

KAPPA_BANDS = [
    (0.85, "几乎完美（达到本项目目标）"),
    (0.80, "接近目标，可微调 Prompt"),
    (0.60, "高度一致，但需优化"),
    (0.40, "中等一致，需重点校准"),
    (0.00, "一致性不足，不建议直接采信"),
]


def kappa_band(k: float) -> str:
    for threshold, desc in KAPPA_BANDS:
        if k >= threshold:
            return desc
    return "异常值（k < 0，系统性分歧）"


def confusion_matrix(r1: Sequence[int], r2: Sequence[int], k: int) -> List[List[int]]:
    m = [[0] * k for _ in range(k)]
    for a, b in zip(r1, r2):
        m[a - 1][b - 1] += 1
    return m


def cohens_kappa(r1: Sequence[int], r2: Sequence[int], k: int = 5,
                 weights: Optional[str] = None) -> float:
    """weights: None=未加权, 'linear'=线性加权, 'quadratic'=二次加权。分值需为 1..k。"""
    pairs = [(a, b) for a, b in zip(r1, r2) if a and b]
    n = len(pairs)
    if n == 0:
        return 0.0
    m = confusion_matrix([p[0] for p in pairs], [p[1] for p in pairs], k)

    def w(i: int, j: int) -> float:
        if weights is None:
            return 0.0 if i == j else 1.0
        d = abs(i - j)
        return d if weights == "linear" else d ** 2

    row = [sum(m[i]) for i in range(k)]
    col = [sum(m[j][j_i] for j in range(k)) for j_i in range(k)]
    total = sum(row)
    if total == 0:
        return 0.0

    num = sum(w(i, j) * m[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * (row[i] * col[j] / total) for i in range(k) for j in range(k))
    if den == 0:
        return 1.0
    po = 1 - num / (total * (k - 1)) if weights else sum(m[i][i] for i in range(k)) / total
    # 统一按加权公式计算，未加权时 w 为 0/1 指示函数
    if weights is None:
        pe = sum(row[i] * col[i] for i in range(k)) / (total ** 2)
        return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0
    po_w = 1 - num / (total * (k - 1))
    pe_w = 1 - den / (total * (k - 1))
    return round((po_w - pe_w) / (1 - pe_w), 4) if pe_w < 1 else 1.0


def agreement_rate(r1: Sequence[int], r2: Sequence[int]) -> float:
    pairs = [(a, b) for a, b in zip(r1, r2) if a and b]
    if not pairs:
        return 0.0
    return round(sum(1 for a, b in pairs if a == b) / len(pairs), 4)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """无第三方依赖的斯皮尔曼相关（并列取平均秩）。"""
    pairs = [(a, b) for a, b in zip(x, y)]
    if len(pairs) < 2:
        return 0.0

    def ranks(vals: List[float]) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r

    rx = ranks([p[0] for p in pairs])
    ry = ranks([p[1] for p in pairs])
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(num / (dx * dy), 4) if dx and dy else 0.0


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) ** 0.5) * (sum((b - my) ** 2 for b in y) ** 0.5)
    return round(num / den, 4) if den else 0.0


# ---------------- 偏差诊断 ----------------

def position_bias(swap_records: Sequence[Dict]) -> Dict[str, float]:
    """交换测试（Swap Test）的位置偏差统计。

    swap_records: 每条含 {'winner_orig': 'A'/'B'/'tie', 'winner_swapped': 'A'/'B'/'tie'}
    两条都指同一个候选答案时记为一致；都选"先出现的位置"即为位置偏差。
    """
    n = len(swap_records)
    if n == 0:
        return {"n": 0, "consistency": 0.0, "position_bias_rate": 0.0,
                "tie_rate": 0.0, "divergent_rate": 0.0}

    consistent, first_pos, ties, divergent = 0, 0, 0, 0
    for r in swap_records:
        w1 = str(r.get("winner_orig") or "").upper()
        w2 = str(r.get("winner_swapped") or "").upper()
        if w1 in ("A", "B") and w2 in ("A", "B"):
            # 交换后 A/B 标签对调，选同一个实体才算一致
            if w1 != w2:
                consistent += 1
            else:
                first_pos += 1  # 两次都选了"当前排在第一位"的，说明被位置带跑
        elif w1 == "TIE" and w2 == "TIE":
            consistent += 1
            ties += 1
        else:
            divergent += 1  # 一次给了明确胜负、另一次判平局或解析失败

    return {
        "n": n,
        "consistency": round(consistent / n, 4),
        "position_bias_rate": round(first_pos / n, 4),
        "tie_rate": round(ties / n, 4),
        "divergent_rate": round(divergent / n, 4),
    }


def length_bias(pref_longer: Sequence[int]) -> Dict[str, float]:
    """长度偏好：judge 选中「更长答案」的比例，>0.5 说明存在长度偏好。"""
    n = len(pref_longer)
    if n == 0:
        return {"n": 0, "prefer_longer_rate": 0.0}
    return {"n": n, "prefer_longer_rate": round(sum(pref_longer) / n, 4)}


def score_stats(scores: Sequence[int]) -> Dict[str, float]:
    vals = [s for s in scores if s]
    if not vals:
        return {"n": 0, "mean": 0.0, "std": 0.0, "dist": {}}
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    dist = {str(i): round(vals.count(i) / len(vals), 4) for i in range(1, 6)}
    return {"n": len(vals), "mean": round(mean, 3), "std": round(std, 3), "dist": dist}
