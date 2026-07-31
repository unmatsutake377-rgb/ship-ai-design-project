"""Michell 보정 계수 (스펙 §4) — 능동 학습 1회전의 '학습' 부분.

모형: ratio_w(B/L) = 1 + b·(B/L). 절편은 물리 앵커(B/L→0에서 Michell
정확 → ratio=1)로 고정 — 소표본(4점)에서 매개변수는 b 하나만 배운다.
앵커드 최소제곱: b = Σ(ratio_i − 1)·x_i / Σ x_i²  (x = B/L).

CFD 조파 추정 = P_자유수면 − P_이중모형: 같은 척을 두 모드로 돌려
형상(점성 압력) 성분을 빼내면 파도 몫만 남는다.
"""
from __future__ import annotations

import pandas as pd

RATIO_CLIP = (0.05, 1.5)   # 외삽 폭주 방지 (스펙 §4)


def fit_wave_ratio(points: list[tuple[float, float]]) -> float:
    num = sum((r - 1.0) * x for x, r in points)
    den = sum(x * x for x, _ in points)
    return num / den


def wave_ratio(bl: float, b: float,
               lo: float = RATIO_CLIP[0], hi: float = RATIO_CLIP[1]) -> float:
    return min(hi, max(lo, 1.0 + b * bl))


def ratios_from_labels(df: pd.DataFrame) -> list[tuple[float, float]]:
    """라벨 CSV에서 (B/L, ratio) 점 추출 — _simple_/_inter_ 짝 필수."""
    points = []
    df = df.dropna(subset=["loa_m", "beam_m"])
    bases: dict[str, dict] = {}
    for _, row in df.iterrows():
        if "_simple_" in row.case_name:
            base = row.case_name.replace("_simple_", "_")
            bases.setdefault(base, {})["simple"] = row
        elif "_inter_" in row.case_name:
            base = row.case_name.replace("_inter_", "_")
            bases.setdefault(base, {})["inter"] = row
    for base, pair in sorted(bases.items()):
        if "simple" not in pair or "inter" not in pair:
            continue
        s, i = pair["simple"], pair["inter"]
        ratio = (i.cfd_pressure_n - s.cfd_pressure_n) / i.emp_rw_n
        points.append((float(i.beam_m / i.loa_m), float(ratio)))
    return points
