"""파랑 굽힘 모멘트 — IACS UR S11 정본 + 표준파 준정적 교차검증.

IACS: 선급 통일 규칙 설계 파랑 굽힘 (극치 통계 내장, 북대서양
10⁻⁸ 확률 수준). 적용 범위 밖(소형선)은 정직 거절 — 소형 종강도는
quasi_static_wave_moment(전 크기 유효)로.

원전: references/IACS_UR_S11.pdf Rev.9 2019 (확보 2026-08-09,
스펙 §3 인덱스 — p3 S11.2.2.1 공식 대조 일치).
"""
from __future__ import annotations

import math

import numpy as np

from src.physics.structure.loads import (
    G_ACC,
    RHO_SEAWATER,
    _cumtrapz,
    station_area,
    still_water_curves,
    weight_linear_density,
)

IACS_L_MIN = 90.0      # 원전 S11.1: 강선 L ≥ 90 m
IACS_L_MAX = 500.0     # 파랑계수 C 구간식 상한


class IACSRangeError(ValueError):
    """UR S11 적용 범위 밖 — 소형선은 표준파 준정적으로."""


def iacs_wave_coefficient(l_m: float) -> float:
    """파랑계수 C — UR S11 p3 구간식 (원전 대조 박제)."""
    if not (IACS_L_MIN <= l_m <= IACS_L_MAX):
        raise IACSRangeError(
            f"L {l_m:.1f} m는 UR S11 범위({IACS_L_MIN:.0f}~"
            f"{IACS_L_MAX:.0f} m) 밖 — quasi_static_wave_moment 사용.")
    if l_m <= 300.0:
        return 10.75 - ((300.0 - l_m) / 100.0) ** 1.5
    if l_m <= 350.0:
        return 10.75
    return 10.75 - ((l_m - 350.0) / 150.0) ** 1.5


def iacs_wave_bending_knm(l_m: float, b_m: float,
                          cb: float) -> tuple[float, float]:
    """미드십 설계 파랑 굽힘 (호깅 +, 새깅 −) [kN·m] — UR S11.

    분포계수 M = 1 (미드십, Fig.2). Cb 하한 0.6 클램프는 원전
    명시 (p3: "not to be taken less than 0.6")."""
    cw = iacs_wave_coefficient(l_m)
    cb_eff = max(cb, 0.60)
    hog = 0.19 * cw * l_m ** 2 * b_m * cb_eff
    sag = -0.11 * cw * l_m ** 2 * b_m * (cb_eff + 0.7)
    return hog, sag
