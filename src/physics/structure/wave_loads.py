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


def quasi_static_wave_moment(mesh, draft: float, blocks,
                             wave_amp: float, wavelength: float,
                             crest_mid: bool = True,
                             n: int = 101) -> dict:
    """표준파 준정적 굽힘 — 정현 파면에 배를 정적으로 얹고
    (침하 이분법 재평형, 트림 보정은 생략·기록) 부력 재적분.

    고전 표준파 계산법 (IACS 이전 세대 정통) — 전 크기 유효,
    IACS(대형 전용)·스트립(동적)과 3중 교차검증 축.
    반환 m_wave_mid_nm = 총 모멘트 − 정수 모멘트 (파랑 성분만,
    부호 관례 M>0 호깅)."""
    (xmin, _, _), (xmax, _, _) = mesh.bounds
    xs = np.linspace(xmin, xmax, n)
    xmid = 0.5 * (xmin + xmax)
    total_w = sum(m for m, _, _ in blocks) * G_ACC
    phase = 0.0 if crest_mid else math.pi
    wave = wave_amp * np.cos(
        2.0 * math.pi * (xs - xmid) / wavelength + phase)

    def buoy_curve(delta: float) -> np.ndarray:
        wl = draft + delta + wave
        areas = [station_area(mesh, x, float(z))
                 for x, z in zip(xs, wl)]
        return np.array(areas) * RHO_SEAWATER * G_ACC

    lo, hi = -abs(wave_amp) - 0.5, abs(wave_amp) + 0.5
    for _ in range(60):                     # 침하 이분법
        mid = 0.5 * (lo + hi)
        if float(np.trapezoid(buoy_curve(mid), xs)) < total_w:
            lo = mid
        else:
            hi = mid
    sinkage = 0.5 * (lo + hi)
    b = buoy_curve(sinkage)
    integ_b = float(np.trapezoid(b, xs))
    scale = total_w / integ_b if integ_b > 0 else 1.0
    b = b * scale

    w = weight_linear_density(xs, blocks)
    shear = _cumtrapz(w - b, xs)
    moment = _cumtrapz(shear, xs)
    ramp = (xs - xs[0]) / (xs[-1] - xs[0])
    moment = moment - moment[-1] * ramp

    still = still_water_curves(mesh, draft, blocks, n=n)
    i_mid = n // 2
    return {
        "m_total_mid_nm": float(moment[i_mid]),
        "m_wave_mid_nm": float(moment[i_mid]
                               - still.moment_nm[i_mid]),
        "sinkage_m": sinkage,
        "buoy_scale": scale,
        "note": "준정적 (동적 증폭 없음)·트림 재평형 생략 — "
                "스트립 동적과 교차검증",
    }
