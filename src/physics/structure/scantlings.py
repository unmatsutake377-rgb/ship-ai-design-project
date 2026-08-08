"""DNV 계보 국부 스캔틀링 (구조 강도 2단계, 스펙 §2·§3 보강 2).

원전: DNV Rules for Ships Pt.3 Ch.1 (2016, ≥100m) — 판 두께
t = 15.8·ka·s·√(p/σ) + tk (p89), 최소 두께 t0 + k·L1/f1 (p89~90),
늑골 Z = 0.63·l²·s·p·wk/f1 (p91). 소형은 Ch.2 대응식 계보 —
20m 미만은 규칙 하한 비례 완화 (C급 정직 표기).

설계 압력: p1 = 10·h0 + pdp (수선 아래, Sec.4 C201 p56) — 파랑
성분은 UR S11 파랑계수 CW 재사용 (같은 계보), ks=2 미드십,
kf 항은 보수 생략.
"""
from __future__ import annotations

import math


def _cw_all_sizes(loa: float) -> float:
    """파랑계수 CW — UR S11 구간식 (L≥90), 소형은 0.0792L 연속
    외삽 (DNV 소형 계보, C급)."""
    if loa >= 90.0:
        from src.physics.structure.wave_loads import iacs_wave_coefficient
        return iacs_wave_coefficient(loa)
    return 0.0792 * loa


def design_pressure_bottom(loa: float, beam: float,
                           draft: float) -> float:
    """선저 설계 압력 [kN/m²] — p1 = 10·T + pdp(z=0).

    pdp = ks·CW − 1.2(T−z), z=0(선저), ks=2 (미드십 0.2~0.7L)."""
    cw = _cw_all_sizes(loa)
    pdp = 2.0 * cw - 1.2 * draft
    return 10.0 * draft + max(pdp, 0.0)


def design_pressure_side(loa: float, beam: float, draft: float,
                         depth: float) -> float:
    """선측 설계 압력 [kN/m²] — 수선(z=T) 평가 + 상부 최소
    6.25+0.025L1 계보 (p56)."""
    cw = _cw_all_sizes(loa)
    p_wl = 2.0 * cw
    return max(p_wl, 6.25 + 0.025 * min(loa, 300.0))


def design_pressure_deck(loa: float) -> float:
    """노천 갑판 최소 압력 [kN/m²] — 원전 'minimum 5' 계보."""
    return max(5.0, 0.025 * min(loa, 300.0) + 1.25)


def _ka(spacing_m: float, span_m: float) -> float:
    """종횡비 보정 ka = (1.1 − 0.25 s/l)², 0.72~1.0 클램프
    (원전 p32)."""
    r = spacing_m / max(span_m, 1e-9)
    ka = (1.1 - 0.25 * r) ** 2
    return min(1.0, max(0.72, ka))


def plate_thickness_mm(pressure_knm2: float, spacing_m: float,
                       span_m: float, sigma_nmm2: float,
                       tk_mm: float = 1.5) -> float:
    """판 두께 [mm] — t = 15.8·ka·s·√(p/σ) + tk (원전 p89)."""
    return (15.8 * _ka(spacing_m, span_m) * spacing_m
            * math.sqrt(pressure_knm2 / sigma_nmm2) + tk_mm)


def min_thickness_mm(loa: float, f1: float, location: str) -> float:
    """최소 두께 [mm] — t0 + k·L1/f1 계보 (원전 p89~90).

    location: 'keel' 7.0+0.05L1 / 'bottom' 5.0+0.04L1 /
    'side'·'deck' 5.0+0.03L1. L1 = min(L, 300). 소형(<20m)은
    상수항 비례 완화 (규칙 하한 과대 — C급 정직, 실선 알루
    소형정 4~6mm 대역 정합)."""
    l1 = min(loa, 300.0)
    base = {"keel": (7.0, 0.05), "bottom": (5.0, 0.04),
            "side": (5.0, 0.03), "deck": (5.0, 0.03)}[location]
    t0, k = base
    if loa < 20.0:
        t0 = t0 * max(loa / 20.0, 0.4)
    return t0 + k * l1 / max(f1, 1e-9)


def stiffener_modulus_cm3(pressure_knm2: float, spacing_m: float,
                          span_m: float, f1: float,
                          wk: float = 1.0) -> float:
    """늑골(보강재) 요구 단면계수 [cm³] — Z = 0.63·l²·s·p·wk/f1
    (원전 p91 계열)."""
    return (0.63 * span_m ** 2 * spacing_m * pressure_knm2 * wk
            / max(f1, 1e-9))


def default_spacing_m(loa: float) -> float:
    """종늑골 간격 통상값 — 대형 0.7m 근방, 소형 비례 축소
    (관행 계보, C급)."""
    return min(0.9, max(0.2, 0.48 + loa / 400.0))
