"""CII 운항 탄소등급 (경제성 캠페인 확장, 스펙 2026-08-09-cii).

원전 (references/, imo.org 공식 — 스펙 §3 인덱스):
- MEPC.352(78) G1: AER = CO₂질량 / (DWT × 운항거리) [gCO₂/(DWT·nm)]
- MEPC.353(78) G2: 2019 기준선 CIIref = a·DWT^{-c} — General cargo
  <20,000 DWT: a=588·c=0.3885 / ≥20,000: a=31948·c=0.792
- MEPC.338(76) G3 (스캔본 이미지 판독): Z = 2023 5%·2024 7%·
  2025 9%·2026 11% — **2027+ 미정** (원전 각주: 추후 강화)
- MEPC.354(78) G4: 등급 경계 exp(d1~d4) — General cargo
  0.83/0.94/1.06/1.19 (비 = attained/required → A~E)

성격: 운항 지표 (설계 게이트 아님) — 성적표·전망 전용. D 3년
연속·E 1년 = 시정계획 의무 (제도 배경, note 병기).
"""
from __future__ import annotations

from src.physics.economics.eedi import CF_MDO

CII_REF_A_SMALL = 588.0        # General cargo < 20,000 DWT (G2)
CII_REF_C_SMALL = 0.3885
CII_REF_A_LARGE = 31948.0      # ≥ 20,000 DWT
CII_REF_C_LARGE = 0.792
DWT_SPLIT = 20_000.0
Z_BY_YEAR = {2023: 5.0, 2024: 7.0, 2025: 9.0, 2026: 11.0}   # G3
DD_GENERAL_CARGO = (0.83, 0.94, 1.06, 1.19)                 # G4


def attained_aer(fuel_t_per_year: float, dwt_t: float,
                 distance_nm_per_year: float,
                 cf_t_co2_per_t: float = CF_MDO) -> dict:
    """운항 실적 AER [gCO₂/(DWT·nm)] — G1."""
    co2_g = fuel_t_per_year * cf_t_co2_per_t * 1e6
    return {"aer_g_per_dwt_nm": co2_g / (max(dwt_t, 1e-9)
                                         * max(distance_nm_per_year,
                                               1e-9)),
            "co2_t_per_year": co2_g / 1e6}


def required_cii(dwt_t: float, year: int) -> dict:
    """연도별 required CII — 기준선 × (1−Z/100). 2027+ Z 미정."""
    if dwt_t >= DWT_SPLIT:
        ref = CII_REF_A_LARGE * dwt_t ** (-CII_REF_C_LARGE)
    else:
        ref = CII_REF_A_SMALL * dwt_t ** (-CII_REF_C_SMALL)
    z = Z_BY_YEAR.get(year)
    if z is None:
        return {"reference_g_per_dwt_nm": ref, "z_pct": None,
                "required_g_per_dwt_nm": ref * (1.0 - 0.11),
                "z_defined": False,
                "note": f"{year}년 Z 미정 (원전 G3 각주 — 개정 대기, "
                        "2026년 11% 유지 가정 병기)"}
    return {"reference_g_per_dwt_nm": ref, "z_pct": z,
            "required_g_per_dwt_nm": ref * (1.0 - z / 100.0),
            "z_defined": True,
            "note": "MEPC.338(76) Table 1"}


def cii_rating(attained_g_per_dwt_nm: float,
               required_g_per_dwt_nm: float) -> dict:
    """A~E 등급 — 비 = attained/required vs G4 경계."""
    ratio = attained_g_per_dwt_nm / max(required_g_per_dwt_nm, 1e-9)
    d1, d2, d3, d4 = DD_GENERAL_CARGO
    if ratio < d1:
        rating = "A"
    elif ratio < d2:
        rating = "B"
    elif ratio < d3:
        rating = "C"
    elif ratio < d4:
        rating = "D"
    else:
        rating = "E"
    return {"rating": rating, "ratio": ratio,
            "note": "D 3년 연속·E 1년 = 시정계획 의무 (MARPOL reg 28)"}


def cii_outlook(attained_g_per_dwt_nm: float, dwt_t: float) -> list:
    """같은 운항 실적의 연도별 등급 전망 (2023~2026 확정 Z)."""
    out = []
    for year in sorted(Z_BY_YEAR):
        req = required_cii(dwt_t, year)
        rat = cii_rating(attained_g_per_dwt_nm,
                         req["required_g_per_dwt_nm"])
        out.append({"year": year,
                    "required_g_per_dwt_nm":
                        req["required_g_per_dwt_nm"],
                    "rating": rat["rating"], "ratio": rat["ratio"]})
    return out
