"""IMO EEDI — attained·required·판정 (경제성 1단계, 스펙 §2).

원전 (references/, imo.org 공식 PDF):
- MEPC.328(76) p39 Table 2: 기준선 General cargo a=107.48·c=0.216
  (reference = a·DWT^{-c} [gCO₂/(t·nm)])
- MEPC.328(76) p37 Table 1 (이미지 판독 관례): Phase 3 (2022-04+)
  감축 30% — General cargo 15,000 DWT+, 3,000~15,000 선형 0→30
  (* 각주 보간 계보), <3,000 적용 밖
- MEPC.364(79) CF 표: MDO 3.206·HFO 3.114 tCO₂/t-fuel

attained = CF·SFC·P75 / (DWT·Vref) — P75 = 75% MCR (원전 관례),
Vref는 설계점에서 P ∝ V³ 역산 (프로펠러 법칙 근사, 정직 표기).
단위: SFC g/kWh × CF t/t → 분자 gCO₂/h, 분모 t × kn = t·nm/h —
비 = gCO₂/(t·nm). 속도 kn = m/s × 3600/1852 (단위 사고 다발 지점).
"""
from __future__ import annotations

CF_MDO = 3.206      # tCO₂/t-fuel (MEPC.364 표 — Diesel/Gas Oil)
CF_HFO = 3.114
REF_A_GENERAL_CARGO = 107.48    # MEPC.328(76) Table 2
REF_C_GENERAL_CARGO = 0.216
PHASE3_REDUCTION_PCT = 30.0     # Table 1, General cargo 15k+
DWT_FULL_REDUCTION = 15_000.0
DWT_MIN_APPLICABLE = 3_000.0
KN_PER_MS = 3600.0 / 1852.0


def attained_eedi(mcr_kw: float, sfoc_g_per_kwh: float, dwt_t: float,
                  v_service_ms: float, p_service_kw: float,
                  cf_t_co2_per_t: float = CF_MDO) -> dict:
    """설계 배의 attained EEDI [gCO₂/(t·nm)]."""
    p75 = 0.75 * mcr_kw
    v_ref_ms = v_service_ms \
        * (p75 / max(p_service_kw, 1e-9)) ** (1.0 / 3.0)
    v_ref_kn = v_ref_ms * KN_PER_MS
    eedi = (cf_t_co2_per_t * sfoc_g_per_kwh * p75) \
        / (max(dwt_t, 1e-9) * max(v_ref_kn, 1e-9))
    return {"eedi_g_per_tnm": eedi, "v_ref_kn": v_ref_kn,
            "p_75_kw": p75,
            "note": "Vref = 설계점 P∝V³ 역산 (프로펠러 법칙 근사)"}


def required_eedi(dwt_t: float, phase: int = 3) -> dict:
    """required EEDI — 기준선 × (1 − 감축률). General cargo 계보만
    (타 선종 정직 미지원)."""
    ref = REF_A_GENERAL_CARGO * dwt_t ** (-REF_C_GENERAL_CARGO)
    if dwt_t < DWT_MIN_APPLICABLE:
        return {"reference_g_per_tnm": ref, "reduction_pct": 0.0,
                "required_g_per_tnm": ref, "applicable": False,
                "note": "DWT < 3,000 — EEDI 규제 적용 밖 (원전 Table 1)"}
    if dwt_t >= DWT_FULL_REDUCTION:
        red = PHASE3_REDUCTION_PCT
    else:
        red = PHASE3_REDUCTION_PCT * (
            (dwt_t - DWT_MIN_APPLICABLE)
            / (DWT_FULL_REDUCTION - DWT_MIN_APPLICABLE))
    return {"reference_g_per_tnm": ref, "reduction_pct": red,
            "required_g_per_tnm": ref * (1.0 - red / 100.0),
            "applicable": True,
            "note": "Phase 3 (2022-04+) — General cargo 계보"}


def eedi_verdict(attained_g_per_tnm: float,
                 required_g_per_tnm: float) -> dict:
    """합불 + 여유율 (%). margin > 0 = 여유 있음."""
    margin = 100.0 * (required_g_per_tnm - attained_g_per_tnm) \
        / max(required_g_per_tnm, 1e-9)
    return {"passed": bool(attained_g_per_tnm <= required_g_per_tnm),
            "margin_pct": margin}
