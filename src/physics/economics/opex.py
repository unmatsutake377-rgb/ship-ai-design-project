"""운항 경제 — 연료비·수송 단가·소형 전기 등가 (경제성 2단계).

대형: 연간 연료 = P_service × SFOC × 운항시간 (만재·가동률 100%
단순화 — 실제 utilization은 후속, 정직 표기). 수송 단가 =
연료비 / (DWT × 연간 해리) [USD/(t·nm)] — 설계 비교용 상대 지표.

소형 전기: Wh/(kg·km) = P_electric / (v[km/h] × payload) — EEDI
철학의 전기판 (규제 아님·성적표 전용, 스펙 §1 정직 표기).

가격 대역 (C급 공개 시세 — #17 수집 후보):
- VLSFO 벙커 450~750 USD/t (2024~26 대역, 기본 600)
- 산업 전기 0.10~0.25 USD/kWh (기본 0.15)
운항시간 6,000 h/년 (연 250일 항해 관행 C급).
"""
from __future__ import annotations

KN_PER_MS = 3600.0 / 1852.0
BUNKER_USD_PER_T_BAND = (450.0, 750.0)     # VLSFO C급 대역
ELEC_USD_PER_KWH_BAND = (0.10, 0.25)       # 산업용 C급 대역
DEFAULT_BUNKER_USD_PER_T = 600.0
DEFAULT_ELEC_USD_PER_KWH = 0.15
DEFAULT_HOURS_PER_YEAR = 6000.0


def annual_fuel(p_service_kw: float, sfoc_g_per_kwh: float,
                hours_per_year: float = DEFAULT_HOURS_PER_YEAR
                ) -> dict:
    """연간 연료 소비 [t/년] — P × SFOC × 시간 (g → t)."""
    fuel_t = p_service_kw * sfoc_g_per_kwh * hours_per_year / 1e6
    return {"fuel_t_per_year": fuel_t,
            "hours_per_year": hours_per_year}


def fuel_opex(p_service_kw: float, sfoc_g_per_kwh: float,
              v_service_ms: float, dwt_t: float,
              bunker_usd_per_t: float = DEFAULT_BUNKER_USD_PER_T,
              hours_per_year: float = DEFAULT_HOURS_PER_YEAR) -> dict:
    """연간 연료비 + 톤·해리 수송 단가 [USD/(t·nm)].

    만재·가동률 100% 단순화 (설계 비교용 상대 지표 — 절대 운임
    아님, 정직 표기)."""
    fuel_t = annual_fuel(p_service_kw, sfoc_g_per_kwh,
                         hours_per_year)["fuel_t_per_year"]
    cost = fuel_t * bunker_usd_per_t
    dist_nm = v_service_ms * KN_PER_MS * hours_per_year
    return {
        "fuel_t_per_year": fuel_t,
        "fuel_cost_usd_per_year": cost,
        "distance_nm_per_year": dist_nm,
        "transport_usd_per_tnm": cost / (max(dwt_t, 1e-9)
                                         * max(dist_nm, 1e-9)),
        "bunker_usd_per_t": bunker_usd_per_t,
        "note": "만재·가동 100% 단순화 — 설계 비교용 상대 지표",
    }


def electric_transport(p_electric_w: float, v_ms: float,
                       payload_kg: float,
                       elec_usd_per_kwh: float =
                       DEFAULT_ELEC_USD_PER_KWH) -> dict:
    """소형 전기 수송 에너지 단가 — Wh/(kg·km) + 전기료.

    EEDI 철학의 전기 등가 (규제 아님 — 성적표 전용)."""
    wh_per_km = p_electric_w / max(v_ms * 3.6, 1e-9)
    wh_per_kg_km = wh_per_km / max(payload_kg, 1e-9)
    return {
        "wh_per_kg_km": wh_per_kg_km,
        "usd_per_kg_km": wh_per_kg_km * elec_usd_per_kwh / 1000.0,
        "elec_usd_per_kwh": elec_usd_per_kwh,
        "note": "전기 등가 성적표 — EEDI 규제 대상 아님 (정직 표기)",
    }
