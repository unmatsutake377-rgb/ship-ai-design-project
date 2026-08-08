"""IMO 조종성 기준 + 7번째 게이트 (조종성 3단계, 스펙 §4).

원전: IMO Resolution MSC.137(76) (2002) — imo.org 공식 PDF
(references/IMO_MSC137_76.pdf, §5.3 Criteria·§3 Application):
- 선회: advance ≤ 4.5L, tactical diameter ≤ 5L
- 초기 선회: 타 10°에서 침로 10° 변할 때까지 이동 ≤ 2.5L
- 10/10 지그재그 1차 오버슈트 ≤ 10°(L/V<10s) / 20°(≥30s) /
  5+½(L/V)° (사이 보간), 2차 ≤ 25°/40°/17.5+0.75(L/V)°
- 20/20 지그재그 1차 ≤ 25°
- 적용: **L ≥ 100 m** 전 선종 (화학·가스운반선은 전 길이)
정직 생략: 정지 성능 (전속 후진 ≤ 15L)은 후진 추력 모델 밖 — note
병기. 초기 선회 판정은 10° 타 시험 별도 실행.
"""
from __future__ import annotations

import math

IMO_MIN_LOA = 100.0     # 원전 §3.1


def imo_first_overshoot_limit_deg(l_over_v_s: float) -> float:
    """10/10 1차 오버슈트 한계 [deg] — 원전 §5.3.3.1."""
    if l_over_v_s < 10.0:
        return 10.0
    if l_over_v_s >= 30.0:
        return 20.0
    return 5.0 + 0.5 * l_over_v_s


def imo_second_overshoot_limit_deg(l_over_v_s: float) -> float:
    """10/10 2차 오버슈트 한계 [deg] — 원전 §5.3.3.2."""
    if l_over_v_s < 10.0:
        return 25.0
    if l_over_v_s >= 30.0:
        return 40.0
    return 17.5 + 0.75 * l_over_v_s


def maneuvering_report(ship, u0: float,
                       rudder_rate_dps: float = 2.34) -> dict:
    """표준 시험 일괄 실행 → 성적표 (판정 없음 — 게이트가 판정).

    rudder_rate_dps 기본 2.34°/s: SOLAS 계보 타장치 요건
    (35°→반대 30° 28초) 실선 통상값."""
    from src.physics.maneuvering.trials import turning_circle, zigzag
    dt = ship.par.lpp / max(u0, 1e-9) / 120.0   # 시간 스케일 비례
    t35 = turning_circle(ship, u0, delta_deg=35.0,
                         rudder_rate_dps=rudder_rate_dps, dt=dt)
    z10 = zigzag(ship, u0, delta_deg=10.0, switch_deg=10.0,
                 rudder_rate_dps=rudder_rate_dps, dt=dt)
    z20 = zigzag(ship, u0, delta_deg=20.0, switch_deg=20.0,
                 rudder_rate_dps=rudder_rate_dps, dt=dt)
    return {
        "advance_over_l": float(t35["advance_over_l"]),
        "transfer_over_l": float(t35["transfer_over_l"]),
        "tactical_diameter_over_l": float(
            t35["tactical_diameter_over_l"]),
        "zz10_first_overshoot_deg": float(z10["first_overshoot_deg"]),
        "zz10_second_overshoot_deg": float(
            z10["second_overshoot_deg"]),
        "zz20_first_overshoot_deg": float(z20["first_overshoot_deg"]),
        "l_over_v_s": float(ship.par.lpp / max(u0, 1e-9)),
    }


def maneuvering_gate(report: dict, loa: float) -> dict:
    """IMO MSC.137(76) 합불 — L ≥ 100 m 전용 (범위 밖은 성적표만).

    정직 생략: 정지 성능(후진)·초기 선회는 미판정 (모델 범위 — note).
    """
    lov = report["l_over_v_s"]
    checks = {
        "advance": bool(report["advance_over_l"] <= 4.5),
        "tactical_diameter": bool(
            report["tactical_diameter_over_l"] <= 5.0),
        "zz10_first": bool(report["zz10_first_overshoot_deg"]
                           <= imo_first_overshoot_limit_deg(lov)),
        "zz10_second": bool(report["zz10_second_overshoot_deg"]
                            <= imo_second_overshoot_limit_deg(lov)),
        "zz20_first": bool(report["zz20_first_overshoot_deg"] <= 25.0),
    }
    return {
        "applicable": loa >= IMO_MIN_LOA,
        "checks": checks,
        "passed": all(checks.values()),
        "limits": {
            "advance": 4.5, "tactical_diameter": 5.0,
            "zz10_first": imo_first_overshoot_limit_deg(lov),
            "zz10_second": imo_second_overshoot_limit_deg(lov),
            "zz20_first": 25.0,
        },
        "note": "IMO MSC.137(76) §5.3 — 정지 성능·초기 선회 미판정 "
                "(후진 모델 밖, 정직 생략)",
    }
