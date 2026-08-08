"""표준 조종 시험 — KVLCC2 원전 재현 앵커 (스펙 §3 ③④)."""
import math

import pytest

from src.physics.maneuvering.kvlcc2 import (
    KVLCC2_COEFFS,
    KVLCC2_L7,
    PAPER_ANCHORS,
)

U0 = 1.179          # 15.5 kn / √45.7 (원전 §3.2)
SCALE = 45.7
RATE_DPS = 1.76 * math.sqrt(SCALE)   # 원전 p12: 실선 1.76°/s Froude 환산


def _r0_schoenherr() -> float:
    """L7 정수 저항 — ITTC 마찰 + 형상계수 1.25 (Schoenherr 계보
    근사, C급). S = 27,194/45.7² m² (SIMMAN 계보)."""
    s_wet = 27194.0 / SCALE ** 2
    re = U0 * 7.00 / 1.14e-6
    cf = 0.075 / (math.log10(re) - 2.0) ** 2
    return 0.5 * 1000.0 * s_wet * U0 ** 2 * cf * 1.25


def _ship():
    from src.physics.maneuvering.mmg import MMGShip, solve_self_propulsion
    r0 = _r0_schoenherr()
    n_p = solve_self_propulsion(KVLCC2_L7, KVLCC2_COEFFS, U0, r0)
    return MMGShip(par=KVLCC2_L7, co=KVLCC2_COEFFS, r0_n=r0,
                   n_p=n_p, u0=U0)


def test_turning_circle_reproduces_paper():
    """35° 선회 — 원전 계산값 (A_D 3.31·D_T 3.36) ±10% 재현.

    원전 자체가 실측과 최대 5.8% 어긋남 — 우리 목표는 '같은 계수·
    같은 모델이면 같은 답' (모델 오차 단독 계측)."""
    from src.physics.maneuvering.trials import turning_circle
    r = turning_circle(_ship(), U0, delta_deg=35.0,
                       rudder_rate_dps=RATE_DPS)
    assert r["advance_over_l"] == pytest.approx(
        PAPER_ANCHORS["turning_advance_cal"], rel=0.10)
    assert r["tactical_diameter_over_l"] == pytest.approx(
        PAPER_ANCHORS["turning_tactical_cal"], rel=0.10)


def test_turning_mirror():
    """±35° 거울 — γR·C2 비대칭 허용 대역 내 좌우 유사."""
    from src.physics.maneuvering.trials import turning_circle
    rp = turning_circle(_ship(), U0, delta_deg=35.0,
                        rudder_rate_dps=RATE_DPS)
    rm = turning_circle(_ship(), U0, delta_deg=-35.0,
                        rudder_rate_dps=RATE_DPS)
    assert rm["tactical_diameter_over_l"] == pytest.approx(
        rp["tactical_diameter_over_l"], rel=0.10)


def test_zigzag_overshoot_band():
    """10/10 지그재그 — 오버슈트 존재·원전 Fig 14 대역 (1차 수 °)."""
    from src.physics.maneuvering.trials import zigzag
    z = zigzag(_ship(), U0, delta_deg=10.0, switch_deg=10.0,
               rudder_rate_dps=RATE_DPS)
    assert 1.0 < z["first_overshoot_deg"] < 15.0
    assert z["second_overshoot_deg"] > 0.0
    assert z["switch_count"] >= 3
