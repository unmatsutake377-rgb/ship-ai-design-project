"""MMG 계수 추정 — 회귀식 검증·KVLCC2 자기 대조 (2단계, 스펙 §5-2)."""
import math

import pytest

from src.physics.maneuvering.kvlcc2 import KVLCC2_COEFFS, KVLCC2_L7


def test_regression_formulas_spot_values():
    """원전 Eq 11·14~18 손계산 재현 (Cb 0.6·L/B 5.5·d/B 0.35)."""
    from src.physics.maneuvering.estimation import estimate_mmg_coeffs
    loa, beam = 55.0, 10.0
    draft = 3.5
    co, notes = estimate_mmg_coeffs(
        loa=loa, beam=beam, draft=draft, cb=0.60,
        displacement_m3=0.60 * loa * beam * draft,
        xg=0.0, dp=2.5, hr=3.5, ar=loa * draft / 54.0,
        w_p0=0.25, t_p=0.20,
        k0=0.2931, k1=0.2753, k2=-0.1385)
    k = 2.0 * draft / loa
    cb_lb = 0.60 / (loa / beam)
    # Kijima Eq 11 (β→v 부호 변환)
    assert co.yv == pytest.approx(-(0.5 * math.pi * k + 1.4 * cb_lb),
                                  rel=1e-6)
    assert co.nv == pytest.approx(-k, rel=1e-6)
    assert co.nr == pytest.approx(-0.54 * k + k * k, rel=1e-6)
    # Yoshimura Eq 15~16 (τ'=0)
    assert co.yvvv == pytest.approx(-(0.185 * loa / beam + 0.48),
                                    rel=1e-6)
    assert co.yvvr == pytest.approx(-0.75, rel=1e-6)
    assert co.nrrr == pytest.approx(0.25 * cb_lb - 0.056, rel=1e-6)
    # 상호작용 Eq 17~18
    assert co.t_r == pytest.approx(0.39, rel=1e-6)
    assert co.a_h == pytest.approx(3.6 * cb_lb, rel=1e-6)
    assert co.x_h_p == pytest.approx(-0.4)
    assert co.ell_r_p == pytest.approx(-0.9)
    eps = 2.26 - 1.82 * (1.0 - 0.25)
    assert co.eps == pytest.approx(eps, rel=1e-6)
    assert co.kappa == pytest.approx(0.55 / eps, rel=1e-6)
    assert notes["band"] == "in"


def test_kvlcc2_estimated_vs_true_turning():
    """추정 계수 KVLCC2 선회 vs 원전 계수 선회 — 추정 오차 단독
    계측 (풀선형 Cb 0.81 = 회귀 대역 밖 C급 외삽, 대역 0.6~1.6)."""
    from src.physics.maneuvering.estimation import estimate_mmg_coeffs
    from src.physics.maneuvering.mmg import MMGShip, solve_self_propulsion
    from src.physics.maneuvering.trials import turning_circle
    p = KVLCC2_L7
    co_est, notes = estimate_mmg_coeffs(
        loa=p.lpp, beam=p.beam, draft=p.draft,
        cb=p.cb, displacement_m3=p.displacement_m3, xg=p.xg,
        dp=p.dp, hr=p.hr, ar=p.ar,
        w_p0=0.40, t_p=0.220,
        k0=0.2931, k1=0.2753, k2=-0.1385, rho=1000.0)
    assert notes["band"] == "extrapolated_full"
    u0, r0 = 1.179, 35.9
    rate = 1.76 * math.sqrt(45.7)
    n_est = solve_self_propulsion(p, co_est, u0, r0)
    n_tru = solve_self_propulsion(p, KVLCC2_COEFFS, u0, r0)
    est = MMGShip(par=p, co=co_est, r0_n=r0, n_p=n_est, u0=u0)
    tru = MMGShip(par=p, co=KVLCC2_COEFFS, r0_n=r0, n_p=n_tru, u0=u0)
    dt_est = turning_circle(est, u0, 35.0, rate)
    dt_tru = turning_circle(tru, u0, 35.0, rate)
    ratio = (dt_est["tactical_diameter_over_l"]
             / dt_tru["tactical_diameter_over_l"])
    print(f"\n[추정/실측 계수] D_T {dt_est['tactical_diameter_over_l']:.2f}"
          f" vs {dt_tru['tactical_diameter_over_l']:.2f} (비 {ratio:.2f})")
    assert 0.6 < ratio < 1.6


def test_small_craft_honest_rejection():
    """Cb < 0.51 (소형 USV) = 회귀 대역 밖 정직 거절."""
    from src.physics.maneuvering.estimation import (
        EstimationRangeError,
        estimate_mmg_coeffs,
    )
    with pytest.raises(EstimationRangeError):
        estimate_mmg_coeffs(loa=3.0, beam=1.2, draft=0.3, cb=0.45,
                            displacement_m3=0.5, xg=0.0, dp=0.2,
                            hr=0.3, ar=0.02, w_p0=0.1, t_p=0.1,
                            k0=0.3, k1=0.27, k2=-0.14)
