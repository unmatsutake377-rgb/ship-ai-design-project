import math

import pytest

from src.ai.hull_generator import generate_hull_mesh, solve_exponents
from src.core.types import MainDimensions
from src.physics.coefficients import (
    XU_DOT_MASS_FRACTION,
    clarke_nondim,
    estimate_coefficients,
)

DIMS = MainDimensions(loa=4.0, beam=1.0, depth=0.40, draft_design=0.25, cb=0.50)


def test_clarke_pinned_value():
    """수계산 고정: L=4, B=1, T=0.25 → (T/L)²=0.00390625.
    Yv' = π·(T/L)²·[1 + 0.40·Cb·(B/T)] = π·0.00390625·[1+0.40·0.5·4] = 0.031906...
    """
    nd = clarke_nondim(loa=4.0, beam=1.0, draft=0.25, cb=0.50)
    expected = math.pi * 0.00390625 * (1 + 0.40 * 0.50 * 4.0)
    assert nd["Yv_p"] == pytest.approx(expected, rel=1e-9)


def test_clarke_all_positive_magnitudes():
    """모든 계수를 크기(양수)로 저장하는 규약 확인."""
    nd = clarke_nondim(loa=4.0, beam=1.0, draft=0.25, cb=0.50)
    for key, value in nd.items():
        assert value > 0, key


def test_sway_added_mass_dominates_cross_terms():
    """세장체 물리: 횡 부가질량(Yv̇)이 교차항(Nv̇)보다 커야 함."""
    nd = clarke_nondim(loa=4.0, beam=1.0, draft=0.25, cb=0.50)
    assert nd["Yv_dot_p"] > nd["Nv_dot_p"]


def _full_set(speed=1.2):
    mesh = generate_hull_mesh(DIMS)
    n, m = solve_exponents(DIMS.cb)
    return estimate_coefficients(
        dims=DIMS, draft=0.20, mass=300.0, lcg=0.0, speed=speed,
        mesh=mesh, n_exp=n, m_exp=m,
    )


def test_surge_added_mass_fraction():
    coeffs = _full_set()
    assert coeffs.xu_dot == pytest.approx(XU_DOT_MASS_FRACTION * 300.0, rel=1e-9)


def test_surge_damping_matches_finite_difference():
    """Xu = dR/du @ U — 자체 저항곡선 중앙차분과 일치해야 함."""
    from src.physics.resistance import total_resistance

    mesh = generate_hull_mesh(DIMS)
    n, m = solve_exponents(DIMS.cb)
    u = 1.2
    r_hi = total_resistance(mesh, DIMS, n, m, draft=0.20, speed=1.05 * u).total
    r_lo = total_resistance(mesh, DIMS, n, m, draft=0.20, speed=0.95 * u).total
    expected = (r_hi - r_lo) / (0.10 * u)
    coeffs = _full_set(speed=u)
    assert coeffs.xu == pytest.approx(expected, rel=1e-6)
    assert coeffs.xu > 0  # 저항은 속도 증가함수


def test_dimensional_scaling_cubic():
    """차원화 검증: yv_dot = Yv̇'·½ρL³ 스케일."""
    coeffs = _full_set()
    nd = coeffs.nondim
    rho_half_l3 = 0.5 * 1025.0 * DIMS.loa ** 3
    assert coeffs.yv_dot == pytest.approx(nd["Yv_dot_p"] * rho_half_l3, rel=1e-9)


def test_extrapolation_warning_always_on():
    coeffs = _full_set()
    assert coeffs.extrapolation_warning is True


def test_straight_line_stability_reported():
    coeffs = _full_set()
    assert isinstance(coeffs.straight_line_stable, bool)
