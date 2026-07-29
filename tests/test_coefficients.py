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
    nd, clamped = clarke_nondim(loa=4.0, beam=1.0, draft=0.25, cb=0.50)
    expected = math.pi * 0.00390625 * (1 + 0.40 * 0.50 * 4.0)
    assert nd["Yv_p"] == pytest.approx(expected, rel=1e-9)
    assert clamped == []  # 날씬한 선형은 회귀 범위 안


def test_clarke_all_positive_magnitudes():
    """모든 계수를 크기(양수)로 저장하는 규약 확인."""
    nd, _ = clarke_nondim(loa=4.0, beam=1.0, draft=0.25, cb=0.50)
    for key, value in nd.items():
        assert value > 0, key


def test_sway_added_mass_dominates_cross_terms():
    """세장체 물리: 횡 부가질량(Yv̇)이 교차항(Nv̇)보다 커야 함."""
    nd, _ = clarke_nondim(loa=4.0, beam=1.0, draft=0.25, cb=0.50)
    assert nd["Yv_dot_p"] > nd["Nv_dot_p"]


def test_stubby_hull_clamped_but_physical():
    """B/L=0.5 (실선 USV 비율): Clarke 범위 밖 → 클램프 발동하되
    모든 계수는 양수 유지 (음의 관성 금지 — M4b 발산 회귀 방지)."""
    nd, clamped = clarke_nondim(loa=1.97, beam=0.985, draft=0.175, cb=0.50)
    assert "Nr_dot_p" in clamped
    for key, value in nd.items():
        assert value > 0, key


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


def test_vertical_plane_estimates_physical():
    """B-3a: 수직면 계수 — 양수, 감쇠비 역산 공식 고정."""
    from src.physics.coefficients import (
        VERTICAL_DAMPING_ZETA,
        vertical_plane_estimates,
    )

    v = vertical_plane_estimates(mass=148.5, ixx=17.7, iyy=36.0,
                                 awp=1.35, ixx_wp=0.096, gm=0.29,
                                 disp_vol=0.145, loa=1.97)
    for key, val in v.items():
        assert val > 0, key
    # 상하축 감쇠 공식 고정: b33 = 2ζ√((m+A33)·ρgAwp)
    c33 = 1025.0 * 9.81 * 1.35
    expected = 2 * VERTICAL_DAMPING_ZETA * math.sqrt((148.5 + 148.5) * c33)
    assert v["z_damping"] == pytest.approx(expected, rel=1e-9)
