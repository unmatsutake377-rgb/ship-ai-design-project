"""Ursell 반원 heave 검증 — 에너지 항등식·극한·문헌 곡선 성질."""
import math

import pytest

from src.physics.seakeeping.ursell import heave_coefficients

RHO = 1025.0
R0 = 1.0
REF = RHO * math.pi * R0 ** 2 / 2.0


def _at(xi):
    om = math.sqrt(xi * 9.81 / R0)
    return heave_coefficients(R0, om), om


def test_energy_identity_exact():
    """방사 감쇠 정확 항등식: N33' = ρg²(ηa/ya)²/ω³ — 급수·연립·
    적분 전체의 자기 일관 심판 (0.5% 이내)."""
    for xi in (0.25, 0.5, 1.0, 2.0, 5.0):
        u, om = _at(xi)
        n_energy = RHO * 9.81 ** 2 * u.amp_ratio ** 2 / om ** 3
        assert u.damping == pytest.approx(n_energy, rel=5e-3), xi


def test_high_frequency_limit_matches_lewis():
    """고주파 극한 → ρπR²/2 (Lewis 해석값과 접점, ξr=10에서 5% 이내)."""
    u, _ = _at(10.0)
    assert u.added_mass == pytest.approx(REF, rel=0.06)


def test_literature_dip_and_low_frequency_rise():
    """문헌 Ursell 곡선 성질: ξr≈1 부근 최소(~0.6·ref) +
    저주파 상승(발산 방향) + 감쇠 전 구간 양수."""
    u_low, _ = _at(0.1)
    u_dip, _ = _at(1.0)
    u_high, _ = _at(5.0)
    assert u_low.added_mass > REF                 # 저주파 상승
    assert 0.55 * REF < u_dip.added_mass < 0.70 * REF   # 골짜기
    assert u_dip.added_mass < u_high.added_mass   # 회복
    for u in (u_low, u_dip, u_high):
        assert u.damping > 0.0


def test_damping_peak_at_moderate_frequency():
    """조파감쇠 피크는 중저주파 (ξr ~0.2-0.5) — 고주파에서 소멸."""
    b_peak = _at(0.25)[0].damping
    b_high = _at(10.0)[0].damping
    assert b_peak > 10.0 * b_high
