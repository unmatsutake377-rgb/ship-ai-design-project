"""Lewis 단면 검증 (내항성 1단계) — 반원 해석 앵커 + 기하 재현."""
import math

import pytest

from src.physics.seakeeping.lewis import (
    LewisRangeError,
    added_mass_heave_inf,
    fit_lewis,
    section_points,
)


def test_semicircle_self_verification():
    """반원 (H=1, σ=π/4): a1=a3=0, M=R — 공식의 자기 검증 성질."""
    sec = fit_lewis(beam=2.0, draft=1.0, sigma=math.pi / 4.0)
    assert sec.a1 == pytest.approx(0.0, abs=1e-12)
    assert sec.a3 == pytest.approx(0.0, abs=1e-12)
    assert sec.scale == pytest.approx(1.0)


def test_semicircle_added_mass_classic():
    """반원 무한주파수 heave 부가질량 = ρπR²/2 (고전 해석값)."""
    sec = fit_lewis(beam=2.0, draft=1.0, sigma=math.pi / 4.0)
    assert added_mass_heave_inf(sec, rho=1025.0) == pytest.approx(
        1025.0 * math.pi * 1.0 ** 2 / 2.0, rel=1e-12)


def test_geometry_reproduction():
    """사상 곡선이 (B/2, T, σ)를 재현하는가 — 적분 폐합."""
    sec = fit_lewis(beam=3.0, draft=1.2, sigma=0.85)
    pts = section_points(sec, n=400)
    ys = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    assert max(ys) == pytest.approx(1.5, rel=1e-6)      # B/2 (수선)
    assert zs[0] == pytest.approx(0.0, abs=1e-9)        # θ=0 = 킬
    assert zs[-1] == pytest.approx(1.2, rel=1e-6)       # θ=π/2 = 수선
    # 반단면적 사다리꼴 적분 → σ 재현 (A_half = σ·(B/2)·T)
    area = abs(sum(0.5 * (y0 + y1) * (z0 - z1)
                   for (y0, z0), (y1, z1) in zip(pts, pts[1:])))
    assert area == pytest.approx(0.85 * 1.5 * 1.2, rel=5e-3)


def test_fuller_section_more_added_mass():
    """성질: 같은 B·T에서 풍만할수록(σ↑) 부가질량 증가."""
    lean = fit_lewis(2.0, 1.0, 0.70)
    full = fit_lewis(2.0, 1.0, 0.95)
    assert added_mass_heave_inf(full) > added_mass_heave_inf(lean)


def test_out_of_range_clamps_to_border():
    """원전 식 7.95 지시: 범위 밖 σ는 가장 가까운 경계로 클램프해
    최선의 Lewis 계수 — H=1에서 하한 3π/32·(2−1)=0.2945."""
    import math as _m

    sec = fit_lewis(2.0, 1.0, 0.20)     # 극단 야윈 → 하한 클램프
    assert sec.sigma == pytest.approx(3 * _m.pi / 32, rel=1e-9)
    with pytest.raises(LewisRangeError):
        fit_lewis(2.0, -1.0, 0.8)       # 비물리 치수는 여전히 거절
