"""IACS UR S11 파랑 굽힘 + 표준파 준정적 (스펙 2026-08-09 §2·§3)."""
import numpy as np
import pytest
import trimesh

G = 9.81
RHO = 1025.0


def test_iacs_cw_anchor_values():
    """원전 구간식 재현 — L=300에서 10.75 (최대 구간 진입점)."""
    from src.physics.structure.wave_loads import iacs_wave_coefficient
    assert iacs_wave_coefficient(300.0) == pytest.approx(10.75)
    assert iacs_wave_coefficient(320.0) == pytest.approx(10.75)
    # 90~300 구간: L=200 → 10.75 − 1.0 = 9.75
    assert iacs_wave_coefficient(200.0) == pytest.approx(9.75)
    # 단조 증가 (90~300)
    ls = np.linspace(90.0, 300.0, 50)
    cws = [iacs_wave_coefficient(l) for l in ls]
    assert all(a <= b + 1e-12 for a, b in zip(cws, cws[1:]))


def test_iacs_bending_signs_and_magnitude():
    """호깅 양수·새깅 음수, 100m 화물선 자릿수 (1e5 kN·m 대역)."""
    from src.physics.structure.wave_loads import iacs_wave_bending_knm
    hog, sag = iacs_wave_bending_knm(100.0, 15.0, 0.75)
    assert hog > 0 > sag
    assert 5e4 < hog < 5e5


def test_iacs_range_honest_rejection():
    """적용 범위 밖 (소형선) = 정직 거절 — 원전 S11.1 L≥90m."""
    from src.physics.structure.wave_loads import (
        IACSRangeError,
        iacs_wave_coefficient,
    )
    with pytest.raises(IACSRangeError):
        iacs_wave_coefficient(10.0)
