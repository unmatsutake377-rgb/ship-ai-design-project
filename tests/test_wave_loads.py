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


def _box_barge(loa=80.0, beam=10.0, depth=6.0):
    return trimesh.creation.box(extents=[loa, beam, depth])


def test_quasi_static_barge_analytic():
    """상자 바지선, λ=L 파정 중앙: |M_wave| = ρ·g·B·a·L²/(2π²).

    직벽 상자는 침하 보정 0 (cos 한 주기 적분 = 0) — 손계산 앵커."""
    from src.physics.structure.wave_loads import quasi_static_wave_moment
    loa, beam, t = 80.0, 10.0, 2.0
    mesh = _box_barge(loa, beam)
    mass = RHO * loa * beam * t
    wl_z = -3.0 + t
    amp = 0.5
    r = quasi_static_wave_moment(mesh, wl_z, [(mass, -40.0, 40.0)],
                                 wave_amp=amp, wavelength=loa, n=201)
    m_analytic = RHO * G * beam * amp * loa ** 2 / (2.0 * np.pi ** 2)
    assert r["m_wave_mid_nm"] == pytest.approx(m_analytic, rel=0.03)
    assert abs(r["sinkage_m"]) < 0.01 * amp


def test_quasi_static_hog_sag_mirror():
    """파정 중앙 = 호깅(+), 파곡 중앙 = 새깅(−) — 부호 거울."""
    from src.physics.structure.wave_loads import quasi_static_wave_moment
    mesh = _box_barge()
    mass = RHO * 80.0 * 10.0 * 2.0
    wl_z = -1.0
    hog = quasi_static_wave_moment(mesh, wl_z, [(mass, -40.0, 40.0)],
                                   wave_amp=0.5, wavelength=80.0,
                                   crest_mid=True)
    sag = quasi_static_wave_moment(mesh, wl_z, [(mass, -40.0, 40.0)],
                                   wave_amp=0.5, wavelength=80.0,
                                   crest_mid=False)
    assert hog["m_wave_mid_nm"] > 0 > sag["m_wave_mid_nm"]
    assert abs(hog["m_wave_mid_nm"]) == pytest.approx(
        abs(sag["m_wave_mid_nm"]), rel=0.05)
