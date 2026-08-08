"""구조 강도 1단계 — 하중 곡선 시험 (스펙 2026-08-09 §5-1)."""
import numpy as np
import pytest
import trimesh

G = 9.81
RHO = 1025.0


def test_weight_blocks_closure():
    """성분 블록 합 = 총중량 (폐합 항등식)."""
    from src.physics.structure.loads import (
        standard_weight_blocks,
        weight_linear_density,
    )
    comp = {"structure": 800.0, "outfit": 200.0, "machinery": 300.0,
            "fuel": 100.0, "payload": 600.0}
    blocks = standard_weight_blocks(comp, xmin=-40.0, loa=80.0)
    xs = np.linspace(-40.0, 40.0, 201)
    w = weight_linear_density(xs, blocks)
    total = np.trapezoid(w, xs)
    assert total == pytest.approx(sum(comp.values()) * G, rel=1e-9)
    assert np.all(w >= 0)


def test_weight_blocks_placement():
    """기관·연료 = 선미 구간, 화물 = 중앙 구간 (통상 배치)."""
    from src.physics.structure.loads import standard_weight_blocks
    blocks = standard_weight_blocks(
        {"machinery": 100.0, "payload": 100.0}, xmin=0.0, loa=100.0)
    named = {}
    for (m, x0, x1), name in zip(blocks, ["machinery", "payload"]):
        named[name] = (x0, x1)
    m0, m1 = named["machinery"]
    p0, p1 = named["payload"]
    assert m1 <= 30.0          # 기관실 = 선미 30% 안
    assert 20.0 <= p0 and p1 <= 90.0   # 화물창 = 중앙부


def _box_barge(loa=80.0, beam=10.0, depth=6.0):
    return trimesh.creation.box(extents=[loa, beam, depth])


def test_still_water_uniform_barge_zero_moment():
    """균일 중량 상자 바지선 → M(x) ≈ 0 전역 (분포 일치 항등식)."""
    from src.physics.structure.loads import still_water_curves
    mesh = _box_barge()
    draft_T = 2.0
    mass = RHO * 80.0 * 10.0 * draft_T
    wl_z = -3.0 + draft_T                      # z ∈ [-3, 3] 상자
    curves = still_water_curves(mesh, wl_z, [(mass, -40.0, 40.0)])
    scale = mass * G * 80.0 / 16.0             # WL/16 기준 스케일
    assert np.max(np.abs(curves.moment_nm)) < 0.01 * scale


def test_still_water_midship_cargo_analytic():
    """중앙 절반 몰림 → |M_mid| = WL/16, 부호 음(새깅) — 손계산 앵커."""
    from src.physics.structure.loads import still_water_curves
    mesh = _box_barge()
    draft_T = 2.0
    mass = RHO * 80.0 * 10.0 * draft_T
    wl_z = -3.0 + draft_T
    curves = still_water_curves(mesh, wl_z, [(mass, -20.0, 20.0)],
                                n=201)
    m_mid = curves.moment_nm[len(curves.xs) // 2]
    m_analytic = mass * G * 80.0 / 16.0
    assert m_mid == pytest.approx(-m_analytic, rel=0.02)
    # 폐합 항등식: 양끝 V·M 잔차가 최대값 대비 미소
    assert abs(curves.shear_residual_n) < 0.02 * np.max(
        np.abs(curves.shear_n))


def test_station_area_box():
    """상자 단면적 = B×t 해석해."""
    from src.physics.structure.loads import station_area
    mesh = _box_barge()
    a = station_area(mesh, 0.0, -1.0)          # 흘수 2m (킬 -3)
    assert a == pytest.approx(10.0 * 2.0, rel=0.01)
