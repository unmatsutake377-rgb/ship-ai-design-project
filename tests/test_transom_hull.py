"""트랜섬 선형 생성기 테스트 (Phase C-1 Task 1)."""
import numpy as np
import pytest
import trimesh

from src.ai.hull_generator import (
    CbOutOfRangeError,
    TRANSOM_BETA,
    TRANSOM_BOW_FRACTION,
    TRANSOM_CM,
    TRANSOM_TAPER_P,
    generate_transom_hull_mesh,
    solve_transom_exponents,
    submerged_transom_area,
)
from src.core.types import MainDimensions

DIMS = MainDimensions(loa=4.0, beam=1.3, depth=0.48, draft_design=0.30,
                      cb=0.45)


def test_solve_recovers_cb():
    n, m = solve_transom_exponents(DIMS.cb)
    k_aft = 1 - (1 - TRANSOM_BETA) / (TRANSOM_TAPER_P + 1)
    cp = (TRANSOM_BOW_FRACTION * n / (n + 1)
          + (1 - TRANSOM_BOW_FRACTION) * k_aft)
    cm = m / (m + 1)
    assert cp * cm == pytest.approx(DIMS.cb, abs=1e-9)


def test_cb_out_of_range_rejected():
    with pytest.raises(CbOutOfRangeError):
        solve_transom_exponents(0.25)
    with pytest.raises(CbOutOfRangeError):
        solve_transom_exponents(0.70)


def test_mesh_watertight_with_transom():
    mesh = generate_transom_hull_mesh(DIMS)
    assert mesh.is_watertight
    assert mesh.volume > 0


def test_underwater_volume_matches_cb():
    """설계 흘수 절단 부피 = Cb·L·B·T (해석 폐합 — Wigley와 동일 규율)."""
    mesh = generate_transom_hull_mesh(DIMS)
    below = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal=[0, 0, -1],
        plane_origin=[0, 0, DIMS.draft_design], cap=True,
    )
    expected = DIMS.cb * DIMS.loa * DIMS.beam * DIMS.draft_design
    assert below.volume == pytest.approx(expected, rel=0.02)


def test_transom_face_width():
    """선미(x=−L/2) 수선(z=T)에서 폭 = β_t·B (트랜섬 실존 확인)."""
    mesh = generate_transom_hull_mesh(DIMS)
    stern = mesh.vertices[np.isclose(mesh.vertices[:, 0], -DIMS.loa / 2,
                                     atol=1e-6)]
    at_wl = stern[np.isclose(stern[:, 2], DIMS.draft_design, atol=0.02)]
    width = at_wl[:, 1].max() - at_wl[:, 1].min()
    assert width == pytest.approx(TRANSOM_BETA * DIMS.beam, rel=0.05)


def test_submerged_transom_area_bounds():
    """0 < A_t < β_t·B·T (H≤1이므로 상한은 직사각형)."""
    a_t = submerged_transom_area(DIMS, draft=0.25)
    assert 0 < a_t < TRANSOM_BETA * DIMS.beam * 0.25
