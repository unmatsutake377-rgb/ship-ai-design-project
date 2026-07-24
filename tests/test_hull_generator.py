import pytest

from src.ai.hull_generator import (
    CB_ENVELOPE,
    CbOutOfRangeError,
    generate_hull_mesh,
    solve_exponents,
)
from src.core.types import MainDimensions

DIMS = MainDimensions(loa=4.0, beam=1.3, depth=0.48, draft_design=0.30, cb=0.50)


def test_solve_exponents_recovers_cb():
    n, m = solve_exponents(cb=0.50, cm=0.78)
    cp = n / (n + 1)
    cm = m / (m + 1)
    assert cp * cm == pytest.approx(0.50, abs=1e-9)


def test_cb_out_of_envelope_raises():
    with pytest.raises(CbOutOfRangeError):
        solve_exponents(cb=CB_ENVELOPE[1] + 0.05)
    with pytest.raises(CbOutOfRangeError):
        solve_exponents(cb=CB_ENVELOPE[0] - 0.05)


def test_mesh_is_watertight():
    mesh = generate_hull_mesh(DIMS)
    assert mesh.is_watertight
    assert mesh.volume > 0


def test_underwater_volume_matches_cb():
    """설계 흘수까지 잘랐을 때 부피 = Cb·L·B·T (해석해, spec §4)."""
    import trimesh

    mesh = generate_hull_mesh(DIMS)
    below = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal=[0, 0, -1],
        plane_origin=[0, 0, DIMS.draft_design], cap=True,
    )
    expected = DIMS.cb * DIMS.loa * DIMS.beam * DIMS.draft_design
    assert below.volume == pytest.approx(expected, rel=0.02)


def test_total_volume_analytic():
    """전체 부피 = 수면하(Cb·L·B·T) + 현측 프리즘(Cp·L·B·(D−T))."""
    mesh = generate_hull_mesh(DIMS)
    n, m = solve_exponents(DIMS.cb)
    cp = n / (n + 1)
    under = DIMS.cb * DIMS.loa * DIMS.beam * DIMS.draft_design
    above = cp * DIMS.loa * DIMS.beam * (DIMS.depth - DIMS.draft_design)
    assert mesh.volume == pytest.approx(under + above, rel=0.02)
