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


def test_cm_for_purpose_presets():
    """용도별 선저 프리셋 (스펙 hull-bottoms, Ship-D 실측 역산)."""
    from src.ai.hull_generator import cm_for_purpose

    assert cm_for_purpose("survey") == 0.85
    assert cm_for_purpose("workboat") == 0.92
    assert cm_for_purpose("patrol") == 0.80
    assert cm_for_purpose("survey", override=0.7) == 0.7
    assert cm_for_purpose("unknown") == 0.78     # 구세계 호환


def test_survey_bottom_within_shipd_band():
    """신 survey 선형의 바닥 폭비율이 Ship-D 실측 25~75분위 안.

    f10 = 바닥 10% 높이 단면 폭 / 최대 폭. Ship-D 300척: 0.25~0.68
    (스펙 §1). 구세계(Cm 0.78)는 0.46 하단 — 신세계 0.85는 중앙 부근."""
    import numpy as np

    from src.ai.hull_generator import generate_hull_mesh
    from src.core.types import MainDimensions

    dims = MainDimensions(loa=3.0, beam=0.9, depth=0.45,
                          draft_design=0.28, cb=0.5)
    mesh = generate_hull_mesh(dims, cm=0.85)
    zmin, zmax = mesh.bounds[0][2], mesh.bounds[1][2]
    z = zmin + 0.10 * (dims.draft_design)      # 흘수 기준 10% 높이
    sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    pts = np.vstack([e.discrete(sec.vertices) for e in sec.entities])
    f10 = (pts[:, 1].max() - pts[:, 1].min()) / dims.beam
    assert 0.25 <= f10 <= 0.75


def test_asym_exponents_hand_calc():
    """비대칭 지수 역산 손계산: Cp·LCB 목표 동시 달성.

    대칭 극한 (lcb=0): n_b = n_s = 기존 solve_exponents와 동일."""
    from src.ai.hull_generator import solve_asym_exponents, solve_exponents

    n_sym, _ = solve_exponents(0.45, 0.85)
    n_b, n_s = solve_asym_exponents(0.45, cm=0.85, lcb_frac=0.0)
    assert n_b == pytest.approx(n_sym, rel=1e-3)
    assert n_s == pytest.approx(n_sym, rel=1e-3)
    # LCB 전방(음수, +x=선수... 규약: lcb_frac<0 = 선미쪽? 아니 —
    # 본 스펙 규약: lcb_frac = (LCB−중앙)/L, +가 선수쪽) → 선수 풍만
    n_b2, n_s2 = solve_asym_exponents(0.45, cm=0.85, lcb_frac=+0.02)
    assert n_b2 > n_s2      # 선수쪽 지수가 커야 선수가 풍만


def test_asym_mesh_lcb_matches_target():
    """생성 메쉬의 실제 부피 도심이 목표 LCB에 안착 (±0.3%L)."""
    import numpy as np

    from src.ai.hull_generator import generate_hull_mesh
    from src.core.types import MainDimensions

    dims = MainDimensions(loa=3.0, beam=0.9, depth=0.45,
                          draft_design=0.28, cb=0.45)
    for target in (0.0, 0.02, -0.02):
        mesh = generate_hull_mesh(dims, cm=0.85, lcb_frac=target)
        assert mesh.is_watertight
        lcb = float(mesh.center_mass[0]) / dims.loa
        assert abs(lcb - target) < 0.003, (target, lcb)


def test_asym_default_symmetric_regression():
    """저수준 기본(asym 미지정) = 대칭 — Michell 해석해 표준기 보존."""
    import numpy as np

    from src.ai.hull_generator import generate_hull_mesh
    from src.core.types import MainDimensions

    dims = MainDimensions(loa=3.0, beam=0.9, depth=0.45,
                          draft_design=0.28, cb=0.45)
    mesh = generate_hull_mesh(dims, cm=0.85)
    # 이산화 잔차 실측 ~7e-5·L — 0.1%L 문턱 (수학 대칭 확인용)
    assert abs(float(mesh.center_mass[0])) < 1e-3 * dims.loa


def test_lcb_by_purpose_within_shipd_band():
    """용도 프리셋 LCB가 Ship-D 실측 10~90분위(−11.5%~+5.7%... 규약
    변환: 전방 = +) 안."""
    from src.ai.hull_generator import LCB_BY_PURPOSE

    for purpose, lcb in LCB_BY_PURPOSE.items():
        assert -0.057 <= lcb <= 0.115, (purpose, lcb)
