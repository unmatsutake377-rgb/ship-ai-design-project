"""MaxBox — 선체 내부 최대 직육면체 (해석 정답지: 바지선)."""
import pytest
import trimesh

from src.physics.maxbox import largest_box, largest_rectangle


def test_largest_rectangle_hand_calc():
    """반폭 히스토그램 [1,3,3,1], 간격 1.0 → 최대 = 3×2칸? 손계산.

    창 [1..2] (2칸, 길이 2.0) × 최소반폭 3 → 면적 2.0×3=6.0 이
    전체 창 4칸×반폭 1 (4.0) 보다 큼."""
    area, x0, x1, w = largest_rectangle([1.0, 3.0, 3.0, 1.0], dx=1.0)
    assert area == pytest.approx(6.0)
    assert w == pytest.approx(3.0)
    assert (x1 - x0) == pytest.approx(2.0)


def test_largest_rectangle_uniform():
    """균일 반폭 → 전체 창이 최적."""
    area, x0, x1, w = largest_rectangle([2.0] * 5, dx=0.5)
    assert area == pytest.approx(2.5 * 2.0)


def test_barge_maxbox_is_whole_box():
    """직육면체 바지선: MaxBox ≈ 선체 전체 (이산 격자 오차 내)."""
    barge = trimesh.creation.box(bounds=[[-2, -0.5, 0.0], [2, 0.5, 0.6]])
    r = largest_box(barge, depth=0.6)
    assert r.volume == pytest.approx(4.0 * 1.0 * 0.6, rel=0.15)
    assert r.width == pytest.approx(1.0, rel=0.15)


def test_wigley_maxbox_reasonable():
    """Wigley: 부피 양수, 선체 상자보다 작고, 중앙 근방 배치."""
    from src.ai.hull_generator import generate_hull_mesh
    from src.core.types import MainDimensions

    dims = MainDimensions(loa=3.0, beam=0.75, depth=0.5,
                          draft_design=0.3, cb=0.444)
    mesh = generate_hull_mesh(dims)
    r = largest_box(mesh, depth=dims.depth)
    assert 0 < r.volume < 3.0 * 0.75 * 0.5
    # 중앙 대칭 선형 → 상자 중심이 중앙 근방
    assert abs(0.5 * (r.x0 + r.x1)) < 0.3 * dims.loa
