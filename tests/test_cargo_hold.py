"""다구획 MaxBox 1단계 — 손계산 정답지 (스펙 2026-08-03 §5)."""
import numpy as np
import pytest
import trimesh

from src.physics.cargo_hold import (
    MIN_BOX_DIM,
    RESERVE_PACKING_FACTOR,
    RESERVED_DENSITY_KG_M3,
    multibay_hold,
    pack_bays,
    reserved_volume_for,
)


def test_reserved_volume_hand_calc():
    """추진 5 kg → 5/1843×3.0 m³ (배터리 실측 밀도 × 여유계수)."""
    expected = 5.0 / RESERVED_DENSITY_KG_M3 * RESERVE_PACKING_FACTOR
    assert reserved_volume_for(5.0) == pytest.approx(expected)
    assert expected == pytest.approx(0.00814, rel=1e-2)


def test_pack_bays_single_flat_histogram():
    """평평한 지형도 = 구획 1개가 전체를 먹음 (손계산).

    반폭 0.5 × 10칸 × dx 0.2 → 상자 길이 2.0, 폭 1.0."""
    boxes = pack_bays([0.5] * 10, dx=0.2, height=0.4, min_dim=0.25)
    assert len(boxes) == 1
    length, width, height, volume, x0, x1 = boxes[0]
    assert length == pytest.approx(2.0)
    assert width == pytest.approx(1.0)
    assert volume == pytest.approx(2.0 * 1.0 * 0.4)


def test_pack_bays_dumbbell_two_bays():
    """아령 지형도 (넓은 로브 2 + 좁은 허리): 단일 상자는 로브
    하나에 갇히지만 다구획은 양쪽을 다 먹는다 — 존재 이유 시험."""
    w = [0.5] * 5 + [0.05] * 2 + [0.5] * 5   # 허리 반폭 0.05 (종잇장)
    dx, h = 0.2, 0.4
    boxes = pack_bays(w, dx=dx, height=h, min_dim=0.25)
    # 허리(폭 0.1 < 0.25)는 상자로 못 씀 — 로브 2개만
    assert len(boxes) == 2
    total = sum(b[3] for b in boxes)
    single = 0.5 * 2 * 1.0 * h               # 로브 하나 (반폭0.5×2, 1.0m)
    assert total == pytest.approx(2 * single)


def test_pack_bays_rejects_paper_slices():
    """전부 종잇장(최소 치수 미달)이면 상자 0개 — 과관대 방지."""
    assert pack_bays([0.05] * 10, dx=0.2, height=0.4, min_dim=0.25) == []
    # 높이 미달도 기각
    assert pack_bays([0.5] * 10, dx=0.2, height=0.1, min_dim=0.25) == []


def test_multibay_hold_box_hull():
    """직육면체 껍데기 L2×B1×D0.5, 예약 0.1 m³ 손계산.

    선미 절단 길이 = 0.1/(1.0×0.5) = 0.2 m → 화물 길이 1.8 m,
    상자 1개 ≈ 1.8×1.0×0.5 = 0.9 m³ (격자 이산화 오차 허용)."""
    hull = trimesh.creation.box(extents=[2.0, 1.0, 0.5])
    hull.apply_translation([1.0, 0.0, 0.25])   # x∈[0,2], z∈[0,0.5]
    hold = multibay_hold(hull, depth=0.5, reserved_volume=0.1,
                         stern="xmax")
    assert hold.reserved_volume == pytest.approx(0.1)
    assert len(hold.boxes) >= 1
    assert hold.total_volume == pytest.approx(0.9, rel=0.15)
    # 절단 경계가 선미(xmax)쪽 0.2 m 부근
    assert hold.stern_cut_x == pytest.approx(1.8, abs=0.15)
    # 상자가 절단 경계를 침범하지 않음
    assert max(b.x1 for b in hold.boxes) <= hold.stern_cut_x + 1e-6


def test_multibay_hold_no_reserve_matches_hull():
    """예약 0이면 절단 없음 — 상자 합이 단일 MaxBox 이상."""
    from src.physics.maxbox import largest_box

    hull = trimesh.creation.box(extents=[2.0, 1.0, 0.5])
    hull.apply_translation([1.0, 0.0, 0.25])
    hold = multibay_hold(hull, depth=0.5, reserved_volume=0.0,
                         stern="xmax")
    single = largest_box(hull, depth=0.5)
    assert hold.total_volume >= single.volume * 0.95   # 이산화 오차 여유
