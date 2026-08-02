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


def test_cargo_kg_interval_hand_calc():
    """2단계 구간 판정 손계산 (스펙 §3).

    구획 2개 (바닥면적 1.0·0.5 m², z0=0.1, 높이 0.4), 짐 밀도 600,
    짐 180 kg (부피 0.3 m³, 총 용량 0.6 m³의 절반):
    - 최저 KG: 균일 수위 h = 0.3/1.5 = 0.2 → 중심 0.1+0.1 = 0.2
    - 최고 KG: 좁은 구획(0.5)부터 기둥 — 꽉 채움 (0.2 m³, 중심 0.3)
      + 나머지 0.1 m³를 넓은 구획에 (h 0.1, 중심 0.15)
      → (120·0.3 + 60·0.15)/180 = 0.25"""
    from src.physics.cargo_hold import BayGeom, cargo_kg_interval

    bays = [BayGeom(floor_area=1.0, z0=0.1, height=0.4),
            BayGeom(floor_area=0.5, z0=0.1, height=0.4)]
    lo, hi = cargo_kg_interval(bays, payload_kg=180.0, density=600.0)
    assert lo == pytest.approx(0.2)
    assert hi == pytest.approx(0.25)


def test_cargo_kg_interval_single_bay_degenerate():
    """구획 1개면 배분 자유도 없음 — 구간이 한 점."""
    from src.physics.cargo_hold import BayGeom, cargo_kg_interval

    bays = [BayGeom(floor_area=1.0, z0=0.0, height=0.5)]
    lo, hi = cargo_kg_interval(bays, payload_kg=300.0, density=600.0)
    # h = 0.5/1.0... 부피 0.5, h=0.5 → 중심 0.25
    assert lo == pytest.approx(0.25)
    assert hi == pytest.approx(lo)


def test_cargo_kg_interval_overflow_returns_none():
    """총 용량 초과 = 배분 불가 (space_ok False와 일관)."""
    from src.physics.cargo_hold import BayGeom, cargo_kg_interval

    bays = [BayGeom(floor_area=0.1, z0=0.0, height=0.1)]
    assert cargo_kg_interval(bays, payload_kg=100.0, density=600.0) is None


def test_gm_band_reachable_overlap():
    """GM 밴드 겹침 판정: 도달 구간이 밴드와 겹치면 합격 배치 존재."""
    from src.physics.cargo_hold import gm_band_reachable

    # KM=0.5, 고정 모멘트 10 kg·m, 짐 100 kg, 총 200 kg, beam 1.0
    # 짐 KG 구간 [0.1, 0.4] → 총 KG [(10+10)/200, (10+40)/200] = [0.1, 0.25]
    # GM/B 구간 = [0.25, 0.40] → 밴드 (0.04, 0.40)와 겹침 → 합격
    ok, margin = gm_band_reachable(
        cargo_kg_lo=0.1, cargo_kg_hi=0.4, km=0.5, fixed_moment=10.0,
        cargo_mass=100.0, total_mass=200.0, beam=1.0,
        band=(0.04, 0.40))
    assert ok and margin > 0
    # 밴드 밖 (KM 낮아 GM 전부 음수) → 불합격
    ok2, margin2 = gm_band_reachable(
        cargo_kg_lo=0.4, cargo_kg_hi=0.5, km=0.2, fixed_moment=50.0,
        cargo_mass=100.0, total_mass=200.0, beam=1.0,
        band=(0.04, 0.40))
    assert not ok2 and margin2 < 0
