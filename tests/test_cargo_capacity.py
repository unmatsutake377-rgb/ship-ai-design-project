"""대형 화물창 용적 게이트 — 손계산·실선·전선 앵커 (스펙 2026-08-09)."""
import pytest
import trimesh

RHO_FUEL = 0.9      # t/m³


def test_box_barge_hand_calc():
    """상자 손계산: L100×B15×D8 = 12,000 m³ 전체.

    이중저 h=B/15=1.0m → 1,500 / 기관실 15% (이중저 위) →
    0.15×100×15×7 = 1,575 / 연료 100t → 111.1 / 잔여 ×0.90."""
    from src.physics.cargo_capacity import hold_volume_large
    mesh = trimesh.creation.box(extents=[100.0, 15.0, 8.0])
    r = hold_volume_large(mesh, depth=8.0, loa=100.0, fuel_t=100.0)
    assert r["gross_m3"] == pytest.approx(12_000.0, rel=0.01)
    assert r["double_bottom_m3"] == pytest.approx(1_500.0, rel=0.02)
    assert r["engine_room_m3"] == pytest.approx(1_575.0, rel=0.03)
    expected = (12_000.0 - 1_500.0 - 1_575.0 - 100.0 / RHO_FUEL) * 0.9
    assert r["hold_m3"] == pytest.approx(expected, rel=0.03)


def test_double_bottom_clamp():
    """이중저 높이 클램프 — B/15, 0.76~2.0 m (SOLAS 계보 C급)."""
    from src.physics.cargo_capacity import double_bottom_height_m
    assert double_bottom_height_m(6.0) == pytest.approx(0.76)   # B/15=0.4
    assert double_bottom_height_m(15.0) == pytest.approx(1.0)
    assert double_bottom_height_m(45.0) == pytest.approx(2.0)   # B/15=3.0


def test_standard_cargo_ship_passes():
    """Cb 0.75 표준 화물선 (8,000t) — 용적 합격 (기존 설계 생존)."""
    from src.ai.hull_generator import generate_hull_mesh
    from src.ai.dimension_estimator import estimate_dimensions
    from src.core.types import GoalSpec
    from src.physics.cargo_capacity import space_gate_large
    goal = GoalSpec(target_speed_ms=6.0, payload_kg=8_000_000.0,
                    purpose="cargo")
    dims = estimate_dimensions(goal)
    mesh = generate_hull_mesh(dims, cm=0.98, lcb_frac=0.02)
    r = space_gate_large(mesh, dims.depth, dims.loa,
                         fuel_t=100.0, payload_t=8_000.0)
    assert r["passed"] is True
    assert r["hold_m3"] > r["required_m3"]


def test_required_volume_stowage():
    """요구 용적 = payload × 1.3 m³/t (일반화물 적재계수 —
    초안 밀도 500 과대 요구를 실선 대역으로 정정)."""
    from src.physics.cargo_capacity import space_gate_large
    import trimesh
    mesh = trimesh.creation.box(extents=[100.0, 15.0, 8.0])
    r = space_gate_large(mesh, 8.0, 100.0, fuel_t=0.0,
                         payload_t=8_000.0)
    assert r["required_m3"] == pytest.approx(10_400.0)
    assert r["passed"] is False       # 상자 8,000t엔 공제 후 부족
