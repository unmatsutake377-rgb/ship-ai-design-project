"""쌍동선 1단계 — 평행축·실선·저항 항등식 앵커 (스펙 2026-08-10)."""
import numpy as np
import pytest
import trimesh

RHO = 1025.0


def test_parallel_axis_hand_calc():
    """평행축 손계산 — 상자 데미헐 2개 (각 L10×b1, 간격 s=4):
    I_T = 2×(L·b³/12 + L·b·(s/2)²) = 2×(0.833 + 40) = 81.67 m⁴.

    병합 메쉬의 waterplane_properties가 이 값을 실측해야 함
    (기존 정역학이 쌍동을 자동 지원하는지 — 1단계 심장)."""
    from src.physics.hydrostatics import waterplane_properties
    demi = trimesh.creation.box(extents=[10.0, 1.0, 2.0])
    left = demi.copy()
    left.apply_translation([0, -2.0, 0])
    right = demi.copy()
    right.apply_translation([0, +2.0, 0])
    cat = trimesh.util.concatenate([left, right])
    area, ixx = waterplane_properties(cat, 0.0)   # 중앙 수선
    assert area == pytest.approx(2 * 10.0 * 1.0, rel=0.01)
    expected = 2.0 * (10.0 * 1.0 ** 3 / 12.0 + 10.0 * 1.0 * 2.0 ** 2)
    assert ixx == pytest.approx(expected, rel=0.01)


def test_catamaran_mesh_and_gm_monotonic():
    """생성기 관통 + 간격 단조: s↑ → GM↑ (평행축 방향)."""
    from src.ai.catamaran import generate_catamaran_mesh
    from src.ai.dimension_estimator import estimate_dimensions
    from src.core.types import GoalSpec
    from src.physics.hydrostatics import evaluate
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=50.0,
                    purpose="survey")
    dims = estimate_dimensions(goal)
    gms = []
    for ratio in (0.5, 0.8):
        mesh = generate_catamaran_mesh(dims, separation_ratio=ratio)
        assert mesh.is_watertight or len(mesh.faces) > 100
        h = evaluate(mesh, 60.0, kg=dims.depth * 0.5,
                     beam=dims.beam, depth=dims.depth)
        gms.append(h.gm)
    assert gms[1] > gms[0] > 0.0


def test_blueboat_scale_sanity():
    """BlueBoat급 (L 1.2·전폭 0.93 — 실선 L/B 1.29 계보) —
    흘수·GM 자릿수 sanity (얕은 흘수·강복원)."""
    from src.ai.catamaran import generate_catamaran_mesh
    from src.core.types import MainDimensions
    from src.physics.hydrostatics import evaluate
    dims = MainDimensions(loa=1.2, beam=0.93, depth=0.25,
                          draft_design=0.12, cb=0.45)
    mesh = generate_catamaran_mesh(dims, separation_ratio=0.75)
    h = evaluate(mesh, 15.0, kg=0.15, beam=dims.beam,
                 depth=dims.depth)
    assert 0.03 < h.draft < 0.25          # 얕은 흘수 자릿수
    assert h.gm > 0.3                     # 쌍동 강복원 (단동 대비 큼)


def test_resistance_twice_demihull():
    """저항 항등식 — 쌍동 저항 = 데미헐 단동 저항 × 2
    (간섭 무시 C급, 스펙 §2)."""
    from src.ai.catamaran import catamaran_resistance
    from src.ai.dimension_estimator import estimate_dimensions
    from src.core.types import GoalSpec
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=50.0,
                    purpose="survey")
    dims = estimate_dimensions(goal)
    r = catamaran_resistance(dims, separation_ratio=0.7,
                             draft=dims.draft_design, speed_ms=1.5)
    assert r["total_n"] == pytest.approx(2.0 * r["demihull_n"],
                                         rel=1e-9)
    assert "간섭" in r["note"]


def test_station_area_catamaran_generalized():
    """스캔라인 일반화 — 쌍동 단면(폐곡선 2개)의 면적 = 데미헐×2.

    현행 max-y 방식은 한쪽 몸통만 읽어 부력 절반 오독 — 짝수 교차
    구간 합산으로 수리 (단동 결과는 불변이어야 함)."""
    from src.physics.structure.loads import station_area
    demi = trimesh.creation.box(extents=[10.0, 1.0, 2.0])
    left = demi.copy(); left.apply_translation([0, -2.0, 0])
    right = demi.copy(); right.apply_translation([0, +2.0, 0])
    cat = trimesh.util.concatenate([left, right])
    a_cat = station_area(cat, 0.0, 0.0)      # 흘수 1 m (z -1~0)
    assert a_cat == pytest.approx(2.0 * 1.0 * 1.0, rel=0.02)
    # 단동 회귀 불변
    mono = trimesh.creation.box(extents=[10.0, 4.0, 2.0])
    a_mono = station_area(mono, 0.0, 0.0)
    assert a_mono == pytest.approx(4.0 * 1.0, rel=0.02)


def test_pipeline_catamaran_e2e(tmp_path):
    """파이프라인 쌍동 관통 — BlueBoat급 조사선."""
    from src.core.types import GoalSpec
    from src.pipeline import run_pipeline
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=50.0,
                    purpose="survey")
    report = run_pipeline(goal, tmp_path, hull_source="formula",
                          seakeeping=False, structure=False,
                          catamaran=True)
    assert report["hull_family"] == "catamaran"
    assert report["hydrostatics"]["gm"] > 0.2   # 쌍동 강복원
    assert report["resistance"]["total"] > 0
    assert "간섭" in report["catamaran"]["note"]


def test_catamaran_seakeeping_gate():
    """쌍동 내항 실판정 — 데미헐 등가 (상호작용 무시 C급).

    선형계 등가: 계수·기진·질량 전부 ×2 → heave·pitch RAO는
    '데미헐 + 절반 질량' 문제와 동일. roll은 쌍동 실측 GM →
    공진 주기 단축 자동."""
    from src.ai.catamaran import seakeeping_gate_catamaran
    from src.ai.dimension_estimator import estimate_dimensions
    from src.core.types import GoalSpec
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=50.0,
                    purpose="survey")
    dims = estimate_dimensions(goal)
    r = seakeeping_gate_catamaran(dims, separation_ratio=0.7,
                                  total_mass=60.0, gm=0.32,
                                  purpose="survey")
    assert not r.get("skipped")
    assert r["sig_roll_deg"] >= 0
    assert "데미헐" in r["note"]


def test_catamaran_roll_period_shorter():
    """물리 방향 — 쌍동 GM(큼) → roll 공진 주기 < 단동."""
    from src.physics.seakeeping.criteria import roll_natural_period
    t_mono = roll_natural_period(0.93, 0.12, 1.2, 0.16)
    t_cat = roll_natural_period(0.93, 0.12, 1.2, 0.32)
    assert t_cat < t_mono


def test_pipeline_catamaran_seakeeping_real(tmp_path):
    """e2e — 쌍동 내항이 스킵 아닌 실판정 (3단계 승격)."""
    from src.core.types import GoalSpec
    from src.pipeline import run_pipeline
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=50.0,
                    purpose="survey")
    report = run_pipeline(goal, tmp_path, hull_source="formula",
                          structure=False, catamaran=True)
    sk = report["seakeeping"]
    assert sk is not None and not sk.get("skipped")
    assert "sig_roll_deg" in sk
