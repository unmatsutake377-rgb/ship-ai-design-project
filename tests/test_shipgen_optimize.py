"""ShipGen 45파라미터 NSGA (스펙 3단계) — Ship-D 로컬 없으면 skip."""
import numpy as np
import pytest

from data import shipd_loader

pytestmark = pytest.mark.skipif(
    not shipd_loader.available(),
    reason="Ship-D 로컬 사본 없음",
)


def test_bounds_from_dataset_cover_real_ships():
    """탐색 경계 = 실값 분포 (하드코딩 아님) — 전 실척이 경계 안."""
    from src.ai.shipgen_optimize import dataset_bounds

    lo, hi = dataset_bounds()
    vectors, _ = shipd_loader.load_vectors()
    assert lo.shape == (45,) and hi.shape == (45,)
    assert np.all(lo <= vectors.min(axis=0))
    assert np.all(hi >= vectors.max(axis=0))
    # 열 0(LOA)은 전 척 10.0 정규화 — 범위 0이 정상 (크기는 scaled_mesh
    # 담당). 변이 폭이 (xu-xl) 비례라 상수 열은 탐색에서 자동 보존.
    assert lo[0] == hi[0] == pytest.approx(10.0)
    assert np.all(hi[1:] > lo[1:])


def test_constraints_ok_detects_violation():
    """원저자 제약 49개: 실척 통과, 고의 위반 검출."""
    from src.ai.shipgen_optimize import constraints_ok

    vectors, _ = shipd_loader.load_vectors()
    assert constraints_ok(vectors[900])
    bad = vectors[900].copy()
    bad[1] = -5.0
    assert not constraints_ok(bad)


def test_smoke_optimize_returns_feasible_front():
    """스모크 (pop 8 × gen 2): 전선이 실측 4중 게이트 통과자만."""
    from src.ai.shipgen_optimize import optimize_shipgen
    from src.core.types import GoalSpec

    goal = GoalSpec(target_speed_ms=1.2, payload_kg=100.0,
                    purpose="survey", endurance_h=4.0)
    df = optimize_shipgen(goal, 3.0, pop_size=8, n_gen=2, seed=5)
    assert len(df) >= 1
    assert df["feasible"].all()
    assert (df["resistance_n"] > 0).all()
    assert "vector_json" in df.columns   # 재현 가능 (45파라미터 보존)


def test_subspace_keeps_fixed_params():
    """부분공간: 자유 11개만 변하고 선수미·벌브는 기준 척 값 유지."""
    import json

    from src.ai.shipgen_optimize import SUBSPACE_MAIN_MID, optimize_shipgen
    from src.core.types import GoalSpec

    vectors, _ = shipd_loader.load_vectors()
    base = vectors[29813]
    goal = GoalSpec(target_speed_ms=1.2, payload_kg=100.0,
                    purpose="survey", endurance_h=4.0)
    df = optimize_shipgen(goal, 3.0, pop_size=6, n_gen=2, seed=5,
                          free_idx=SUBSPACE_MAIN_MID, base_vector=base)
    assert len(df) >= 1
    v = np.array(json.loads(df["vector_json"].iloc[0]))
    assert np.allclose(v[11:], base[11:])   # 고정부 보존
