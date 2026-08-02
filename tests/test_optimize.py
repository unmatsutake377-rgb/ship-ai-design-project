"""NSGA-II 최적화 테스트 — 소형 실행 (pop 6 × gen 2, ~20초)."""
import numpy as np
import pytest

from src.core.types import GoalSpec
from src.optimize import dims_from_vector, evaluate_candidate, optimize_design

GOAL = GoalSpec(target_speed_ms=1.2, payload_kg=60.0, purpose="survey",
                endurance_h=4.0)


def test_dims_from_vector_roundtrip():
    dims = dims_from_vector(np.array([3.0, 2.0, 4.0, 0.5]))
    assert dims.loa == 3.0
    assert dims.loa / dims.beam == pytest.approx(2.0)
    assert dims.beam / dims.draft_design == pytest.approx(4.0)


def test_evaluate_candidate_feasible_case():
    # 60 kg 적재의 자연 크기(~1.8 m) 근방 — 3 m는 과대 크기라 GM 과대로
    # 정당하게 걸러짐 (필터 동작 확인함)
    r = evaluate_candidate(np.array([1.8, 2.0, 3.7, 0.5]), GOAL)
    assert r["feasible"], r.get("reason")
    assert r["resistance_n"] > 0
    assert r["stability_margin"] > 0


def test_evaluate_candidate_infeasible_fast_for_short_hull():
    """짧은 배 + 상대적 고속 → 반배수량 → 도태 페널티."""
    fast_goal = GoalSpec(target_speed_ms=1.5, payload_kg=60.0,
                         purpose="survey")
    r = evaluate_candidate(np.array([1.2, 2.0, 4.0, 0.5]), fast_goal)
    assert not r["feasible"]
    assert r["resistance_n"] >= 1e6


@pytest.fixture(scope="module")
def pareto():
    return optimize_design(GOAL, pop_size=6, n_gen=2, seed=3)


def test_optimizer_returns_feasible_candidates(pareto):
    assert len(pareto) >= 1
    assert pareto["feasible"].all()
    assert np.isfinite(pareto[["resistance_n", "total_mass_kg",
                               "stability_margin"]].to_numpy()).all()


def test_pareto_no_domination(pareto):
    """비지배 확인: 어떤 후보도 다른 후보를 전 목적에서 이기지 못함."""
    f = np.column_stack([pareto["resistance_n"], pareto["total_mass_kg"],
                         -pareto["stability_margin"]])
    n = len(f)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dominates = (f[i] <= f[j]).all() and (f[i] < f[j]).any()
            assert not dominates, f"{i}가 {j}를 지배"


def test_candidate_rejected_when_payload_too_bulky():
    """#27 후속: 무게는 실려도 공간(MaxBox)이 안 나오면 최적화 후보 도태.

    payload_volume을 괴물값(50 m³)으로 강제 — 어떤 소형 후보도 불가."""
    import numpy as np

    from src.core.types import GoalSpec
    from src.optimize import evaluate_candidate

    goal = GoalSpec(target_speed_ms=1.2, payload_kg=100.0, purpose="survey")
    x = np.array([3.0, 3.5, 4.0, 0.47])   # 정상이면 통과할 후보
    ok = evaluate_candidate(x, goal)
    assert ok["feasible"] is True          # 기본 밀도 경로는 통과
    bad = evaluate_candidate(x, goal, payload_volume=50.0)
    assert bad["feasible"] is False
    assert "공간" in bad["reason"]


def test_gate_trains_and_predicts():
    """문지기 학습: 라벨 500장 → 예측 확률 [0,1] 반환."""
    import numpy as np

    from src.optimize import _train_gate

    gate = _train_gate()
    assert gate is not None
    p, obj = gate.predict(np.array([[3.0, 3.5, 4.0, 0.47]]))
    assert 0.0 <= float(p[0]) <= 1.0


def test_gated_run_produces_feasible_front():
    """문지기 켠 소형 런: 전선 비어있지 않고 전부 실물리 검증 행."""
    from src.core.types import GoalSpec
    from src.optimize import optimize_design

    goal = GoalSpec(target_speed_ms=1.2, payload_kg=100.0, purpose="survey")
    df = optimize_design(goal, pop_size=8, n_gen=3, seed=3,
                         surrogate_gate=True)
    assert len(df) >= 1
    assert df["feasible"].all()
