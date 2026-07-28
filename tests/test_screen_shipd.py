"""Ship-D 선별 테스트 — 로컬 사본 없으면 skip. 소형 표본(8척) 스모크."""
import numpy as np
import pytest

from data import shipd_loader

pytestmark = pytest.mark.skipif(
    not shipd_loader.available(),
    reason="Ship-D 로컬 사본 없음 (data/shipd/)",
)

from src.core.types import GoalSpec

GOAL = GoalSpec(target_speed_ms=1.2, payload_kg=100.0, purpose="survey",
                endurance_h=4.0)


@pytest.fixture(scope="module")
def result():
    from src.screen_shipd import screen

    return screen(GOAL, target_loa=3.0, n_samples=8, seed=11)


def test_screen_returns_some_feasible(result):
    assert len(result) >= 1
    assert result["feasible"].all()
    for col in ("hull_id", "resistance_n", "total_mass_kg",
                "stability_margin", "pareto"):
        assert col in result.columns
    assert np.isfinite(result[["resistance_n", "total_mass_kg"]]
                       .to_numpy()).all()


def test_pareto_marks_nondominated(result):
    f = np.column_stack([result["resistance_n"], result["total_mass_kg"],
                         -result["stability_margin"]])
    front = result["pareto"].to_numpy()
    assert front.any()
    for i in np.where(front)[0]:
        for j in range(len(f)):
            if i == j:
                continue
            assert not ((f[j] <= f[i]).all() and (f[j] < f[i]).any()), \
                f"파레토 표시 {i}가 {j}에 지배됨"


def test_single_evaluation_reports_reason_on_failure():
    from src.screen_shipd import evaluate_shipd_hull

    vectors, _ = shipd_loader.load_vectors()
    fast_goal = GoalSpec(target_speed_ms=5.0, payload_kg=100.0,
                         purpose="survey")
    r = evaluate_shipd_hull(vectors[0], fast_goal, target_loa=3.0)
    assert not r["feasible"]
    assert "반배수량" in r["reason"]
