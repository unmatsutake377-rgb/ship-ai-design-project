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


def test_screen_keeps_all_rows_with_labels(result):
    """탈락 행도 보존 — 대리모델 분류 학습용 (결손 라벨 버그 회귀 방지)."""
    assert len(result) == 8  # 표본 전체 보존
    ok = result[result["feasible"]]
    assert len(ok) >= 1
    for col in ("hull_id", "resistance_n", "total_mass_kg",
                "stability_margin", "pareto", "reason"):
        assert col in result.columns
    assert np.isfinite(ok[["resistance_n", "total_mass_kg"]]
                       .to_numpy()).all()
    # 탈락 행은 pareto=False + 사유 기록
    rejected = result[~result["feasible"]]
    if len(rejected):
        assert (~rejected["pareto"]).all()
        assert rejected["reason"].str.len().gt(0).all()


def test_pareto_marks_nondominated(result):
    ok = result[result["feasible"]].reset_index(drop=True)
    f = np.column_stack([ok["resistance_n"], ok["total_mass_kg"],
                         -ok["stability_margin"]])
    front = ok["pareto"].to_numpy()
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


def test_space_columns_present(result):
    """다구획 공간 열 (2026-08-03 재개방): feasible 의미는 보존."""
    for col in ("space_ok", "hold_volume_m3", "n_bays"):
        assert col in result.columns
    ok = result[result["feasible"]]
    # 파레토 표시는 공간 합격까지 요구
    assert (result[result["pareto"]]["space_ok"]).all()
    # 공간 불합격이어도 feasible(중량·안정) 라벨은 살아 있음 (라벨 호환)
    assert len(ok) >= 1
