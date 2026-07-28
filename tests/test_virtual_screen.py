"""가상 선별 스모크 — Ship-D + screen.csv 실측 결과 필요 시에만."""
from pathlib import Path

import pytest

from data import shipd_loader

SCREEN_CSV = Path("outputs/shipd_pareto/screen.csv")

pytestmark = pytest.mark.skipif(
    not (shipd_loader.available() and SCREEN_CSV.exists()),
    reason="Ship-D 또는 실측 screen.csv 없음",
)

from src.core.types import GoalSpec

GOAL = GoalSpec(target_speed_ms=1.2, payload_kg=100.0, purpose="survey",
                endurance_h=4.0)


def test_virtual_screen_smoke():
    from src.virtual_screen import virtual_screen

    combined, metrics = virtual_screen(GOAL, target_loa=3.0,
                                       screen_csv=SCREEN_CSV,
                                       top_k=3, epochs=120, seed=2)
    # 합산 결과: 기존 실측 + 재검증 신규, 전부 실물리 수치
    assert combined["feasible"].all()
    assert combined["pareto"].any()
    assert set(combined["source"]) <= {"screen300", "surrogate_pick"}
    assert 0.0 <= metrics["reverify_pass_rate"] <= 1.0
    assert metrics["n_train"] > metrics["n_val"]
