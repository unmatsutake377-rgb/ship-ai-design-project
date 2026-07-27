import pytest

from src.hitl.elo import (
    INITIAL_RATING,
    K_FACTOR,
    compute_ratings,
    expected_score,
    record_comparison,
)


def test_expected_score_equal_ratings():
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_stronger_favored():
    assert expected_score(1700, 1500) > 0.75


def test_single_comparison_zero_sum(tmp_path):
    csv = tmp_path / "comparisons.csv"
    record_comparison("hull_A", "hull_B", csv)
    ratings = compute_ratings(csv)
    # 동점 출발 → 승자 +16, 패자 -16 (K=32, 기대 0.5)
    assert ratings["hull_A"] == pytest.approx(INITIAL_RATING + K_FACTOR * 0.5)
    assert ratings["hull_B"] == pytest.approx(INITIAL_RATING - K_FACTOR * 0.5)
    # 제로섬: 총점 보존
    assert sum(ratings.values()) == pytest.approx(2 * INITIAL_RATING)


def test_upset_gains_more(tmp_path):
    """이변(약자 승리)이 예상된 승리보다 점수 변동이 커야 함 — ELO 핵심."""
    csv = tmp_path / "c.csv"
    # A가 B를 3번 이겨 강자가 된 상태
    for _ in range(3):
        record_comparison("A", "B", csv)
    strong = compute_ratings(csv)
    gap_before = strong["A"] - strong["B"]
    # 이변: B가 A를 이김
    record_comparison("B", "A", csv)
    after = compute_ratings(csv)
    upset_gain = after["B"] - strong["B"]
    assert upset_gain > K_FACTOR * 0.5  # 예상승 이득(≤16)보다 큼
    assert after["A"] - after["B"] < gap_before


def test_replay_deterministic(tmp_path):
    csv = tmp_path / "c.csv"
    record_comparison("A", "B", csv)
    record_comparison("C", "A", csv)
    assert compute_ratings(csv) == compute_ratings(csv)


def test_missing_file_empty(tmp_path):
    assert compute_ratings(tmp_path / "none.csv") == {}


def test_self_comparison_rejected(tmp_path):
    with pytest.raises(ValueError, match="자기 자신"):
        record_comparison("A", "A", tmp_path / "c.csv")
