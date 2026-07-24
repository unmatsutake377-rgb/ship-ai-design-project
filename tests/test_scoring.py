import pytest

from src.hitl.scoring import load_scores, record_score


def test_record_and_load(tmp_path):
    csv = tmp_path / "user_scores.csv"
    record_score("hull_001", 4, csv)
    record_score("hull_002", 2, csv)
    df = load_scores(csv)
    assert list(df.columns) == ["hull_id", "score", "timestamp"]
    assert len(df) == 2
    assert df.iloc[0]["hull_id"] == "hull_001"
    assert df.iloc[0]["score"] == 4


def test_append_preserves_existing(tmp_path):
    csv = tmp_path / "user_scores.csv"
    record_score("a", 1, csv)
    record_score("a", 5, csv)  # 같은 선박 재평가 허용 (이력 보존)
    assert len(load_scores(csv)) == 2


@pytest.mark.parametrize("bad", [0, 6, -1, 3.5, "3"])
def test_reject_invalid_score(tmp_path, bad):
    with pytest.raises(ValueError):
        record_score("hull_001", bad, tmp_path / "s.csv")


def test_load_missing_file_returns_empty(tmp_path):
    df = load_scores(tmp_path / "none.csv")
    assert df.empty
    assert list(df.columns) == ["hull_id", "score", "timestamp"]
