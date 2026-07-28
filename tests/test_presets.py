import pytest

from src.ai.presets import purpose_presets


def test_survey_speed_is_data_driven():
    p = purpose_presets()
    assert p["survey"].speed_source == "data"
    assert p["survey"].n_samples >= 2
    assert 0.5 < p["survey"].default_speed_ms < 3.5  # 실선 순항 상식 범위


def test_patrol_falls_back_with_label():
    """patrol은 순항속도 실측 없음 (활주정뿐) — 개략값 + 출처 표시."""
    p = purpose_presets()
    assert p["patrol"].speed_source == "fallback"
    assert p["patrol"].n_samples == 0


def test_missing_csv_degrades_to_fallback(tmp_path):
    p = purpose_presets(tmp_path / "none.csv")
    assert all(v.speed_source == "fallback" for v in p.values())
    assert set(p) == {"survey", "patrol", "workboat"}
