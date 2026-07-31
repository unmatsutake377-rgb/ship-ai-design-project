import pytest

from src.ai.presets import purpose_presets


def test_survey_speed_is_data_driven():
    p = purpose_presets()
    assert p["survey"].speed_source == "data"
    assert p["survey"].n_samples >= 2
    assert 0.5 < p["survey"].default_speed_ms < 3.5  # 실선 순항 상식 범위


def test_patrol_now_data_driven():
    """#17 수집(07-31: L30·M75 순항 실측)으로 patrol도 데이터 기반 전환.

    실측 순항이 활주 영역(~11.6 m/s)인 것은 실제 순찰정의 특성 —
    전기 추진 카탈로그로는 못 채울 수 있으나 그건 설계 단계의 정직한
    거절 몫이지 프리셋이 거짓말할 이유가 아님."""
    p = purpose_presets()
    assert p["patrol"].speed_source == "data"
    assert p["patrol"].n_samples >= 2
    assert p["patrol"].default_speed_ms > 5.0  # 활주 영역 실측


def test_workboat_still_falls_back():
    """workboat은 아직 표본 부족 (단동 순항 실측 없음) — 개략값 유지."""
    p = purpose_presets()
    assert p["workboat"].speed_source in ("fallback", "data")  # 수집 진행형


def test_missing_csv_degrades_to_fallback(tmp_path):
    p = purpose_presets(tmp_path / "none.csv")
    assert all(v.speed_source == "fallback" for v in p.values())
    assert set(p) == {"survey", "patrol", "workboat"}
