"""6번째 게이트 — run_pipeline 종강도 통합 (구조 3단계, 스펙 §4)."""
import pytest

from src.core.types import GoalSpec
from src.pipeline import run_pipeline


def test_small_survey_gate_runs(tmp_path):
    """2m 조사선 — 소형 경로: 준정적 표준파, 알루 재료."""
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=100.0,
                    purpose="survey")
    report = run_pipeline(goal, tmp_path, hull_source="formula",
                          seakeeping=False)
    st = report["structure"]
    assert st is not None
    if st.get("skipped"):
        pytest.fail(f"소형 구조 게이트 스킵됨: {st['note']}")
    assert "표준파" in st["wave_source"]
    assert st["material"] == "al5083"
    assert st["t_bottom_mm"] > 0
    # 게이트 합성: 구조 불합격이면 전체 불합격
    if not st["passed"]:
        assert report["passed"] is False


def test_structure_flag_off(tmp_path):
    """structure=False — 게이트 생략 (기존 시험 격리 관례)."""
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=100.0,
                    purpose="survey")
    report = run_pipeline(goal, tmp_path, hull_source="formula",
                          seakeeping=False, structure=False)
    assert report["structure"] is None


def test_100m_cargo_gate_iacs(tmp_path):
    """100m 화물선 — IACS 하중 경로, 규칙 두께 합격 기대
    (2단계 비 1.06 실증의 게이트 판)."""
    goal = GoalSpec(target_speed_ms=7.0, payload_kg=5_000_000.0,
                    purpose="cargo")
    report = run_pipeline(goal, tmp_path, seakeeping=False)
    st = report["structure"]
    assert st is not None
    if st.get("skipped"):
        pytest.fail(f"대형 구조 게이트 스킵됨: {st['note']}")
    assert "IACS" in st["wave_source"]
    assert st["material"] == "mild_steel"
    assert st["passed"] is True
    assert st["z_deck_m3"] >= st["z_required_m3"]
    # 정수 + 파랑 모멘트 부호·자릿수 (1단계 실측 계보)
    assert st["m_wave_hog_knm"] > 0 > st["m_wave_sag_knm"]
    assert report["passed"] == (report["large"]["passed"]
                                and st["passed"])
