"""8번째 게이트 — run_pipeline EEDI·경제 통합 (경제성 3단계)."""
import pytest

from src.core.types import GoalSpec
from src.pipeline import run_pipeline


def test_large_cargo_economics_gate(tmp_path):
    """8000t 화물선 — EEDI 판정 + 운항 경제 성적표.

    1단계 실측: 현행 프리셋 13.6kn은 불합격 예상 — 전체 passed에
    반영되는지 확인 (규제가 설계를 심판하는 순간)."""
    goal = GoalSpec(target_speed_ms=7.0, payload_kg=8_000_000.0,
                    purpose="cargo")
    report = run_pipeline(goal, tmp_path, seakeeping=False,
                          structure=False, maneuvering=False)
    ec = report["economics"]
    assert ec is not None
    assert ec["kind"] == "eedi"
    assert ec["applicable"] is True          # DWT 8000 ≥ 3000
    assert ec["attained_g_per_tnm"] > 0
    assert ec["fuel_cost_usd_per_year"] > 0
    # CII 병기 (운항 성적표 — 게이트 아님)
    cii = ec["cii"]
    assert cii["rating_2026"] in "ABCDE"
    assert len(cii["outlook"]) == 4
    assert cii["aer_g_per_dwt_nm"] > 0
    # 판정과 전체 passed 합성
    if not ec["passed"]:
        assert report["passed"] is False


def test_small_survey_electric_report(tmp_path):
    """2m 조사선 — 전기 등가 성적표 (규제 아님·항상 통과)."""
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=100.0,
                    purpose="survey")
    report = run_pipeline(goal, tmp_path, hull_source="formula",
                          seakeeping=False, structure=False)
    ec = report["economics"]
    assert ec is not None
    assert ec["kind"] == "electric"
    assert ec["wh_per_kg_km"] > 0
    assert ec["passed"] is True
    assert ec["applicable"] is False


def test_economics_flag_off(tmp_path):
    """economics=False — 게이트 생략."""
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=100.0,
                    purpose="survey")
    report = run_pipeline(goal, tmp_path, hull_source="formula",
                          seakeeping=False, structure=False,
                          economics=False)
    assert report["economics"] is None
