"""파이프라인 Ship-D 편입 (스펙 4단계) — 수식 생성기 강등.

Ship-D 로컬 없으면 skip (formula 폴백은 아래 별도 시험이 담당).
"""
import pytest

from data import shipd_loader
from src.core.types import GoalSpec


def _goal(speed=1.2, payload=100.0, purpose="survey"):
    return GoalSpec(target_speed_ms=speed, payload_kg=payload,
                    purpose=purpose, endurance_h=4.0)


@pytest.mark.skipif(not shipd_loader.available(),
                    reason="Ship-D 로컬 사본 없음")
def test_pipeline_auto_judges_shipd_vs_formula(tmp_path):
    """auto: 실척 후보를 수식 기준선과 실측 비교 — 이긴 쪽 채택.

    어느 쪽이 이길지는 표본 운 — 판정 흔적(hull_note)과 관통이 계약."""
    from src.pipeline import run_pipeline

    report = run_pipeline(_goal(), tmp_path, shipd_pool=40)
    assert report["hull_source"] in ("shipd", "formula")
    assert report["hull_note"]           # 판정 사유 필수 (rel 병기)
    assert "배" in report["hull_note"]   # 비율 명시
    assert report["passed"]


@pytest.mark.skipif(not shipd_loader.available(),
                    reason="Ship-D 로컬 사본 없음")
def test_pipeline_shipd_forced_skips_guard(tmp_path):
    """--hull-source shipd: 사용자 강제 — 품질 방어 없이 실척 채택."""
    from src.pipeline import run_pipeline

    report = run_pipeline(_goal(), tmp_path, hull_source="shipd",
                          shipd_pool=40)
    assert report["hull_source"] == "shipd"
    assert report["hull_id"] >= 0
    assert report["passed"]


@pytest.mark.skipif(not shipd_loader.available(),
                    reason="Ship-D 로컬 사본 없음")
def test_pipeline_formula_override(tmp_path):
    """--hull-source formula: 기존 수식 경로 유지 (폴백 보존)."""
    from src.pipeline import run_pipeline

    report = run_pipeline(_goal(), tmp_path, hull_source="formula")
    assert report["hull_source"] == "formula"
    assert "hull_cm" in report


def test_pipeline_planing_keeps_formula(tmp_path):
    """활주 체계는 hull_source와 무관하게 Savitsky 수식 유지 (오너 철학)."""
    from src.pipeline import run_pipeline

    goal = GoalSpec(target_speed_ms=6.0, payload_kg=20.0, purpose="patrol",
                    endurance_h=1.0)   # 활주는 장항속 배터리 무게 비상식
    report = run_pipeline(goal, tmp_path, loa=2.8, hull_source="shipd")
    assert report["regime"] == "PLANING"
    assert report["hull_source"] == "formula"


def test_pipeline_semi_patrol_cm_wired(tmp_path):
    """patrol 반배수량: 트랜섬 선저 Cm 배선 (0.80 요청 — Cb에 따라
    정직 클램프될 수 있음, hull_cm이 기본 0.65보다 풍만해야 함)."""
    from src.pipeline import run_pipeline

    goal = GoalSpec(target_speed_ms=2.6, payload_kg=60.0, purpose="patrol",
                    endurance_h=2.0)
    report = run_pipeline(goal, tmp_path)
    assert report["regime"] == "SEMI_DISPLACEMENT"
    assert report["hull_cm"] is not None and report["hull_cm"] > 0.66
