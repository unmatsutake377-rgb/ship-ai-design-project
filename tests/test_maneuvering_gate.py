"""7번째 게이트 — run_pipeline 조종성 통합 (3단계, 스펙 §4)."""
import pytest

from src.core.types import GoalSpec
from src.pipeline import run_pipeline


def test_imo_limits_formulas():
    """IMO 오버슈트 한계 구간식 — 원전 §5.3.3 재현."""
    from src.physics.maneuvering.criteria import (
        imo_first_overshoot_limit_deg,
        imo_second_overshoot_limit_deg,
    )
    assert imo_first_overshoot_limit_deg(5.0) == 10.0
    assert imo_first_overshoot_limit_deg(20.0) == pytest.approx(15.0)
    assert imo_first_overshoot_limit_deg(40.0) == 20.0
    assert imo_second_overshoot_limit_deg(20.0) == pytest.approx(32.5)


def test_large_cargo_gate_runs(tmp_path):
    """대형 화물선 — 조종 성적표 생성 + IMO 판정 (L≥100 적용)."""
    goal = GoalSpec(target_speed_ms=7.0, payload_kg=8_000_000.0,
                    purpose="cargo")
    report = run_pipeline(goal, tmp_path, seakeeping=False,
                          structure=False)
    mv = report["maneuvering"]
    assert mv is not None
    if mv.get("skipped"):
        pytest.fail(f"조종 게이트 스킵됨: {mv['note']}")
    assert mv["tactical_diameter_over_l"] > 0
    assert mv["coeff_grade"] in ("B", "C")
    loa = report["dimensions"]["loa"]
    if loa >= 100.0:
        assert mv["applicable"] is True
        # 판정 결과와 전체 passed 합성 확인
        if not mv["passed"]:
            assert report["passed"] is False
    else:
        assert mv["applicable"] is False
        assert mv["passed"] is True     # 성적표만 — 게이트 미적용


def test_maneuvering_flag_off(tmp_path):
    """maneuvering=False — 게이트 생략."""
    goal = GoalSpec(target_speed_ms=7.0, payload_kg=5_000_000.0,
                    purpose="cargo")
    report = run_pipeline(goal, tmp_path, seakeeping=False,
                          structure=False, maneuvering=False)
    assert report["maneuvering"] is None
