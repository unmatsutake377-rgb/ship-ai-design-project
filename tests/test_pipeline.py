import json
import subprocess
import sys

import pytest

from src.core.types import GoalSpec
from src.pipeline import run_pipeline


def test_run_pipeline_survey(tmp_path):
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=100.0, purpose="survey")
    report = run_pipeline(goal, tmp_path)
    # 리포트 필수 필드
    for key in ("goal", "dimensions", "regime", "weights", "hydrostatics",
                "resistance", "propulsion", "passed", "mesh_file"):
        assert key in report, key
    assert report["regime"] == "DISPLACEMENT"
    assert report["resistance"]["total"] > 0
    assert report["propulsion"]["count"] == 2
    assert report["propulsion"]["total_thrust_n"] >= 2 * report["resistance"]["total"]


def test_design_spiral_converges_and_realistic(tmp_path):
    """나선 수렴 + 실측 추진계가 고정비율(15%) 개략보다 가벼워야 함."""
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=100.0, purpose="survey")
    report = run_pipeline(goal, tmp_path)
    p = report["propulsion"]
    w = report["weights"]
    assert 2 <= p["spiral_iterations"] <= 12
    assert p["battery_mass_kg"] > 0
    # 실측 추진계(모터+배터리) < 옛 고정비율 15%·전체
    assert w["propulsion_mass"] < 0.15 * w["total_mass"]
    # 중량 폐합: 전체 = 구조 + 적재 + 추진
    assert w["total_mass"] == pytest.approx(
        w["structure_mass"] + w["payload_mass"] + w["propulsion_mass"],
        rel=1e-6,
    )
    # 산출물 파일 존재
    assert (tmp_path / report["mesh_file"]).exists()
    assert (tmp_path / "report.json").exists()
    with open(tmp_path / "report.json") as f:
        assert json.load(f)["passed"] == report["passed"]


def test_pipeline_rejects_fast_speed(tmp_path):
    """반배수량 영역 요청 → 명시적 중단 (spec §2.1)."""
    from src.core.regime import UnsupportedRegimeError

    goal = GoalSpec(target_speed_ms=5.0, payload_kg=100.0, purpose="survey")
    with pytest.raises(UnsupportedRegimeError):
        run_pipeline(goal, tmp_path)


def test_report_contains_speed_limit(tmp_path):
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=100.0, purpose="survey")
    report = run_pipeline(goal, tmp_path)
    vmax = report["max_displacement_speed"]
    assert vmax > report["goal"]["target_speed_ms"]  # 통과했으니 여유 있어야


def test_rejection_message_has_alternatives(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline",
         "--speed", "6.0", "--payload", "100", "--purpose", "survey",
         "--out", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 3
    err = result.stdout + result.stderr
    assert "한계속도" in err   # 이 크기가 낼 수 있는 속도
    assert "최소" in err       # 이 속도에 필요한 길이


def test_cli_smoke(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline",
         "--speed", "1.5", "--payload", "100", "--purpose", "survey",
         "--out", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode in (0, 2), result.stderr
    assert "GM" in result.stdout


def test_cli_unsupported_regime_exit_code(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline",
         "--speed", "6.0", "--payload", "100", "--purpose", "survey",
         "--out", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 3
    assert "배수량형" in (result.stdout + result.stderr)
