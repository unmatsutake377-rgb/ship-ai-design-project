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
                "resistance", "passed", "mesh_file"):
        assert key in report, key
    assert report["regime"] == "DISPLACEMENT"
    assert report["resistance"]["total"] > 0
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
