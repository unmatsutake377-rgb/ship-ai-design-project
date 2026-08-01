import json
import math
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
                "resistance", "propulsion", "coefficients", "passed",
                "mesh_file"):
        assert key in report, key
    assert isinstance(report["coefficients"]["straight_line_stable"], bool)
    assert report["coefficients"]["extrapolation_warning"] is True
    assert report["weights"]["izz"] > 0
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


def test_pipeline_semi_displacement_designs(tmp_path):
    """Phase C-1: 반배수량 요청이 이제 설계됨 — 트랜섬 계열 경로."""
    goal = GoalSpec(target_speed_ms=3.0, payload_kg=100.0, purpose="patrol")
    report = run_pipeline(goal, tmp_path)
    assert report["regime"] == "SEMI_DISPLACEMENT"
    assert report["hull_family"] == "transom"
    assert report["resistance"]["total"] > 0
    assert report["passed"] in (True, False)  # 필터는 정상 판정


def test_pipeline_planing_designs(tmp_path):
    """Phase C-2: 활주 요청이 이제 설계됨 — Savitsky 평형 경로.

    저항 성분: total = 유도(Δ·tanτ = rw) + 바닥마찰/cosτ (rf=Df).
    τ를 리포트에 안 실으므로 부등식으로 검사: rf+rw ≤ total ≤ rf/cos15°+rw."""
    goal = GoalSpec(target_speed_ms=6.0, payload_kg=20.0, purpose="patrol",
                    endurance_h=1.0)
    report = run_pipeline(goal, tmp_path, loa=2.8)
    assert report["regime"] == "PLANING"
    assert report["hull_family"] == "planing_deadrise"
    r = report["resistance"]
    assert r["rf"] > 0 and r["rw"] > 0
    assert r["rf"] + r["rw"] <= r["total"] * 1.001
    assert r["total"] <= r["rf"] / math.cos(math.radians(15.0)) + r["rw"] + 1e-6
    # 활주용 GM/B 완화 밴드(0.04~1.50)로 정역학 필터도 통과해야 함
    assert report["passed"] is True


def test_maxbox_space_check_rejects_oversized(tmp_path):
    """#27: 무게는 실려도 부피가 안 들어가면 불합격."""
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=100.0, purpose="survey")
    report = run_pipeline(goal, tmp_path, payload_volume=50.0)  # 50 m³ 괴물
    assert report["checks_space"] is False
    assert report["passed"] is False
    assert report["maxbox"]["volume"] < 50.0


def test_maxbox_default_density_passes(tmp_path):
    """기본 밀도 환산 경로: 기존 데모(100 kg 조사장비)는 공간 합격 유지."""
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=100.0, purpose="survey")
    report = run_pipeline(goal, tmp_path)
    assert report["checks_space"] is True
    assert "밀도 가정" in report["maxbox"]["volume_basis"]
    assert report["maxbox"]["margin_ratio"] > 0


def test_report_contains_speed_limit(tmp_path):
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=100.0, purpose="survey")
    report = run_pipeline(goal, tmp_path)
    vmax = report["max_displacement_speed"]
    assert vmax > report["goal"]["target_speed_ms"]  # 통과했으니 여유 있어야


def test_formerly_rejected_speed_now_reaches_physics(tmp_path):
    """C-2 이전 '체계 미지원' 거절 케이스 — 이제 물리까지 간다.

    6 m/s·100 kg는 모터 카탈로그 한계로 실패하지만, 사유가 '체계
    미지원'이 아니라 물리·카탈로그(추력/평형)여야 한다. exit 3은
    이제 '설계 불가' 일반 코드."""
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline",
         "--speed", "6.0", "--payload", "100", "--purpose", "survey",
         "--out", str(tmp_path)],
        capture_output=True, text=True,
    )
    err = result.stdout + result.stderr
    assert "추후 지원" not in err and "미지원" not in err
    if result.returncode == 3:  # 정직한 거절 — 사유는 물리/카탈로그
        # 카탈로그 확장(08-02, Riptide 2종)으로 모터 벽은 통과 —
        # 이제 활주 평형(LCG-압력중심)까지 도달해 거절됨 (층 심화)
        assert ("모터" in err or "추력" in err or "평형" in err
                or "압력중심" in err or "LCG" in err)
    else:
        assert result.returncode in (0, 2)


def test_cli_speed_optional_uses_preset(tmp_path):
    """3입력 UX (#25): 속도 생략 → 용도 프리셋 (실선 순항 중앙값)."""
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline",
         "--payload", "100", "--purpose", "survey", "--out", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode in (0, 2), result.stderr
    assert "용도 프리셋 적용" in result.stdout
    assert "실선" in result.stdout  # survey는 데이터 근거


def test_cli_smoke(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline",
         "--speed", "1.5", "--payload", "100", "--purpose", "survey",
         "--out", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode in (0, 2), result.stderr
    assert "GM" in result.stdout


