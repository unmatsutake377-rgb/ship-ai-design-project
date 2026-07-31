"""훅 CLI — 생성 모드와 파싱 모드 (OpenFOAM 불필요)."""
import json
import subprocess
import sys

import trimesh


def _fake_report_dir(d):
    d.mkdir(parents=True)
    box = trimesh.creation.box(bounds=[[-1.85, -0.5, 0.0], [1.85, 0.5, 0.6]])
    box.export(d / "hull.stl")
    report = {"goal": {"target_speed_ms": 1.5},
              "dimensions": {"loa": 3.7, "beam": 0.99},
              "hydrostatics": {"draft": 0.25},
              "mesh_file": "hull.stl",
              "resistance": {"rf": 11.0, "rw": 7.0, "total": 18.0}}
    (d / "report.json").write_text(json.dumps(report))
    return d


def test_cli_build_prints_next_step(tmp_path):
    rep = _fake_report_dir(tmp_path / "rep")
    result = subprocess.run(
        [sys.executable, "-m", "src.cfd.hook", "--report", str(rep),
         "--mode", "simple", "--out", str(tmp_path / "case")],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "run_case.sh" in result.stdout       # 다음 단계 안내
    assert (tmp_path / "case" / "system" / "controlDict").exists()


def test_cli_parse_only_appends_label(tmp_path):
    rep = _fake_report_dir(tmp_path / "rep")
    case = tmp_path / "case"
    subprocess.run([sys.executable, "-m", "src.cfd.hook", "--report",
                    str(rep), "--mode", "simple", "--out", str(case)],
                   capture_output=True, text=True)
    # 가짜 실행 결과 심기
    d = case / "postProcessing" / "forces" / "0"
    d.mkdir(parents=True)
    rows = "".join(f"{t}\t(10 0 0)\t(6 0 0)\t(4 0 0)\n" for t in range(10))
    (d / "force.dat").write_text("# header\n" + rows)
    csv = tmp_path / "labels.csv"
    result = subprocess.run(
        [sys.executable, "-m", "src.cfd.hook", "--report", str(rep),
         "--mode", "simple", "--out", str(case),
         "--parse-only", "--labels", str(csv)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert csv.exists()
    assert "20.0" in result.stdout              # ×2 적용된 전저항
