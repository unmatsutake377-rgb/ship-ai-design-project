import xml.etree.ElementTree as ET

import pytest
import yaml

from src.core.types import GoalSpec
from src.pipeline import run_pipeline
from src.sim_adapters.ros2_export import export_hydro_yaml, export_urdf


@pytest.fixture(scope="module")
def report_and_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("design")
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=100.0, purpose="survey")
    report = run_pipeline(goal, out)
    return report, out


def test_urdf_valid_xml_and_values(report_and_dir, tmp_path):
    report, design_dir = report_and_dir
    urdf_path = export_urdf(report, design_dir / report["mesh_file"], tmp_path)
    tree = ET.parse(urdf_path)  # 파싱 실패 = 불량 XML
    root = tree.getroot()
    assert root.tag == "robot"

    inertial = root.find("./link/inertial")
    mass = float(inertial.find("mass").get("value"))
    assert mass == pytest.approx(report["weights"]["total_mass"], rel=1e-6)
    izz = float(inertial.find("inertia").get("izz"))
    # izz는 성분 모델 하계와 iyy(0.25L 회전반경) 중 큰 값 (삼각 부등식 보장)
    assert izz >= report["weights"]["izz"] - 1e-9
    iyy_val = float(inertial.find("inertia").get("iyy"))
    ixx_val = float(inertial.find("inertia").get("ixx"))
    assert ixx_val + izz >= iyy_val  # Gazebo가 거부하던 위반 재발 방지
    ixx = float(inertial.find("inertia").get("ixx"))
    iyy = float(inertial.find("inertia").get("iyy"))
    # 횡동요(ixx) < 종동요(iyy): kxx=0.35B < kyy=0.25L은 B/L<0.714에서 항상
    # 성립 (우리 통통한 배 L/B 2도 포함). izz 비교는 선형별로 다름 — 안 함
    assert 0 < ixx < iyy

    mesh = root.find(".//visual/geometry/mesh")
    assert mesh.get("filename").endswith(".stl")


def test_hydro_yaml_roundtrip(report_and_dir, tmp_path):
    report, _ = report_and_dir
    path = export_hydro_yaml(report, tmp_path)
    data = yaml.safe_load(open(path))

    am = data["added_mass"]
    assert am["x_dot_u"] == pytest.approx(
        report["coefficients"]["xu_dot"], rel=1e-6)
    damp = data["linear_damping"]
    assert damp["yv"] == pytest.approx(
        report["coefficients"]["yv"], rel=1e-6)
    thr = data["thrusters"]
    assert thr["count"] == 2
    assert thr["max_thrust_n"] == pytest.approx(
        report["propulsion"]["motor"]["thrust_max_n"], rel=1e-6)
    # 부호 규약 문서화 필수 (크기 저장 — Fossen 조립 시 부호)
    assert "sign_convention" in data


def test_sdf_box_mode_matches_barge_analytic(report_and_dir, tmp_path):
    """box 모드: 초기 잠김 깊이 = 해석해 T = m/(ρ·L·B) (B-2c 1단계 근거)."""
    from src.sim_adapters.ros2_export import export_sdf

    report, design_dir = report_and_dir
    path = export_sdf(report, design_dir / report["mesh_file"], tmp_path,
                      collision="box")
    world = ET.parse(path).getroot()
    model = ET.parse(tmp_path / "model.sdf").getroot()

    mass = float(model.find(".//inertial/mass").text)
    assert mass == pytest.approx(report["weights"]["total_mass"], rel=1e-6)

    d = report["dimensions"]
    draft_analytic = mass / (1025.0 * d["loa"] * d["beam"])
    pose_z = float(world.find(".//include/pose").text.split()[2])
    assert pose_z == pytest.approx(d["depth"] / 2 - draft_analytic, abs=1e-6)

    # 부력 플러그인: 수상선용 graded 필수 (uniform=잠수함 가정 — 무한 상승
    # 사고 실측 후 교체, 2026-07-29). Gazebo 실검증: 정착 오차 0.1mm.
    buoy = [p for p in world.iter("plugin")
            if "Buoyancy" in p.get("name", "")]
    assert len(buoy) == 1
    graded = buoy[0].find("graded_buoyancy")
    assert graded is not None
    assert graded.find("default_density").text.strip() == "1025"


def test_sdf_mesh_mode_initial_pose_is_predicted_draft(report_and_dir,
                                                       tmp_path):
    from src.sim_adapters.ros2_export import export_sdf

    report, design_dir = report_and_dir
    path = export_sdf(report, design_dir / report["mesh_file"], tmp_path,
                      collision="mesh")
    world = ET.parse(path).getroot()
    pose_z = float(world.find(".//include/pose").text.split()[2])
    assert pose_z == pytest.approx(-report["hydrostatics"]["draft"], abs=1e-6)


def test_cli_writes_both_files(report_and_dir, tmp_path):
    import json
    import subprocess
    import sys

    report, design_dir = report_and_dir
    report_file = tmp_path / "report.json"
    with open(report_file, "w") as f:
        json.dump(report, f)
    out_dir = tmp_path / "export"
    r = subprocess.run(
        [sys.executable, "-m", "src.sim_adapters.ros2_export",
         "--report", str(report_file),
         "--mesh", str(design_dir / report["mesh_file"]),
         "--out", str(out_dir)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out_dir / "hull.urdf").exists()
    assert (out_dir / "hydrodynamics.yaml").exists()
