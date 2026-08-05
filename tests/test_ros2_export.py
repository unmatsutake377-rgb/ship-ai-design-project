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
    report = run_pipeline(goal, out, hull_source="formula")
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
    iyy_val = float(inertial.find("inertia").get("iyy"))
    ixx_val = float(inertial.find("inertia").get("ixx"))
    # izz 하계 = 성분 모델과 iyy 중 큰 값 — 단 물리 상한
    # izz ≤ 0.999(ixx+iyy)가 우선 (Ship-D 실척은 성분 모델이 상한을
    # 넘을 수 있어 절단 발동 — 절단됐다면 상한 근처가 정답)
    lower = min(max(report["weights"]["izz"], iyy_val),
                0.999 * (ixx_val + iyy_val))
    assert izz >= lower - 1e-9
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


def test_sdf_mesh_mode_envelope_box_buoyancy(report_and_dir, tmp_path):
    """mesh 모드: 시각=메쉬, 부력체=외피 상자 (gz 한계 — 메쉬 부력 미지원·
    등가 부피 상자는 전복. 실유체 정합은 B-3 계수 플러그인 과제)."""
    from src.sim_adapters.ros2_export import export_sdf

    report, design_dir = report_and_dir
    path = export_sdf(report, design_dir / report["mesh_file"], tmp_path,
                      collision="mesh")
    world = ET.parse(path).getroot()
    model = ET.parse(tmp_path / "model.sdf").getroot()

    d = report["dimensions"]
    t_box = report["weights"]["total_mass"] / (1025.0 * d["loa"] * d["beam"])
    pose_z = float(world.find(".//include/pose").text.split()[2])
    # 링크 좌표계 = 상자 중심 (collision 오프셋 금지 — 피치 발산 실측)
    assert pose_z == pytest.approx(d["depth"] / 2 - t_box, abs=1e-6)
    # 부력 collision은 원점 상자, 시각은 −D/2 내린 메쉬
    assert model.find(".//collision/geometry/box") is not None
    assert model.find(".//visual/geometry/mesh") is not None
    cg_pose = model.find(".//inertial/pose").text.split()
    assert float(cg_pose[0]) == 0.0  # 세로 CG 오프셋 0 (gz 트림 한계 회피)
    # 유체 플러그인은 **모델 안** (월드 레벨 = 조용히 무효, 07-29 실측)
    hydro_w = [p for p in world.iter("plugin")
               if "Hydrodynamics" in p.get("name", "")]
    assert len(hydro_w) == 0
    hydro_m = [p for p in model.iter("plugin")
               if "Hydrodynamics" in p.get("name", "")]
    assert len(hydro_m) == 1
    # 부가질량 0 (dartsim 수치 폭발 회피 — 감쇠만), 감쇠는 음수 조립
    assert hydro_m[0].find("zDotW").text.strip() == "0"
    assert float(hydro_m[0].find("zW").text) < 0
    # 항력 2차항 수술 (2026-08-03, 활주 스텝 실험의 처방): 선형 20% +
    # 제곱 80%, 두 항 모두 실제 저항 R에 앵커 — 목표속도에서 합이
    # 정확히 R (검증점 보존이 수식으로 보장됨)
    u_t = report["goal"]["target_speed_ms"]
    r_t = report["resistance"]["total"]
    xu = float(hydro_m[0].find("xU").text)
    xuu = float(hydro_m[0].find("xUabsU").text)
    assert xu == pytest.approx(-0.2 * r_t / u_t, rel=1e-3)
    assert xuu == pytest.approx(-0.8 * r_t / u_t ** 2, rel=1e-3)
    drag_at_target = -xu * u_t - xuu * u_t ** 2
    assert drag_at_target == pytest.approx(r_t, rel=1e-3)


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
