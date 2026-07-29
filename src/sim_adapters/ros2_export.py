"""ROS2 내보내기 어댑터 (spec §2.4, Phase B-1).

본질은 계수 변환기 — URDF는 기하·관성만 나르므로 (spec 명시),
유체역학(부가질량·감쇠·추력기)은 별도 YAML로 내보낸다.

산출물:
- hull.urdf: base_link (질량, 관성 3축, 메쉬 visual/collision)
- hydrodynamics.yaml: Fossen 계수 세트 + 추력기 배치
  부호 규약: 전부 크기(양수) — Gazebo 플러그인/시뮬 조립 시 부호 적용
  (M4a CoefficientSet 규약 그대로, python_sim.step과 동일)

관성 가정 (명명 상수): Izz는 M4a 분포모델 실계산값, Ixx·Iyy는
회전반경 개략 (kxx≈0.35B 횡동요, kyy≈0.25L 종동요 — 통상값).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

KXX_OVER_B = 0.35  # 횡동요 회전반경/폭 (통상값)
KYY_OVER_L = 0.25  # 종동요 회전반경/길이 (통상값)


def _inertia_triplet(report: dict) -> tuple[float, float, float]:
    """내보내기용 관성 3축 (ixx, iyy, izz) — 삼각 부등식 보장.

    weights.izz(성분 점질량 모델)는 중앙 탑재 하중의 기여를 0으로 보는
    **하계** — iyy(전체 질량 × 0.25L 회전반경)보다 작으면 물리 모순
    (Gazebo가 invalid inertia로 거부, 2026-07-29 실측). 관례상
    kzz ≈ kyy(세장 선체)이므로 izz 하한을 iyy로 둔다.
    """
    d = report["dimensions"]
    w = report["weights"]
    mass = w["total_mass"]
    ixx = mass * (KXX_OVER_B * d["beam"]) ** 2
    iyy = mass * (KYY_OVER_L * d["loa"]) ** 2
    izz = max(w["izz"], iyy)
    return ixx, iyy, izz


def export_urdf(report: dict, mesh_path: str | Path,
                out_dir: str | Path) -> Path:
    """report.json + STL → hull.urdf."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = report["dimensions"]
    w = report["weights"]
    mass = w["total_mass"]
    ixx, iyy, izz = _inertia_triplet(report)

    robot = ET.Element("robot", name="generated_hull")
    link = ET.SubElement(robot, "link", name="base_link")

    inertial = ET.SubElement(link, "inertial")
    # 무게중심: x=LCG (선체 중앙 기준), z=KG (킬 기준)
    ET.SubElement(inertial, "origin",
                  xyz=f"{w['lcg']:.6f} 0 {w['kg']:.6f}", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
    ET.SubElement(inertial, "inertia",
                  ixx=f"{ixx:.6f}", iyy=f"{iyy:.6f}", izz=f"{izz:.6f}",
                  ixy="0", ixz="0", iyz="0")

    mesh_file = Path(mesh_path).name
    for tag in ("visual", "collision"):
        el = ET.SubElement(link, tag)
        geom = ET.SubElement(el, "geometry")
        ET.SubElement(geom, "mesh", filename=f"package://hull/{mesh_file}")

    tree = ET.ElementTree(robot)
    ET.indent(tree)
    path = out / "hull.urdf"
    tree.write(path, encoding="unicode", xml_declaration=True)
    return path


def export_hydro_yaml(report: dict, out_dir: str | Path) -> Path:
    """Fossen 계수 + 추력기 배치 → hydrodynamics.yaml."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    c = report["coefficients"]
    p = report["propulsion"]
    d = report["dimensions"]
    data = {
        "sign_convention": (
            "모든 계수는 크기(양수). Fossen 방정식 조립 시 감쇠·부가질량에 "
            "관례 부호(−)를 적용할 것 — src/sim_adapters/python_sim.step 참조"
        ),
        "reference": {
            "speed_ms": report["goal"]["target_speed_ms"],
            "extrapolation_warning": c["extrapolation_warning"],
            "clamped_terms": list(c.get("clamped_terms", [])),
        },
        "added_mass": {
            "x_dot_u": c["xu_dot"],   # [kg]
            "y_dot_v": c["yv_dot"],   # [kg]
            "n_dot_r": c["nr_dot"],   # [kg·m²]
        },
        "linear_damping": {
            "xu": c["xu"],            # [N/(m/s)] — 저항곡선 미분
            "yv": c["yv"],
            "nv": c["nv"],
            "nr": c["nr"],
        },
        "thrusters": {
            "count": p["count"],
            "separation_m": 0.8 * d["beam"],
            "max_thrust_n": p["motor"]["thrust_max_n"],
            "model": p["motor"]["name"],
        },
    }
    path = out / "hydrodynamics.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return path


def export_sdf(report: dict, mesh_path: str | Path, out_dir: str | Path,
               collision: str = "mesh") -> Path:
    """Gazebo용 model.sdf + 부력 월드 world.sdf (Phase B-2b).

    collision:
    - "mesh": 선체 STL 그대로 (Gazebo 부력의 메쉬 부피 처리는 근사 가능성 —
      결과 해석 시 감안)
    - "box": L×B×D 직육면체 — 정착 흘수 해석해 T=m/(ρ·L·B)가 존재하는
      바지선 검정용 (Gazebo 부력 셋업 자체의 정답지 검증, B-2c 1단계)
    수면은 z=0, 모델은 예측 흘수만큼 잠긴 자세로 초기 배치.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = report["dimensions"]
    w = report["weights"]
    h = report["hydrostatics"]
    mass = w["total_mass"]
    ixx, iyy, izz = _inertia_triplet(report)

    if collision == "box":
        geom_xml = (f"<box><size>{d['loa']:.4f} {d['beam']:.4f} "
                    f"{d['depth']:.4f}</size></box>")
        # 상자 중심 = 형심 중앙 → 초기 z: 상자 하면이 해석해 흘수만큼 잠기게
        draft_expected = mass / (1025.0 * d["loa"] * d["beam"])
        init_z = d["depth"] / 2.0 - draft_expected
        cg_z = 0.0   # 상자 중심 기준
        cg_x = 0.0   # 검정 목적: 트림 없는 순수 부력 확인 — CG를 부심 위에
    else:
        mesh_file = Path(mesh_path).name
        geom_xml = f"<mesh><uri>{mesh_file}</uri></mesh>"
        draft_expected = h["draft"]
        init_z = -draft_expected  # 메쉬 z=0이 킬 → 킬이 흘수만큼 물밑
        cg_z = w["kg"]
        cg_x = w["lcg"]

    model_sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="generated_hull">
    <link name="base_link">
      <inertial>
        <pose>{cg_x:.6f} 0 {cg_z:.6f} 0 0 0</pose>
        <mass>{mass:.6f}</mass>
        <inertia>
          <ixx>{ixx:.6f}</ixx><iyy>{iyy:.6f}</iyy><izz>{izz:.6f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="hull_collision"><geometry>{geom_xml}</geometry></collision>
      <visual name="hull_visual"><geometry>{geom_xml}</geometry></visual>
    </link>
  </model>
</sdf>
"""
    world_sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="ship_water">
    <physics name="default" type="dartsim">
      <max_step_size>0.005</max_step_size>
      <real_time_factor>0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-buoyancy-system" name="gz::sim::systems::Buoyancy">
      <!-- 수상선은 graded 필수: uniform은 잠수함 가정(상시 전부피 부력)이라
           수면 개념이 없어 모델이 무한 상승함 (2026-07-29 실측: z=45,411km) -->
      <graded_buoyancy>
        <default_density>1025</default_density>
        <density_change>
          <above_depth>0</above_depth>
          <density>1</density>
        </density_change>
      </graded_buoyancy>
      <enable>generated_hull</enable>
    </plugin>
    <include>
      <uri>model.sdf</uri>
      <pose>0 0 {init_z:.6f} 0 0 0</pose>
    </include>
  </world>
</sdf>
"""
    (out / "model.sdf").write_text(model_sdf)
    path = out / "world.sdf"
    path.write_text(world_sdf)
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="ROS2 내보내기 (Phase B-1/B-2)")
    parser.add_argument("--report", required=True)
    parser.add_argument("--mesh", required=True, help="hull STL 경로")
    parser.add_argument("--out", default="ros2_ws/export")
    parser.add_argument("--collision", default="mesh", choices=["mesh", "box"],
                        help="SDF collision: mesh(실선체) | box(바지선 검정)")
    args = parser.parse_args(argv)

    with open(args.report) as f:
        report = json.load(f)
    urdf = export_urdf(report, args.mesh, args.out)
    hydro = export_hydro_yaml(report, args.out)
    sdf = export_sdf(report, args.mesh, args.out, collision=args.collision)
    print(f"내보내기 완료: {urdf}, {hydro}, {sdf}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
