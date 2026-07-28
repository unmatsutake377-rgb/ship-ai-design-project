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


def export_urdf(report: dict, mesh_path: str | Path,
                out_dir: str | Path) -> Path:
    """report.json + STL → hull.urdf."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = report["dimensions"]
    w = report["weights"]
    mass = w["total_mass"]
    ixx = mass * (KXX_OVER_B * d["beam"]) ** 2
    iyy = mass * (KYY_OVER_L * d["loa"]) ** 2
    izz = w["izz"]

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


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="ROS2 내보내기 (Phase B-1)")
    parser.add_argument("--report", required=True)
    parser.add_argument("--mesh", required=True, help="hull STL 경로")
    parser.add_argument("--out", default="ros2_ws/export")
    args = parser.parse_args(argv)

    with open(args.report) as f:
        report = json.load(f)
    urdf = export_urdf(report, args.mesh, args.out)
    hydro = export_hydro_yaml(report, args.out)
    print(f"내보내기 완료: {urdf}, {hydro}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
