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

from src.sim_adapters.python_sim import (
    ACCEPT_RADIUS_OVER_L,
    SLOWDOWN_MIN_FRACTION,
    SLOWDOWN_RADIUS_OVER_L,
    THRUSTER_SEP_OVER_B,
)

KXX_OVER_B = 0.35  # 횡동요 회전반경/폭 (통상값)
KYY_OVER_L = 0.25  # 종동요 회전반경/길이 (통상값)

# 부력 collision 분할 (2026-08-03 gz 부양 자세 캠페인):
# gz graded 부력은 단일 상자의 부심을 기울기 무관 고정점처럼 취급 —
# 수선면 복원(메타센터) 부재로 CG가 부심 위인 모든 배가 도립 진자
# 전복 (피치 지수 성장 실측, 시정수 ~3s). 상자를 격자 분할하면
# 기운 쪽 조각이 더 잠겨 부심이 이동 = 복원을 이산 재현 (E3: 4×3
# 분할로 60 s 직립 완치 실측).
GZ_BUOY_NX, GZ_BUOY_NY = 4, 3


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
    # 상한도 필요 (활주 선체 실측, 2026-08-01): 폭이 좁으면 ixx가 작아
    # ixx+iyy < izz가 될 수 있는데, 실물 질량 분포는 항상
    # izz ≤ ixx+iyy (평면 판에서 등호) — 위반 시 Gazebo가 거부.
    izz = min(izz, 0.999 * (ixx + iyy))
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


def _rudder_geometry(report: dict) -> dict:
    """러더 제원 (스펙 4단계) — 3단계 사이징 그대로 + gz 배치.

    스팬은 상자 흘수의 0.9로 상한 (LiftDrag는 수면을 몰라 수면 관통
    부분에도 힘을 계산 — 잠긴 판만 두는 게 정직). 상한 걸리면 코드
    (chord)로 면적 보존 (종횡비 하락은 한계 명기 — cla는 Mandel ΛE=3
    유지)."""
    from src.physics.cargo_hold import RESERVED_DENSITY_KG_M3  # noqa: F401
    from src.sim_adapters.python_sim import SLOWDOWN_MIN_FRACTION
    from src.sim_adapters.rudder import (
        RUDDER_AR_GEOMETRIC,
        RUDDER_K_LAMBDA,
        RUDDER_MAX_RAD,
        RUDDER_STALL_RAD,
        RudderModel,
        lift_slope_mandel,
    )

    d = report["dimensions"]
    p = report["propulsion"]
    w = report["weights"]
    u_t = report["goal"]["target_speed_ms"]
    sep = THRUSTER_SEP_OVER_B * d["beam"]
    rud = RudderModel.for_vessel(
        d["loa"], 0.15 * d["loa"], beam=d["beam"],
        required_moment=float(p["motor"]["thrust_max_n"]) * sep,
        u_design=SLOWDOWN_MIN_FRACTION * u_t)
    draft_box = w["total_mass"] / (1025.0 * d["loa"] * d["beam"])
    span = min((RUDDER_AR_GEOMETRIC * rud.area) ** 0.5, 0.9 * draft_box)
    chord = rud.area / span
    return {
        "area": rud.area, "span": span, "chord": chord,
        "x_pos": -0.48 * d["loa"],           # gz 프레임: 선미 = −x
        "z_center": -d["depth"] / 2.0 + span / 2.0,   # 킬부터 세움
        "cla": lift_slope_mandel(RUDDER_K_LAMBDA * RUDDER_AR_GEOMETRIC),
        "stall_rad": RUDDER_STALL_RAD, "max_rad": RUDDER_MAX_RAD,
    }


def export_sdf(report: dict, mesh_path: str | Path, out_dir: str | Path,
               collision: str = "mesh",
               waypoints: list[tuple[float, float]] | None = None,
               follower: bool = False, rudder: bool = False) -> Path:
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
        visual_xml = geom_xml
        visual_pose_z = 0.0
        collision_z = 0.0  # 상자 중심 = 링크 원점
        cg_z = 0.0   # 상자 중심 기준
        cg_x = 0.0   # 검정 목적: 트림 없는 순수 부력 확인 — CG를 부심 위에
    else:
        # Gazebo graded 부력의 구조적 한계 (2026-07-29 실측 2건):
        # ① 메쉬 collision → 부력 0으로 조용히 무시 (자유낙하 z=−18,123km)
        # ② 등가 부피 상자(폭 축소) → 복원력 파괴로 전복 (GM<0)
        # → 부력체는 외피 상자 L×B×D: 복원력 건전, 정착 흘수는 상자
        #   해석해 T_box = m/(ρLB) 기준 (선체 실흘수보다 얕음 — 상자는
        #   Cb=1이라). 선체 실흘수·실유체력 재현은 부피 부력으로 불가 —
        #   B-3의 계수 기반 플러그인(hydrodynamics.yaml)이 담당할 과제.
        mesh_file = Path(mesh_path).name
        # STL을 내보내기 폴더로 복사 — 상대 uri가 /sim 마운트에서 풀리게
        # (미복사 시 'uri could not be resolved' 실측, 2026-08-01)
        import shutil as _shutil

        dst = Path(out_dir) / mesh_file
        if Path(mesh_path).resolve() != dst.resolve():
            _shutil.copy(mesh_path, dst)
        geom_xml = (f"<box><size>{d['loa']:.4f} {d['beam']:.4f} "
                    f"{d['depth']:.4f}</size></box>")
        # 링크 좌표계 = 부력 상자 중심 (collision 오프셋 0 — 오프셋을 주면
        # 부력 작용점 계산이 틀어져 피치 발산함을 실측, 07-29).
        # 메쉬 시각만 −D/2 내려 킬을 상자 하면에 정렬.
        visual_xml = (f"<mesh><uri>{mesh_file}</uri></mesh>")
        visual_pose_z = -d["depth"] / 2.0
        draft_expected = mass / (1025.0 * d["loa"] * d["beam"])  # 상자 해석해
        init_z = d["depth"] / 2.0 - draft_expected  # 상자 중심 기준
        collision_z = 0.0
        cg_z = w["kg"] - d["depth"] / 2.0  # 킬 기준 KG → 상자 중심 기준
        # 세로 CG 오프셋은 gz 내보내기에서 0: graded 부력의 트림 평형이
        # 정적 예측(1.7°)의 ~16배 피치를 만드는 것을 실측 (07-29 분리 실험:
        # lcg=0이면 1.3mm 정확도로 정착). 트림 정보는 리포트에 보존 —
        # gz 데모는 무트림 부양으로 한정 (한계 문서화)
        cg_x = 0.0

    # 추력기 2발 + 오도메트리 (B-3c): 우리 제어기(LOS+고유값 게인)가
    # gz 토픽으로 조종. B-3b 실측 결론(내장 추종기 구조적 미달)의 해법.
    sep = THRUSTER_SEP_OVER_B * d["beam"]
    prop_x = -0.45 * d["loa"]
    prop_z = -d["depth"] / 2.0 + 0.02  # 상자 중심 프레임 — 킬 근처 (침수)
    thruster_xml = ""
    for side, y in (("left", +sep / 2), ("right", -sep / 2)):
        thruster_xml += f"""
    <link name="{side}_prop">
      <pose>{prop_x:.4f} {y:.4f} {prop_z:.4f} 0 0 0</pose>
      <inertial><mass>0.05</mass>
        <inertia><ixx>1e-5</ixx><iyy>1e-5</iyy><izz>1e-5</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <visual name="{side}_prop_visual">
        <!-- 시각만 회전 — 링크를 돌리면 조인트 축(=추력 방향)이 수직이 됨
             (07-29 실측: 전진 불능·표류의 원인) -->
        <pose>0 0 0 0 1.5708 0</pose>
        <geometry><cylinder><radius>0.04</radius><length>0.02</length></cylinder></geometry>
      </visual>
    </link>
    <joint name="{side}_prop_joint" type="revolute">
      <parent>base_link</parent>
      <child>{side}_prop</child>
      <axis><xyz>1 0 0</xyz></axis>
    </joint>
    <plugin filename="gz-sim-thruster-system" name="gz::sim::systems::Thruster">
      <namespace>generated_hull</namespace>
      <joint_name>{side}_prop_joint</joint_name>
      <thrust_coefficient>0.004</thrust_coefficient>
      <fluid_density>1025</fluid_density>
      <propeller_diameter>0.08</propeller_diameter>
    </plugin>"""
    thruster_xml += """
    <plugin filename="gz-sim-odometry-publisher-system"
            name="gz::sim::systems::OdometryPublisher">
      <odom_publish_frequency>10</odom_publish_frequency>
    </plugin>"""

    # 러더 (스펙 4단계): 링크 + z축 revolute 조인트 + LiftDrag(수중
    # 밀도 1025) + JointPositionController (서보). LiftDrag는 gz가
    # 자기 방식으로 양력을 계산 — 파이썬 러더 모형과 독립 (교차 검증).
    if rudder:
        rg = _rudder_geometry(report)
        thruster_xml += f"""
    <link name="rudder">
      <pose>{rg['x_pos']:.4f} 0 {rg['z_center']:.4f} 0 0 0</pose>
      <!-- 관성 현실화: 깃털 관성(1e-4)에 서보 p=5는 폭주 실측
           (2026-08-03: 각속도 25,000 rad/s — 러더가 프로펠러化,
           반동 토크로 선체 전복). 실물급 질량·관성 + 조인트 감쇠. -->
      <inertial><mass>0.5</mass>
        <inertia><ixx>5e-3</ixx><iyy>5e-3</iyy><izz>5e-3</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <visual name="rudder_visual">
        <geometry><box><size>{rg['chord']:.4f} 0.01 {rg['span']:.4f}</size></box></geometry>
      </visual>
    </link>
    <joint name="rudder_joint" type="revolute">
      <parent>base_link</parent>
      <child>rudder</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit><lower>{-rg['max_rad']:.4f}</lower><upper>{rg['max_rad']:.4f}</upper>
          <effort>20</effort></limit>
        <dynamics><damping>0.5</damping></dynamics>
      </axis>
    </joint>
    <plugin filename="gz-sim-joint-state-publisher-system"
            name="gz::sim::systems::JointStatePublisher">
      <joint_name>rudder_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-lift-drag-system" name="gz::sim::systems::LiftDrag">
      <link_name>rudder</link_name>
      <air_density>1025</air_density>
      <cla>{rg['cla']:.4f}</cla>
      <alpha_stall>{rg['stall_rad']:.4f}</alpha_stall>
      <cla_stall>-0.5</cla_stall>
      <cda>0.05</cda>
      <cda_stall>1.0</cda_stall>
      <area>{rg['area']:.5f}</area>
      <cp>0 0 0</cp>
      <forward>1 0 0</forward>
      <upward>0 1 0</upward>
    </plugin>
    <plugin filename="gz-sim-joint-position-controller-system"
            name="gz::sim::systems::JointPositionController">
      <joint_name>rudder_joint</joint_name>
      <p_gain>2.0</p_gain>
      <d_gain>0.4</d_gain>
      <cmd_max>15</cmd_max>
      <cmd_min>-15</cmd_min>
    </plugin>"""

    # 웨이포인트 추종 (B-3b, 실험용 — 기본 꺼짐): gz 내장 TrajectoryFollower.
    # 구조적 미달 실측 — 정밀 제어는 B-3c(waypoint_controller.py).
    # 켜면 우리 제어기와 동시에 배를 잡아당기므로 병용 금지.
    follower_xml = ""
    if waypoints and follower:
        p = report["propulsion"]
        force = p["motor"]["thrust_max_n"]  # 1발 상당 — 순항 추력 여유권
        torque = (p["motor"]["thrust_max_n"] * 0.8 * report["dimensions"]["beam"]
                  / 2.0)
        wp_xml = "\n".join(f"          <waypoint>{x:.3f} {y:.3f}</waypoint>"
                           for x, y in waypoints)
        follower_xml = f"""
    <plugin filename="gz-sim-trajectory-follower-system"
            name="gz::sim::systems::TrajectoryFollower">
      <link_name>base_link</link_name>
      <loop>false</loop>
      <force>{force:.2f}</force>
      <torque>{torque:.2f}</torque>
      <waypoints>
{wp_xml}
      </waypoints>
    </plugin>"""

    model_sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="generated_hull">{follower_xml}
    <link name="base_link">
      <inertial>
        <pose>{cg_x:.6f} 0 {cg_z:.6f} 0 0 0</pose>
        <mass>{mass:.6f}</mass>
        <inertia>
          <ixx>{ixx:.6f}</ixx><iyy>{iyy:.6f}</iyy><izz>{izz:.6f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
<!--COLLISION-->
      <visual name="hull_visual">
        <pose>0 0 {visual_pose_z:.6f} 0 0 0</pose>
        <geometry>{visual_xml}</geometry>
      </visual>
    </link>{thruster_xml}<!--HYDRO-->
  </model>
</sdf>
"""
    # 부력 collision: mesh 모드는 격자 분할 (전복 완치, 모듈 상단
    # GZ_BUOY_* 참조), box(바지선 검정) 모드는 단일 유지 (원검증 보존)
    if collision == "box":
        collision_xml = f"""
      <collision name="hull_collision">
        <pose>0 0 {collision_z:.6f} 0 0 0</pose>
        <geometry>{geom_xml}</geometry>
      </collision>"""
    else:
        cells = []
        for i in range(GZ_BUOY_NX):
            for j in range(GZ_BUOY_NY):
                cx = -d["loa"] / 2 + (i + 0.5) * d["loa"] / GZ_BUOY_NX
                cy = -d["beam"] / 2 + (j + 0.5) * d["beam"] / GZ_BUOY_NY
                cells.append(
                    f"""
      <collision name="hull_c{i}{j}">
        <pose>{cx:.4f} {cy:.4f} {collision_z:.6f} 0 0 0</pose>
        <geometry><box><size>{d['loa']/GZ_BUOY_NX:.4f} """
                    f"""{d['beam']/GZ_BUOY_NY:.4f} {d['depth']:.4f}</size></box></geometry>
      </collision>""")
        collision_xml = "".join(cells)
    model_sdf = model_sdf.replace("<!--COLLISION-->", collision_xml)

    c = report["coefficients"]
    # 수직면 3축 계수 (B-3a): 정역학 실계산값 기반 실추정 —
    # 부분 계수 차용은 비물리 모멘트로 피치 발산했음 (07-29 실측)
    from src.physics.coefficients import vertical_plane_estimates

    vp = vertical_plane_estimates(
        mass=mass, ixx=ixx, iyy=iyy,
        awp=h["waterplane_area"], ixx_wp=h["waterplane_ixx"],
        gm=h["gm"], disp_vol=h["displacement_volume"], loa=d["loa"],
    )
    # gz Hydrodynamics 플러그인: 감쇠만 적용 (SNAME 부호 음수).
    # 부가질량은 dartsim에서 수치 폭발 실측 (07-29: 위치 1e125 m —
    # 우리 스케일의 부가질량 비율에서 발산) → 0으로 두고 감쇠만.
    # 경로 추종에는 감쇠가 본질 — 부가질량 정합은 후속 과제 (한계 명시).
    hydro_xml = f"""
    <plugin filename="gz-sim-hydrodynamics-system"
            name="gz::sim::systems::Hydrodynamics">
      <link_name>base_link</link_name>
      <xDotU>0</xDotU>
      <yDotV>0</yDotV>
      <zDotW>0</zDotW>
      <kDotP>0</kDotP>
      <mDotQ>0</mDotQ>
      <nDotR>0</nDotR>
      <xU>{-0.2 * report['resistance']['total'] / max(report['goal']['target_speed_ms'], 1e-6):.4f}</xU>
      <xUabsU>{-0.8 * report['resistance']['total'] / max(report['goal']['target_speed_ms'], 1e-6) ** 2:.4f}</xUabsU>
      <yV>{-c['yv']:.4f}</yV>
      <zW>{-vp['z_damping']:.4f}</zW>
      <kP>{-vp['k_damping']:.4f}</kP>
      <mQ>{-vp['m_damping']:.4f}</mQ>
      <nR>{-c['nr']:.4f}</nR>
    </plugin>"""
    # Hydrodynamics는 **모델 플러그인** — 월드 레벨에 두면
    # 'base_link does not exist'로 조용히 죽어 무감쇠 배가 됨 (07-29 실측)
    model_sdf = model_sdf.replace("<!--HYDRO-->", hydro_xml)

    world_sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="ship_water">
    <physics name="default" type="dartsim">
      <max_step_size>0.005</max_step_size>
      <real_time_factor>1</real_time_factor>
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


def export_control_yaml(report: dict, out_dir: str | Path,
                        waypoints: list[tuple[float, float]] | None = None,
                        steering: str = "diff",
                        u_override: float | None = None) -> Path:
    """제어기 파라미터 내보내기 (B-3c) — python_sim과 동일 법칙.

    게인은 design_gains(고유값 판별 사다리)에서 — 컨테이너 제어기가
    이 YAML만 읽으면 우리 시뮬과 같은 제어를 재현.
    """
    from src.sim_adapters.python_sim import design_gains, vessel_from_report

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    vessel = vessel_from_report(report)
    u_d = u_override if u_override is not None \
        else report["goal"]["target_speed_ms"]
    # 게인·lookahead는 설계 속도 기준 유지 (저속 판정 실험에서도
    # 제어기는 동일 — python 1단계 실험 규약과 맞춤)
    kp_psi, kd_psi, lookahead, info = design_gains(
        vessel, report["goal"]["target_speed_ms"])

    # gz 플랜트 보정 (07-30 실측): 파이썬 모델 기준 게인은 gz에서 과공격
    # (모델 불일치 — 부가질량 0·제어 5Hz 지연) → 지그재그 σ 1.5m.
    # 완화 배율 적용 후 σ 0.22m (7배 개선), 완주 유지·가속.
    # 재튜닝 (2026-08-03, 항력 2차항 세계 기준 7런 스윕): 옛 스케일
    # (0.35/0.5/1.5)은 미끄러운 선형 세계용 붕대였음 — 정직해진 세계에선
    # 원설계(1/1/1)가 최적 (σ 1.74→1.31 @1.4 m/s). 잔여 요동의 구조
    # 원인은 게인이 아니라 "차동 주입 자기지속 루프" (별건 등록).
    GZ_KP_SCALE, GZ_KD_SCALE, GZ_LOOKAHEAD_SCALE = 1.0, 1.0, 1.0
    kp_psi *= GZ_KP_SCALE
    kd_psi *= GZ_KD_SCALE
    lookahead *= GZ_LOOKAHEAD_SCALE

    data = {
        "u_desired": u_d,
        "kp_psi": float(kp_psi),
        "kd_psi": float(kd_psi),
        "kp_u": float(8.0 * vessel.m_x / 10.0),
        "lookahead": float(lookahead),
        "accept_radius": float(ACCEPT_RADIUS_OVER_L * vessel.loa),
        "slowdown_radius": float(SLOWDOWN_RADIUS_OVER_L * vessel.loa),
        "slowdown_min_fraction": float(SLOWDOWN_MIN_FRACTION),
        "thrust_max": float(vessel.thrust_max),
        "thruster_separation": float(vessel.thruster_sep),
        "design_info": {k: (bool(v) if isinstance(v, bool) else float(v))
                        if not isinstance(v, str) else v
                        for k, v in info.items()},
        "waypoints": [[float(x), float(y)] for x, y in (waypoints or [])],
        "steering": steering,
    }
    if steering in ("rudder2", "rudder1"):
        rg = _rudder_geometry(report)
        data["rudder"] = {
            "area": float(rg["area"]), "cla": float(rg["cla"]),
            "x_pos": float(rg["x_pos"]), "max_rad": float(rg["max_rad"]),
            "stall_rad": float(rg["stall_rad"]),
            # 부호: gz LiftDrag 방향 규약은 스텝 실험으로 검정 —
            # 뒤집히면 여기만 −1 (컨트롤러가 곱해 씀)
            "sign": 1.0,
            # gz 저속 러더 반전 아티팩트 (2026-08-03 직립 실측:
            # δ+0.15 → u0.77에서 yaw −83° / u1.37에서 +112°) —
            # 문턱 아래는 차동 조타 (구성 A 저속 철학과 일치)
            "min_speed": 1.1,
        }
    path = out / "control.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
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
    parser.add_argument("--course-square", action="store_true",
                        help="사각 코스(변 10L) 웨이포인트 추종 포함 (B-3b)")
    parser.add_argument("--steering", default="diff",
                        choices=["diff", "rudder2", "rudder1"],
                        help="조타: diff(차동) | rudder2(차동+러더) | "
                             "rudder1(단일 상당+러더 — 차동 보조 없음)")
    parser.add_argument("--u-desired", type=float, default=None,
                        help="목표 속도 덮어쓰기 (저속 판정 실험용)")
    args = parser.parse_args(argv)

    with open(args.report) as f:
        report = json.load(f)
    waypoints = None
    if args.course_square:
        s = 10.0 * report["dimensions"]["loa"]
        waypoints = [(s, 0.0), (s, s), (0.0, s), (0.0, 0.0)]
    urdf = export_urdf(report, args.mesh, args.out)
    hydro = export_hydro_yaml(report, args.out)
    sdf = export_sdf(report, args.mesh, args.out, collision=args.collision,
                     waypoints=waypoints,
                     rudder=(args.steering in ("rudder2", "rudder1")))
    if args.course_square:
        export_control_yaml(report, args.out, waypoints=waypoints,
                            steering=args.steering,
                            u_override=args.u_desired)
    if args.collision == "mesh":
        print("⚠ mesh 모드는 실험적 — 부양 검증은 box(바지선 해석해) 모드가 "
              "기준. 실선체 완전 부양은 B-3(6자유도 계수 정합) 과제.")
    print(f"내보내기 완료: {urdf}, {hydro}, {sdf}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
