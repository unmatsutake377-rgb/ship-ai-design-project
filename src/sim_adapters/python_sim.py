"""Fossen 3자유도 웨이포인트 추종 시뮬레이션 (spec §2.4, M4b).

구성: 차동 추력 2발(러더 없음) + LOS 유도 + 선수각 PD + 속도 P 제어.

운동방정식 (M4a 계수, 크기→부호 조립은 여기서):
  m_x·u̇ = (T_L + T_R) − R(u) + m_y·v·r
  m_y·v̇ = −m_x·u·r − Yv·v
  I_z·ṙ = (T_R − T_L)·d/2 − Nr·r − Nv·v
  ẋ = u·cosψ − v·sinψ,  ẏ = u·sinψ + v·cosψ,  ψ̇ = r

명시된 단순화 (2차 사이클에서 보강):
- 교차 감쇠 Yr·r, 교차 부가질량 Yṙ·Nv̇ 생략
- 전진 오일러 적분 (감쇠 지배 시스템, dt=0.05 s)
- 추력기 후진 한계 = 전진 한계와 대칭 가정
- 저항 R(u)는 사전 샘플 보간 (시뮬 루프에서 Michell 재호출 안 함)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.ai.hull_generator import generate_hull_mesh, solve_exponents
from src.core.types import MainDimensions

THRUSTER_SEP_OVER_B = 0.8   # 추력기 좌우 간격 / 폭
# 수용 반경: 이 거리 안에 들어오면 "도달" 판정. 2.0L로 뒀더니 20 m 코스에서
# 4 m 밖을 지나가며 도달 처리됨 (오너 지적) — 1.0L로 조정.
# 실험 (2026-07-26): 2.0L 121s/최악3.8m, 1.0L 140s/1.9m, 0.5L 148s/0.4m.
ACCEPT_RADIUS_OVER_L = 1.0

# LOS lookahead (#21): 웨이포인트 조준(순수추적)이 아니라 "경로선에서
# 벗어난 거리(cross-track)를 lookahead 거리 앞에서 되찾는 방향"을 조준.
# lookahead 거리는 design_gains가 게인과 함께 고유값 판별로 선정.

# 코너 감속: 웨이포인트 이 거리 안에서 목표 속도를 선형 축소 (하한 비율).
# 스윕 (07-27): 강한 감속(3L/0.4)은 저속 미끄럼 회복 지연으로 역효과(1.304).
# 완만(2L/0.6)이 최적: 경로비 1.153 — 이 코스 기하의 물리 바닥 근처.
SLOWDOWN_RADIUS_OVER_L = 2.0
SLOWDOWN_MIN_FRACTION = 0.6
DT_DEFAULT = 0.05           # [s]

# 선수각 제어 설계 — 고유값 판별 기반 (2026-07-27 개편):
# 발견: 통통한 맨몸 선체(L/B~2)는 직진 방향 불안정 (판별식 C<0, 실선도
# 그래서 스케그·쌍동을 씀). 헤딩 P만으로는 횡미끄럼-선회 결합이 한계
# 사이클(잔물결 1.3m 고착)을 만든다 — dt 무관, 게인 단순 조정 무효 확인.
# 해법: 추종 루프 선형화 행렬의 고유값을 실행 시 검사해, 후보 사다리에서
# 첫 안정 조합(최대 실부 < 여유)을 채택. 후보는 (Kp 모멘트비, lookahead/L,
# Kd/I_z) — 약한 것부터 에스컬레이션.
GAIN_CANDIDATES = [
    (0.7, 6.0, 0.0),   # 순한 기본 (안정 선체용)
    (1.5, 6.0, 3.0),
    (3.0, 4.0, 3.0),
    (6.0, 4.0, 7.5),   # 불안정 선체 제압용 (현 데모 선체가 여기 안착)
]
STABILITY_MARGIN = 0.02     # 요구: 최대 고유값 실부 < -이 값
KP_SPEED = 8.0              # 속도 오차 → 공통 추력 (m_x/10 스케일 곱)


def _tracking_matrix(vessel: VesselModel, u0: float, kp: float, kd: float,
                     lookahead: float) -> np.ndarray:
    """직선 추종 폐루프 선형화: 상태 [횡이탈 e, ψ, v, r]."""
    return np.array([
        [0.0, u0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, -vessel.yv / vessel.m_y,
         -(vessel.m_x / vessel.m_y) * u0],
        [-kp / (lookahead * vessel.i_z), -kp / vessel.i_z,
         -vessel.nv / vessel.i_z, -(vessel.nr + kd) / vessel.i_z],
    ])


def design_gains(vessel: VesselModel, u_desired: float
                 ) -> tuple[float, float, float, dict]:
    """고유값 판별로 (kp_psi, kd_psi, lookahead) 선정. 실패 시 마지막 후보 + 경고."""
    moment_max = vessel.thrust_max * vessel.thruster_sep
    info = {}
    for kp_frac, look_l, kd_over_iz in GAIN_CANDIDATES:
        kp = kp_frac * moment_max / (math.pi / 2.0)
        kd = kd_over_iz * vessel.i_z
        lookahead = look_l * vessel.loa
        eig = np.linalg.eigvals(_tracking_matrix(vessel, u_desired, kp, kd,
                                                 lookahead))
        worst = float(max(e.real for e in eig))
        info = {"kp_frac": kp_frac, "lookahead_over_l": look_l,
                "kd_over_iz": kd_over_iz, "max_eig_real": worst,
                "stable": worst < -STABILITY_MARGIN}
        if info["stable"]:
            return kp, kd, lookahead, info
    return kp, kd, lookahead, info  # 전부 불안정 — 마지막 후보 + stable=False

RESISTANCE_SAMPLES = 8      # 저항곡선 사전 샘플 수
RESISTANCE_SPEED_FACTOR = 1.6  # 샘플 상한 / 목표 속도


def ssa(angle: float) -> float:
    """최단 부호 각 (smallest signed angle) [-π, π)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class VesselModel:
    """시뮬레이션용 선박 모델 — 파이프라인 리포트에서 재구성."""

    loa: float
    m_x: float             # m + Xu̇ [kg]
    m_y: float             # m + Yv̇ [kg]
    i_z: float             # Izz + Nṙ [kg·m²]
    yv: float              # 횡 감쇠 크기 [N/(m/s)]
    nv: float              # [N·m/(m/s)]
    nr: float              # [N·m/(rad/s)]
    thrust_max: float      # 추력기 1발 한계 [N]
    thruster_sep: float    # 좌우 간격 [m]
    speeds: tuple          # 저항곡선 샘플 속도
    resistances: tuple     # 저항곡선 샘플 값

    def resistance(self, u: float) -> float:
        """운동 방향에 저항하는 힘 (보간, 후진은 전진 곡선 대칭 가정)."""
        magnitude = float(np.interp(abs(u), self.speeds, self.resistances))
        return math.copysign(magnitude, u)


def vessel_from_report(report: dict) -> VesselModel:
    """report.json에서 선박 모델 재구성 (메쉬·저항곡선 재계산 — 결정적)."""
    from src.physics.resistance import total_resistance

    d = report["dimensions"]
    dims = MainDimensions(loa=d["loa"], beam=d["beam"], depth=d["depth"],
                          draft_design=d["draft_design"], cb=d["cb"])
    c = report["coefficients"]
    w = report["weights"]
    h = report["hydrostatics"]
    p = report["propulsion"]

    mesh = generate_hull_mesh(dims)
    n_exp, m_exp = solve_exponents(dims.cb)
    u_target = report["goal"]["target_speed_ms"]
    speeds = np.linspace(0.05, RESISTANCE_SPEED_FACTOR * u_target,
                         RESISTANCE_SAMPLES)
    resistances = [total_resistance(mesh, dims, n_exp, m_exp,
                                    draft=h["draft"], speed=s).total
                   for s in speeds]
    # 정지점 포함 (R(0)=0)
    speeds = np.concatenate([[0.0], speeds])
    resistances = [0.0] + resistances

    return VesselModel(
        loa=dims.loa,
        m_x=w["total_mass"] + c["xu_dot"],
        m_y=w["total_mass"] + c["yv_dot"],
        i_z=w["izz"] + c["nr_dot"],
        yv=c["yv"], nv=c["nv"], nr=c["nr"],
        thrust_max=float(p["motor"]["thrust_max_n"]),
        thruster_sep=THRUSTER_SEP_OVER_B * dims.beam,
        speeds=tuple(speeds), resistances=tuple(resistances),
    )


def step(vessel: VesselModel, state: np.ndarray, t_l: float, t_r: float,
         dt: float) -> np.ndarray:
    """상태 [x, y, ψ, u, v, r] 한 스텝 전진 (오일러)."""
    x, y, psi, u, v, r = state
    u_dot = ((t_l + t_r) - vessel.resistance(u)
             + vessel.m_y * v * r) / vessel.m_x
    v_dot = (-vessel.m_x * u * r - vessel.yv * v) / vessel.m_y
    r_dot = ((t_r - t_l) * vessel.thruster_sep / 2.0
             - vessel.nr * r - vessel.nv * v) / vessel.i_z
    return np.array([
        x + dt * (u * math.cos(psi) - v * math.sin(psi)),
        y + dt * (u * math.sin(psi) + v * math.cos(psi)),
        psi + dt * r,
        u + dt * u_dot,
        v + dt * v_dot,
        r + dt * r_dot,
    ])


@dataclass
class SimResult:
    time: list = field(default_factory=list)
    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    psi: list = field(default_factory=list)
    u: list = field(default_factory=list)
    thrust_l: list = field(default_factory=list)
    thrust_r: list = field(default_factory=list)
    waypoints_reached: int = 0
    success: bool = False
    duration_s: float = 0.0
    control_design: dict = field(default_factory=dict)


def simulate_waypoints(vessel: VesselModel, waypoints: list[tuple[float, float]],
                       u_desired: float, dt: float = DT_DEFAULT,
                       t_max: float = 600.0) -> SimResult:
    """LOS 유도 + PD 선수각 + P 속도 제어로 웨이포인트 순회."""
    accept = ACCEPT_RADIUS_OVER_L * vessel.loa
    # 고유값 판별 게인 선정 (모듈 상단 설계 노트 참조)
    kp_psi, kd_psi, lookahead, design_info = design_gains(vessel, u_desired)
    kp_u = KP_SPEED * vessel.m_x / 10.0       # [N/(m/s)]

    state = np.zeros(6)
    result = SimResult()
    result.control_design = design_info
    wp_index = 0
    prev_wp = (0.0, 0.0)  # 경로선 시작점 (출발 위치)
    steps = int(t_max / dt)

    for k in range(steps):
        t = k * dt
        wx, wy = waypoints[wp_index]
        x, y, psi, u, v, r = state

        # 도달 판정 2중 (2026-08-02 활주 검증이 잡은 잠복 버그):
        # ① 수용 반경 안 ② 종점 통과 — 빠른 배가 반경을 스치면 LOS가
        # 경로선을 무한 연장해 직진 폭주 (Gazebo 실측 1,265m 이탈).
        px, py = prev_wp
        alpha = math.atan2(wy - py, wx - px)
        seg_len = math.hypot(wx - px, wy - py)
        s_along = (x - px) * math.cos(alpha) + (y - py) * math.sin(alpha)
        if math.hypot(wx - x, wy - y) < accept or s_along > seg_len:
            prev_wp = (wx, wy)
            wp_index += 1
            result.waypoints_reached = wp_index
            if wp_index == len(waypoints):
                result.success = True
                result.duration_s = t
                break
            wx, wy = waypoints[wp_index]

        # lookahead LOS: 경로선(이전 WP→현재 WP) 기준 이탈 거리를
        # lookahead 앞 지점에서 되찾는 방향각
        px, py = prev_wp
        alpha = math.atan2(wy - py, wx - px)          # 경로선 방향
        e_ct = (-(x - px) * math.sin(alpha)
                + (y - py) * math.cos(alpha))          # cross-track (+좌측)
        psi_d = alpha + math.atan2(-e_ct, lookahead)

        # 코너 감속: 웨이포인트 접근 시 목표 속도 축소 (관성 오버슛 억제)
        dist_wp = math.hypot(wx - x, wy - y)
        slow_r = max(SLOWDOWN_RADIUS_OVER_L * vessel.loa, 1e-9)
        u_cmd = u_desired * float(np.clip(dist_wp / slow_r,
                                          SLOWDOWN_MIN_FRACTION, 1.0))

        # PD + P — 배분은 선회 우선: 포화 시에도 명령 모멘트 보존
        moment_cmd = kp_psi * ssa(psi_d - psi) - kd_psi * r
        diff = moment_cmd / vessel.thruster_sep  # (T_R − T_L)/2
        diff = float(np.clip(diff, -vessel.thrust_max, vessel.thrust_max))
        headroom = vessel.thrust_max - abs(diff)
        common = float(np.clip(kp_u * (u_cmd - u), -headroom, headroom))
        t_l = common - diff
        t_r = common + diff

        state = step(vessel, state, t_l, t_r, dt)
        result.time.append(t)
        result.x.append(float(state[0]))
        result.y.append(float(state[1]))
        result.psi.append(float(state[2]))
        result.u.append(float(state[3]))
        result.thrust_l.append(t_l)
        result.thrust_r.append(t_r)
    else:
        result.duration_s = t_max
    return result


def default_square_course(loa: float) -> list[tuple[float, float]]:
    """기본 사각 코스 (변 10·L)."""
    s = 10.0 * loa
    return [(s, 0.0), (s, s), (0.0, s), (0.0, 0.0)]


def plot_trajectory(result: SimResult, waypoints: list[tuple[float, float]],
                    path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
    ax.plot(result.x, result.y, lw=1.2, color="#2563eb", label="궤적")
    wx, wy = zip(*waypoints)
    ax.plot(wx, wy, "o--", color="#d97706", alpha=0.7, label="웨이포인트")
    ax.plot(0, 0, "s", color="#059669", label="시작")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    status = "완주" if result.success else f"{result.waypoints_reached}개 도달"
    ax.set_title(f"웨이포인트 추종 — {status}, {result.duration_s:.0f} s")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="웨이포인트 추종 시뮬 (M4b)")
    parser.add_argument("--report", required=True, help="report.json 경로")
    parser.add_argument("--out", default="outputs", help="출력 디렉토리")
    args = parser.parse_args(argv)

    with open(args.report) as f:
        report = json.load(f)
    vessel = vessel_from_report(report)
    waypoints = default_square_course(vessel.loa)
    u_d = report["goal"]["target_speed_ms"]

    result = simulate_waypoints(vessel, waypoints, u_desired=u_d)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_trajectory(result, waypoints, out / "trajectory.png")
    summary = {
        "success": result.success,
        "waypoints_reached": result.waypoints_reached,
        "duration_s": result.duration_s,
        "mean_speed_ms": float(np.mean(result.u)) if result.u else 0.0,
    }
    with open(out / "sim_result.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"웨이포인트 {result.waypoints_reached}/{len(waypoints)} 도달 — "
          f"{'완주' if result.success else '미완'} ({result.duration_s:.0f} s)")
    print(f"궤적: {out / 'trajectory.png'}")
    return 0 if result.success else 2


if __name__ == "__main__":
    import sys

    sys.exit(main())
