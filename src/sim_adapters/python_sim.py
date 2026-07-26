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
ACCEPT_RADIUS_OVER_L = 2.0  # 웨이포인트 수용 반경 / 길이
DT_DEFAULT = 0.05           # [s]

# 제어 게인 (무차원 튜닝 상수 — 테스트는 게인이 아니라 거동을 검증)
KP_HEADING = 4.0   # 선수각 오차 → 차동 추력 (I_z/s² 스케일 곱)
KD_HEADING = 3.0
KP_SPEED = 8.0     # 속도 오차 → 공통 추력 (m_x/s 스케일 곱)

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


def simulate_waypoints(vessel: VesselModel, waypoints: list[tuple[float, float]],
                       u_desired: float, dt: float = DT_DEFAULT,
                       t_max: float = 600.0) -> SimResult:
    """LOS 유도 + PD 선수각 + P 속도 제어로 웨이포인트 순회."""
    accept = ACCEPT_RADIUS_OVER_L * vessel.loa
    kp_psi = KP_HEADING * vessel.i_z          # [N·m/rad] 스케일
    kd_psi = KD_HEADING * vessel.i_z          # [N·m/(rad/s)]
    kp_u = KP_SPEED * vessel.m_x / 10.0       # [N/(m/s)]

    state = np.zeros(6)
    result = SimResult()
    wp_index = 0
    steps = int(t_max / dt)

    for k in range(steps):
        t = k * dt
        wx, wy = waypoints[wp_index]
        x, y, psi, u, v, r = state

        if math.hypot(wx - x, wy - y) < accept:
            wp_index += 1
            result.waypoints_reached = wp_index
            if wp_index == len(waypoints):
                result.success = True
                result.duration_s = t
                break
            wx, wy = waypoints[wp_index]

        # LOS + PD + P
        psi_d = math.atan2(wy - y, wx - x)
        moment_cmd = kp_psi * ssa(psi_d - psi) - kd_psi * r
        diff = moment_cmd / vessel.thruster_sep  # (T_R − T_L)/2
        common = kp_u * (u_desired - u)
        t_l = float(np.clip(common - diff, -vessel.thrust_max,
                            vessel.thrust_max))
        t_r = float(np.clip(common + diff, -vessel.thrust_max,
                            vessel.thrust_max))

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
