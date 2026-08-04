"""ELO 대결용 비교 매체 생성 (#34 오너 제안, 2026-08-02).

배경: 정지 산점도+제원표로는 후보 차이가 체감이 안 됨 (오너: "3D
형상으로 그리고 물에 나아가는 짧은 영상을 비교해서 보여주고 거기서
고르는 게 좋아 보인다"). HITL 인터페이스는 판단 재료가 전부다.

산출 2종 (둘 다 좌우 나란히, 동일 축척·동일 코스 — 공정 비교):
- 회전 3D GIF: 두 선체 형상을 같은 각도로 돌려가며
- 주행 GIF: 같은 웨이포인트 코스를 각자의 물리(중량·저항·게인)로
  달리는 top-view 애니메이션 — 속도·항적 차이가 눈에 보임
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib import animation

from src.ai.hull_generator import generate_hull_mesh, solve_exponents
from src.core.types import GoalSpec, MainDimensions
from src.physics.coefficients import estimate_coefficients
from src.sim_adapters.python_sim import (
    SimResult,
    simulate_waypoints,
    vessel_from_report,
)


def report_for_dims(dims: MainDimensions, goal: GoalSpec) -> dict:
    """파레토 후보(치수)를 vessel_from_report가 먹는 리포트로 평가.

    run_pipeline의 Wigley 경로 축약 — 나선 수렴 + 계수까지 동일 물리."""
    from src.pipeline import design_spiral

    mesh = generate_hull_mesh(dims)
    weights, hydro, resist, motors, batt_kg, _ = design_spiral(
        mesh, dims, goal)
    n_exp, m_exp = solve_exponents(dims.cb)
    coeffs = estimate_coefficients(
        dims=dims, draft=hydro.draft, mass=weights.total_mass,
        lcg=weights.lcg, speed=goal.target_speed_ms,
        mesh=mesh, n_exp=n_exp, m_exp=m_exp)
    return {
        "goal": dataclasses.asdict(goal),
        "dimensions": dataclasses.asdict(dims),
        "weights": dataclasses.asdict(weights),
        "hydrostatics": dataclasses.asdict(hydro),
        "coefficients": dataclasses.asdict(coeffs),
        "propulsion": {"motor": {"thrust_max_n":
                                 float(motors.motor["thrust_max_n"])}},
    }


def _mesh_for(row) -> tuple[MainDimensions, trimesh.Trimesh]:
    """파레토 행 → 치수·메쉬 — optimize와 동일 변환 규약 사용.

    (손수 조립하면 depth 규약이 어긋나 평형 흘수 > 설계 흘수 →
    Wigley 수식이 NaN — 실측 후 dims_from_vector로 통일)"""
    from src.optimize import dims_from_vector

    dims = dims_from_vector(np.array([row.loa, row.lb, row.bt, row.cb]))
    return dims, generate_hull_mesh(dims)


def rotating_gif(meshes: list[trimesh.Trimesh], labels: list[str],
                 path: str | Path, n_frames: int = 36, fps: int = 12) -> Path:
    """두 선체 3D 회전 비교 GIF — 동일 축척·동일 시점."""
    lim = max(float(np.abs(m.bounds).max()) for m in meshes)
    fig = plt.figure(figsize=(10, 5), dpi=90)
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d")
            for i in range(len(meshes))]
    surfs = []
    for ax, m, label in zip(axes, meshes, labels):
        v, f = m.vertices, m.faces
        surfs.append(ax.plot_trisurf(v[:, 0], v[:, 1], f, v[:, 2],
                                     color="#0f766e", edgecolor="none",
                                     alpha=0.95, shade=True))
        ax.set_title(label, fontsize=11)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)   # 세 축 동일 — 왜곡 방지 (07-27 교훈)
        ax.set_axis_off()

    def update(k):
        for ax in axes:
            ax.view_init(elev=18, azim=k * 360 / n_frames)
        return surfs

    anim = animation.FuncAnimation(fig, update, frames=n_frames)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return Path(path)


def race_gif(results: list[SimResult], dims_list: list[MainDimensions],
             labels: list[str], waypoints: list[tuple[float, float]],
             path: str | Path, fps: int = 15,
             vessels: list | None = None,
             duration_s: float = 5.0) -> Path:
    """같은 코스 주행 비교 GIF — top view, 항적 + 선체 + 시계 + 저항.

    vessels 주입 시 실시간 저항력(속도→저항곡선 보간)을 배 뒤 화살표와
    숫자로 표시 (오너 제안 2026-08-02: "실제 물의 저항이 보이게").
    duration_s: GIF 재생 길이 목표 — stride를 자동 산정."""
    t_end = max(r.time[-1] for r in results)
    dt_sim = results[0].time[1] - results[0].time[0]
    total_steps = int(t_end / dt_sim)
    frames = max(2, int(duration_s * fps))
    stride = max(1, total_steps // frames)
    frames = total_steps // stride
    wx, wy = zip(*waypoints)
    lim_pad = max(max(map(abs, wx)), max(map(abs, wy))) * 0.25 + 2

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), dpi=90)
    for ax, label in zip(axes, labels):
        ax.plot(wx, wy, "o--", color="#94a3b8", ms=8, lw=1)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11)
        ax.set_xlim(min(wx) - lim_pad, max(wx) + lim_pad)
        ax.set_ylim(min(wy) - lim_pad, max(wy) + lim_pad)
        ax.grid(alpha=0.2)
    trails = [ax.plot([], [], "-", color="#0f766e", lw=1.4)[0]
              for ax in axes]
    hulls = [ax.fill([], [], color="#dc2626", alpha=0.9)[0] for ax in axes]
    clocks = [ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                      fontsize=10) for ax in axes]
    # 저항 화살표 (배 뒤로 끌어당기는 물의 손) + 크기 표시
    r_max = 1.0
    if vessels is not None:
        r_max = max(max(v.resistances) for v in vessels) or 1.0
    arrows = [ax.annotate("", xy=(0, 0), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="-|>", color="#2563eb",
                                          lw=2.2)) for ax in axes]
    r_texts = [ax.text(0.02, 0.88, "", transform=ax.transAxes, va="top",
                       fontsize=10, color="#2563eb") for ax in axes]

    def hull_poly(dims, x, y, psi):
        L, B = dims.loa, dims.beam
        pts = np.array([[L / 2, 0], [L / 4, B / 2], [-L / 2, B / 2],
                        [-L / 2, -B / 2], [L / 4, -B / 2]])
        c, s = np.cos(psi), np.sin(psi)
        rot = pts @ np.array([[c, s], [-s, c]])
        return rot + [x, y]

    arrow_scale = lim_pad * 2.0 / r_max   # 최대 저항 = 화살표 기준 길이

    def update(k):
        for j, (r, dims, trail, hull, clock) in enumerate(
                zip(results, dims_list, trails, hulls, clocks)):
            i = min(k * stride, len(r.time) - 1)
            trail.set_data(r.x[: i + 1], r.y[: i + 1])
            hull.set_xy(hull_poly(dims, r.x[i], r.y[i], r.psi[i]))
            done = " ✓완주" if (r.success and i == len(r.time) - 1) else ""
            clock.set_text(f"t={r.time[i]:.0f}s  u={r.u[i]:.2f} m/s{done}")
            if vessels is not None:
                res_n = float(np.interp(r.u[i], vessels[j].speeds,
                                        vessels[j].resistances))
                length = res_n * arrow_scale
                hx, hy = np.cos(r.psi[i]), np.sin(r.psi[i])
                tail = (r.x[i] - hx * (dims.loa / 2 + length),
                        r.y[i] - hy * (dims.loa / 2 + length))
                head = (r.x[i] - hx * dims.loa / 2,
                        r.y[i] - hy * dims.loa / 2)
                arrows[j].xy = tail          # 화살촉이 뒤로 (끌어당김)
                arrows[j].set_position(head)
                r_texts[j].set_text(f"물의 저항 ← {res_n:.1f} N")
        return trails + hulls

    # 끝 프레임 1.5초 정지 (오너 피드백: 마지막 완주 표시가 안 보였음)
    anim = animation.FuncAnimation(
        fig, lambda k: update(min(k, frames - 1)),
        frames=frames + int(1.5 * fps))
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return Path(path)


def turning_gif(vessels: list, dims_list: list[MainDimensions],
                labels: list[str], path: str | Path,
                fps: int = 15, duration_s: float = 5.0,
                u_entry: float = 1.2) -> Path:
    """선회 시연 비교 GIF — 민첩성의 체감판 (오너: "영상이 중요해").

    **목표 속도로 순항 중** 전타 고정 (좌 최대 / 우 0) — 각자의
    물리로 원을 그림. 지표(agility_metrics)와 같은 방정식이라
    영상 = 지표의 눈 버전.

    진입 속도는 u_entry로 통일 (2026-08-04 오너 발굴 수리): 옛
    "전추력 20초 가속"은 각자의 종단속도로 진입시켜 — 저저항 배가
    외삽 영역 종단 10.3 m/s(목표의 8.6배, 비물리)로 122 m를 날아가는
    불공정·비물리 시연이었음. 실물 전타 시연 규약 = 순항 중 타 꺾기."""
    from src.sim_adapters.python_sim import step

    trajs = []
    for v in vessels:
        state = np.zeros(6)
        state[3] = u_entry                   # 순항 속도로 진입 (통일)
        xs, ys = [], []
        for _ in range(3000):
            state = step(v, state, v.thrust_max, 0.0, 0.05)
            xs.append(state[0])
            ys.append(state[1])
        trajs.append((np.array(xs), np.array(ys), state))

    lim = max(max(np.abs(t[0]).max(), np.abs(t[1]).max())
              for t in trajs) * 1.15 + 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), dpi=90)
    steps_total = len(trajs[0][0])
    frames = max(2, int(duration_s * fps))
    stride = max(1, steps_total // frames)

    trails, boats = [], []
    for ax, (xs, ys, st), dims, label in zip(axes, trajs, dims_list, labels):
        d_sim = 0.5 * ((xs[-1500:].max() - xs[-1500:].min())
                       + (ys[-1500:].max() - ys[-1500:].min()))
        ax.set_title(f"{label}\n선회지름 ≈ {d_sim:.1f} m "
                     f"= {d_sim / dims.loa:.1f}×배길이", fontsize=11)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        trails.append(ax.plot([], [], "-", color="#0f766e", lw=1.4)[0])
        boats.append(ax.plot([], [], "o", color="#dc2626", ms=9)[0])

    def update(k):
        i = min(k * stride, steps_total - 1)
        for (xs, ys, _), trail, boat in zip(trajs, trails, boats):
            trail.set_data(xs[: i + 1], ys[: i + 1])
            boat.set_data([xs[i]], [ys[i]])
        return trails + boats

    anim = animation.FuncAnimation(
        fig, lambda k: update(min(k, frames - 1)),
        frames=frames + int(1.5 * fps))
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return Path(path)


def make_duel_media(row_a, row_b, labels: tuple[str, str],
                    goal: GoalSpec, out_dir: str | Path,
                    course_l: float = 6.0) -> tuple[Path, Path]:
    """파레토 행 2개 → (회전 3D GIF, 주행 비교 GIF)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = [_mesh_for(r) for r in (row_a, row_b)]
    dims_list = [d for d, _ in pairs]
    meshes = [m for _, m in pairs]
    # 공통 절대 코스 (두 배 평균 길이 기준) — 같은 트랙이어야 공정
    l_ref = 0.5 * (dims_list[0].loa + dims_list[1].loa)
    wps = [(course_l * l_ref, 0.0), (course_l * l_ref, course_l * l_ref)]
    results, vessels = [], []
    for dims in dims_list:
        report = report_for_dims(dims, goal)
        vessel = vessel_from_report(report)
        vessels.append(vessel)
        results.append(simulate_waypoints(vessel, wps,
                                          u_desired=goal.target_speed_ms))
    g1 = rotating_gif(meshes, list(labels), out / "duel_shape.gif")
    # 12 s 재생 (오너 4R: "5초 압축은 느린 배의 여정이 안 보인다")
    g2 = race_gif(results, dims_list, list(labels), wps,
                  out / "duel_race.gif", vessels=vessels, duration_s=12.0)
    g3 = turning_gif(vessels, dims_list, list(labels),
                     out / "duel_turning.gif",
                     u_entry=goal.target_speed_ms)
    return g1, g2, g3
