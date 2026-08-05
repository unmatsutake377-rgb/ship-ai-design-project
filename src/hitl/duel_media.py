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

    from src.ai.hull_generator import cm_for_purpose

    from src.ai.hull_generator import lcb_for_purpose

    hull_cm = cm_for_purpose(goal.purpose)
    hull_lcb = lcb_for_purpose(goal.purpose)
    mesh = generate_hull_mesh(dims, cm=hull_cm, lcb_frac=hull_lcb)
    weights, hydro, resist, motors, batt_kg, _ = design_spiral(
        mesh, dims, goal, cm=hull_cm, lcb_frac=hull_lcb)
    n_exp, m_exp = solve_exponents(dims.cb, hull_cm)
    coeffs = estimate_coefficients(
        dims=dims, draft=hydro.draft, mass=weights.total_mass,
        lcg=weights.lcg, speed=goal.target_speed_ms,
        mesh=mesh, n_exp=n_exp, m_exp=m_exp)
    return {
        "hull_cm": hull_cm,
        "hull_lcb_frac": hull_lcb,
        "goal": dataclasses.asdict(goal),
        "dimensions": dataclasses.asdict(dims),
        "weights": dataclasses.asdict(weights),
        "hydrostatics": dataclasses.asdict(hydro),
        "coefficients": dataclasses.asdict(coeffs),
        "propulsion": {"motor": {"thrust_max_n":
                                 float(motors.motor["thrust_max_n"])}},
    }


def _mesh_for(row, purpose: str = "survey"
              ) -> tuple[MainDimensions, trimesh.Trimesh]:
    """파레토 행 → 치수·메쉬 — optimize와 동일 변환 규약 사용.

    (손수 조립하면 depth 규약이 어긋나 평형 흘수 > 설계 흘수 →
    Wigley 수식이 NaN — 실측 후 dims_from_vector로 통일)
    cm 배선 (2026-08-05 오너 발굴): 물리·파레토는 신세계 Cm인데
    회전 GIF 메쉬만 구세계 기본으로 생성되던 누락 — 용도 프리셋 적용."""
    from src.ai.hull_generator import cm_for_purpose
    from src.optimize import dims_from_vector

    dims = dims_from_vector(np.array([row.loa, row.lb, row.bt, row.cb]))
    from src.ai.hull_generator import lcb_for_purpose

    return dims, generate_hull_mesh(dims, cm=cm_for_purpose(purpose),
                                    lcb_frac=lcb_for_purpose(purpose))


def rotating_gif(meshes: list[trimesh.Trimesh], labels: list[str],
                 path: str | Path, n_frames: int = 48, fps: int = 12) -> Path:
    """두 선체 3D 회전 비교 GIF — 동일 축척·동일 시점.

    선저 가시화 2종 (2026-08-05 오너: "U자 굴곡이 안 보인다"):
    ① 스테이션 라인 — 조선 도면처럼 단면 곡선을 몸통 위에 그려
      굴곡을 선으로 노출 (단색 곡면은 곡률 정보를 숨김)
    ② 시점 순환 — 위(+25°)→옆(0°)→아래(−30°)를 오가며 회전:
      배 밑을 올려다보는 구간이 생김."""
    lim = max(float(np.abs(m.bounds).max()) for m in meshes)
    fig = plt.figure(figsize=(10, 5), dpi=90)
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d")
            for i in range(len(meshes))]
    surfs = []
    for ax, m, label in zip(axes, meshes, labels):
        v, f = m.vertices, m.faces
        surfs.append(ax.plot_trisurf(v[:, 0], v[:, 1], f, v[:, 2],
                                     color="#0f766e", edgecolor="none",
                                     alpha=0.55, shade=True))
        # 스테이션 라인: 길이 방향 9곳의 단면 곡선
        (xmin, _, _), (xmax, _, _) = m.bounds
        for xf in np.linspace(0.08, 0.92, 9):
            x = xmin + xf * (xmax - xmin)
            sec = m.section(plane_origin=[float(x), 0, 0],
                            plane_normal=[1, 0, 0])
            if sec is None or not len(sec.entities):
                continue
            for e in sec.entities:
                pts = np.asarray(e.discrete(sec.vertices))
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        color="#134e4a", lw=1.1, alpha=0.95)
        # 용골선 (중심선 바닥 윤곽) — 세로 굴곡
        xs_k = np.linspace(xmin + 0.02, xmax - 0.02, 40)
        keel = []
        for x in xs_k:
            sec = m.section(plane_origin=[float(x), 0, 0],
                            plane_normal=[1, 0, 0])
            if sec is None or not len(sec.entities):
                continue
            pts = np.vstack([e.discrete(sec.vertices)
                             for e in sec.entities])
            j = np.argmin(pts[:, 2])
            keel.append((x, pts[j, 1], pts[j, 2]))
        if keel:
            kk = np.array(keel)
            ax.plot(kk[:, 0], kk[:, 1], kk[:, 2], color="#dc2626",
                    lw=1.6, alpha=0.9)
        ax.set_title(label, fontsize=11)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)   # 세 축 동일 — 왜곡 방지 (07-27 교훈)
        ax.set_axis_off()

    def update(k):
        # 시점 순환: 사인파로 +25° ~ −30° (아래에서 보는 구간 포함)
        elev = -2.5 + 27.5 * np.cos(2 * np.pi * k / n_frames)
        for ax in axes:
            ax.view_init(elev=float(elev), azim=k * 360 / n_frames)
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
    # 화면 범위 = 코스 ∪ 실제 궤적 전체 (오너 2026-08-04: "화면
    # 크기를 늘려서 다 보이게" — 이탈 궤적이 잘리던 문제)
    all_x = list(wx) + [x for r in results for x in r.x]
    all_y = list(wy) + [y for r in results for y in r.y]
    lim_pad = (max(max(all_x) - min(all_x),
                   max(all_y) - min(all_y))) * 0.08 + 2

    speed_factor = t_end / max(duration_s, 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=90)
    fig.suptitle(f"{speed_factor:.0f}배속 재생 — 시뮬 {t_end:.0f}초를 "
                 f"{duration_s:.0f}초에 (시계는 시뮬 시간)", fontsize=11)
    for ax, label in zip(axes, labels):
        ax.plot(wx, wy, "o--", color="#94a3b8", ms=8, lw=1)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11)
        ax.set_xlim(min(all_x) - lim_pad, max(all_x) + lim_pad)
        ax.set_ylim(min(all_y) - lim_pad, max(all_y) + lim_pad)
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


def section_overlay_png(meshes, dims_list, labels, path: str | Path) -> Path:
    """중앙 단면 겹침 비교 — 선저·풍만도 차이의 정면 뷰 (2026-08-05
    오너 발굴 후 정식 편입: 회전 3D는 밑부분 차이를 숨기는 매체)."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    colors = ["#0f766e", "#dc2626"]
    for mesh, dims, label, color in zip(meshes, dims_list, labels, colors):
        zs = np.linspace(0.001, dims.depth - 0.001, 50)
        ys = []
        for z in zs:
            sec = mesh.section(plane_origin=[0, 0, z],
                               plane_normal=[0, 0, 1])
            if sec is None or not len(sec.entities):
                ys.append(0.0)
                continue
            pts = np.vstack([e.discrete(sec.vertices)
                             for e in sec.entities])
            near = pts[np.abs(pts[:, 0]) < 0.05 * dims.loa]
            ys.append(float(np.abs(near[:, 1]).max()) if len(near)
                      else float(np.abs(pts[:, 1]).max()))
        ys = np.array(ys)
        ax.plot(np.concatenate([-ys[::-1], ys]),
                np.concatenate([zs[::-1], zs]),
                color=color, lw=2.2, label=label, alpha=0.9)
        ax.axhline(dims.draft_design, color=color, ls=":", lw=0.8,
                   alpha=0.5)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_title("중앙 단면 겹침 — 배 몸통을 정면에서 자른 모양",
                 fontsize=11)
    ax.set_xlabel("반폭 [m]")
    ax.set_ylabel("높이 [m] (점선 = 각자의 수선)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def lines_plan_png(meshes, dims_list, labels, path: str | Path) -> Path:
    """조선 표준 선도(lines plan)풍 2단 도면 (2026-08-05 오너 제시
    이미지 재현): 각 배마다 ① 정면 겹침도(body plan — 오른쪽 반 =
    선수 스테이션들, 왼쪽 반 = 선미 스테이션들; 비대칭이 좌우 차이로
    보임) ② 스테이션 사선 배열 (원근 늘어놓기)."""
    import numpy as np

    n = len(meshes)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4.5 * n), dpi=110)
    if n == 1:
        axes = [axes]
    for row, (mesh, dims, label) in enumerate(
            zip(meshes, dims_list, labels)):
        (xmin, _, _), (xmax, _, _) = mesh.bounds
        L = xmax - xmin
        x_mid = 0.5 * (xmin + xmax)
        ax_body, ax_persp = axes[row]

        def station(x):
            sec = mesh.section(plane_origin=[float(x), 0, 0],
                               plane_normal=[1, 0, 0])
            if sec is None or not len(sec.entities):
                return None
            pts = np.vstack([e.discrete(sec.vertices)
                             for e in sec.entities])
            keep = pts[pts[:, 1] >= -1e-6]      # 우현 반쪽
            return keep[np.argsort(keep[:, 2])]

        # ① body plan: 선수 스테이션 → 오른쪽, 선미 → 왼쪽 (관례)
        n_st = 6
        for i, xf in enumerate(np.linspace(0.05, 0.48, n_st)):
            for side, sign, cmap in ((x_mid + xf * L, +1, "#0f766e"),
                                     (x_mid - xf * L, -1, "#dc2626")):
                st = station(side)
                if st is None:
                    continue
                shade = 0.35 + 0.6 * i / n_st
                ax_body.plot(sign * st[:, 1], st[:, 2], lw=1.3,
                             color=cmap, alpha=shade)
        ax_body.axvline(0, color="#94a3b8", lw=0.8, ls=":")
        ax_body.axhline(dims.draft_design, color="#2563eb", lw=0.8,
                        ls="--", alpha=0.6)
        ax_body.set_title(f"{label} — 정면 겹침도 (우=선수 스테이션, "
                          f"좌=선미)", fontsize=10)
        ax_body.set_aspect("equal")
        ax_body.grid(alpha=0.2)

        # ② 사선 배열 (오너 제시 이미지풍): 스테이션들을 옆으로 늘어놓기
        for i, xf in enumerate(np.linspace(0.03, 0.97, 14)):
            st = station(xmin + xf * L)
            if st is None:
                continue
            off = xf * L * 0.9
            ys = np.concatenate([-st[::-1, 1], st[:, 1]])
            zs = np.concatenate([st[::-1, 2], st[:, 2]])
            ax_persp.plot(off + ys, zs + off * 0.12, lw=1.1,
                          color="#134e4a", alpha=0.9)
        ax_persp.set_title(f"{label} — 스테이션 사선 배열 "
                           f"(선수 ← → 선미)", fontsize=10)
        ax_persp.set_aspect("equal")
        ax_persp.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def make_duel_media(row_a, row_b, labels: tuple[str, str],
                    goal: GoalSpec, out_dir: str | Path,
                    course_l: float = 6.0) -> tuple[Path, Path]:
    """파레토 행 2개 → (회전 3D GIF, 주행 비교 GIF)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = [_mesh_for(r, purpose=goal.purpose)
             for r in (row_a, row_b)]
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
    # 20 s 재생 (오너: 시간·화면 늘려 여정 전체 가시화)
    g2 = race_gif(results, dims_list, list(labels), wps,
                  out / "duel_race.gif", vessels=vessels, duration_s=20.0)
    g3 = turning_gif(vessels, dims_list, list(labels),
                     out / "duel_turning.gif",
                     u_entry=goal.target_speed_ms)
    g4 = section_overlay_png(meshes, dims_list, list(labels),
                             out / "duel_section.png")
    g5 = lines_plan_png(meshes, dims_list, list(labels),
                        out / "duel_lines.png")
    return g1, g2, g3, g4, g5
