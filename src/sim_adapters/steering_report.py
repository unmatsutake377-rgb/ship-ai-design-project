"""조타 구성별 조종성 비교 리포트 (스펙 2026-08-03 §3-2단계).

한 설계(report.json)에 대해 3가지 조타 구성을 같은 코스에서 달리게
하고 나란히 채점한다:
  diff    — 차동 2발 (기존)
  rudder2 — 구성 A: 차동 2발 + 러더
  rudder1 — 구성 B: 단일 추력기 + 러더

공정성: 차동은 forward_only=True(후진 금지 — 실물·gz 프로펠러 근사)로
달린다. 러더 배분은 구조상 이미 전진 전용이므로 같은 세계 조건.
후진 허용 세계로 비교하면 차동의 자기지속 루프가 숨어 비교가 왜곡됨
(1단계 발견).

채점 항목:
  - 완주 여부 / 소요 시간
  - 경로비 (실주행 거리 / 코스 길이 — 1.0에 가까울수록 곧게 감)
  - 횡이탈 σ (코스 선분에서 벗어난 거리의 표준편차)
  - 저속 추종 (0.77 m/s 명령 시 실제 평균 속도 — 자기지속 루프 검사)

사용: python -m src.sim_adapters.steering_report --report outputs/X/report.json --out outputs/steering_X
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.sim_adapters.python_sim import (
    STEERING_MODES,
    SimResult,
    default_square_course,
    simulate_course,
    vessel_from_report,
)

LOW_SPEED_CMD = 0.77   # 자기지속 루프 검사용 저속 명령 [m/s] (1단계 판정관)
T_MAX = 1500.0


def _cross_track_sigma(result: SimResult,
                       waypoints: list[tuple[float, float]]) -> float:
    """궤적 각 점에서 코스 폴리라인까지 최단 거리의 표준편차."""
    pts = [(0.0, 0.0)] + list(waypoints)
    xs, ys = np.array(result.x), np.array(result.y)
    if len(xs) == 0:
        return float("nan")
    dists = np.full(len(xs), np.inf)
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        dx, dy = x2 - x1, y2 - y1
        seg2 = dx * dx + dy * dy
        t = np.clip(((xs - x1) * dx + (ys - y1) * dy) / max(seg2, 1e-12),
                    0.0, 1.0)
        d = np.hypot(xs - (x1 + t * dx), ys - (y1 + t * dy))
        dists = np.minimum(dists, d)
    return float(np.std(dists))


def _score(result: SimResult, waypoints: list[tuple[float, float]],
           course_len: float) -> dict:
    xs, ys = np.array(result.x), np.array(result.y)
    path = float(np.sum(np.hypot(np.diff(xs), np.diff(ys)))) if len(xs) > 1 else 0.0
    u_tail = result.u[len(result.u) // 4:]
    return {
        "success": result.success,
        "duration_s": round(result.duration_s, 1),
        "path_ratio": round(path / course_len, 3),
        "cross_track_sigma_m": round(_cross_track_sigma(result, waypoints), 3),
        "u_mean_ms": round(float(np.mean(u_tail)), 3) if u_tail else float("nan"),
    }


def compare_steering(report: dict) -> dict:
    """3구성 × (목표 속도 코스 + 저속 추종) 채점표."""
    vessel = vessel_from_report(report)
    waypoints = default_square_course(vessel.loa)
    course_len = 4 * 10.0 * vessel.loa
    u_target = report["goal"]["target_speed_ms"]

    table: dict = {"u_target_ms": u_target, "low_speed_cmd_ms": LOW_SPEED_CMD,
                   "modes": {}}
    trajectories = {}
    for mode in STEERING_MODES:
        fwd = (mode == "diff")     # 러더 배분은 구조상 전진 전용
        cruise = simulate_course(vessel, waypoints, u_target, steering=mode,
                                 t_max=T_MAX, forward_only=fwd)
        slow = simulate_course(vessel, waypoints, LOW_SPEED_CMD,
                               steering=mode, t_max=T_MAX, forward_only=fwd)
        table["modes"][mode] = {
            "cruise": _score(cruise, waypoints, course_len),
            "low_speed": _score(slow, waypoints, course_len),
        }
        trajectories[mode] = cruise
    table["_trajectories"] = trajectories   # 그림용 (JSON 저장 전 제거)
    return table


def plot_compare(table: dict, waypoints: list[tuple[float, float]],
                 path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    labels = {"diff": "차동 2발", "rudder2": "A: 차동+러더",
              "rudder1": "B: 단일+러더"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), dpi=110)
    wx, wy = zip(*waypoints)
    for ax, mode in zip(axes, STEERING_MODES):
        res = table["_trajectories"][mode]
        s = table["modes"][mode]
        ax.plot(res.x, res.y, lw=1.1, color="#2563eb")
        ax.plot(wx, wy, "o--", color="#d97706", alpha=0.6, ms=4)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        c, ls = s["cruise"], s["low_speed"]
        ax.set_title(f"{labels[mode]}\n경로비 {c['path_ratio']} σ{c['cross_track_sigma_m']}m"
                     f" | 저속 {ls['u_mean_ms']}/{table['low_speed_cmd_ms']}",
                     fontsize=9)
    fig.suptitle("조타 구성 비교 — 목표 속도 궤적 (전진 전용 세계)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="조타 구성 비교 리포트")
    parser.add_argument("--report", required=True, help="report.json 경로")
    parser.add_argument("--out", default="outputs/steering_compare",
                        help="출력 디렉토리")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.report).read_text())
    table = compare_steering(report)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vessel_loa = report["dimensions"]["loa"]
    waypoints = default_square_course(vessel_loa)
    plot_compare(table, waypoints, out / "steering_compare.png")
    table.pop("_trajectories")
    (out / "steering_compare.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2))

    print(f"목표 {table['u_target_ms']} m/s / 저속 명령 {LOW_SPEED_CMD} m/s")
    hdr = f"{'구성':<12} {'완주':<4} {'시간s':>7} {'경로비':>7} {'σ[m]':>7} {'저속u':>7}"
    print(hdr)
    for mode in STEERING_MODES:
        c = table["modes"][mode]["cruise"]
        ls = table["modes"][mode]["low_speed"]
        print(f"{mode:<12} {'✓' if c['success'] else '✗':<4} "
              f"{c['duration_s']:>7} {c['path_ratio']:>7} "
              f"{c['cross_track_sigma_m']:>7} {ls['u_mean_ms']:>7}")
    print(f"저장: {out}/steering_compare.json, .png")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
