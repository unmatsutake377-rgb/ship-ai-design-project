"""NSGA-II 다목적 선형 최적화 (spec §7 M5b-2 1차, 오너 결정 1·3).

설계 변수 4개: [L, L/B, B/T, Cb] — Wigley 공간 (Ship-D 45파라미터 확장은 2차).
목적 3개 (전부 최소화):
  f1 = 전저항 @ 목표속도 [N]
  f2 = 전체 중량 [kg] (비용 대리값)
  f3 = −안정여유,  여유 = min(GM/B − 0.04, 0.40 − GM/B) — 밴드 중앙일수록 큼
제약: 배수량형(Fn<0.4), 정역학 필터 3종, Cb 도달범위.
위반·예외(침수·수렴실패 등) = 사망 페널티 (자연 도태).

평가는 물리 직접 (파이프라인의 design_spiral 재사용) — 대리모델 가속은 M5b-1 후.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai.hull_generator import generate_hull_mesh
from src.core.regime import FN_DISPLACEMENT_MAX, froude_length
from src.core.types import GoalSpec, MainDimensions
from src.pipeline import design_spiral

# 설계 변수 범위 (실선 통계 주변 + 생성기 도달범위 안)
BOUNDS_LOW = np.array([1.2, 1.7, 2.5, 0.35])   # L, L/B, B/T, Cb
BOUNDS_HIGH = np.array([6.0, 4.0, 6.0, 0.60])
DEPTH_OVER_DRAFT = 1.6  # dimension_estimator와 동일 (건현 여유)

DEATH_PENALTY = np.array([1e6, 1e6, 1e6])
GM_BAND = (0.04, 0.40)


def dims_from_vector(x: np.ndarray) -> MainDimensions:
    loa, lb, bt, cb = (float(v) for v in x)
    beam = loa / lb
    draft = beam / bt
    return MainDimensions(loa=loa, beam=beam, depth=DEPTH_OVER_DRAFT * draft,
                          draft_design=draft, cb=cb)


def evaluate_candidate(x: np.ndarray, goal: GoalSpec) -> dict:
    """후보 1개 평가. 실패는 feasible=False + 페널티 목적값."""
    dims = dims_from_vector(x)
    base = {"loa": dims.loa, "lb": dims.loa / dims.beam,
            "bt": dims.beam / dims.draft_design, "cb": dims.cb}
    try:
        if froude_length(goal.target_speed_ms, dims.loa) >= FN_DISPLACEMENT_MAX:
            raise ValueError("반배수량 영역")
        mesh = generate_hull_mesh(dims)
        weights, hydro, resist, motors, batt_kg, _ = \
            design_spiral(mesh, dims, goal)
        if not hydro.passed:
            raise ValueError(f"정역학 필터 불합격: {hydro.checks}")
        gmb = hydro.gm / dims.beam
        margin = min(gmb - GM_BAND[0], GM_BAND[1] - gmb)
        return {**base, "resistance_n": resist.total,
                "total_mass_kg": weights.total_mass,
                "stability_margin": margin, "feasible": True}
    except Exception as exc:  # 물리 위반 전부 도태 대상
        return {**base, "resistance_n": float(DEATH_PENALTY[0]),
                "total_mass_kg": float(DEATH_PENALTY[1]),
                "stability_margin": -float(DEATH_PENALTY[2]),
                "feasible": False, "reason": str(exc)[:80]}


def optimize_design(goal: GoalSpec, pop_size: int = 24, n_gen: int = 20,
                    seed: int = 1, verbose: bool = False) -> pd.DataFrame:
    """NSGA-II 실행 → 최종 세대 비지배(파레토) 후보 DataFrame."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize

    class ShipDesignProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=4, n_obj=3, xl=BOUNDS_LOW, xu=BOUNDS_HIGH)

        def _evaluate(self, x, out, *args, **kwargs):
            r = evaluate_candidate(x, goal)
            out["F"] = [r["resistance_n"], r["total_mass_kg"],
                        -r["stability_margin"]]

    res = minimize(ShipDesignProblem(), NSGA2(pop_size=pop_size),
                   ("n_gen", n_gen), seed=seed, verbose=verbose)

    rows = [evaluate_candidate(x, goal) for x in np.atleast_2d(res.X)]
    df = pd.DataFrame([r for r in rows if r["feasible"]])
    return df.reset_index(drop=True)


def plot_pareto(df: pd.DataFrame, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=110)
    sc = ax.scatter(df["resistance_n"], df["total_mass_kg"],
                    c=df["stability_margin"], cmap="viridis", s=60,
                    edgecolor="#334155", linewidth=0.5)
    fig.colorbar(sc, label="안정여유 min(GM/B−0.04, 0.40−GM/B)")
    ax.set_xlabel("전저항 @ 목표속도 [N]")
    ax.set_ylabel("전체 중량 [kg]")
    ax.set_title("파레토 전선 — 저항 vs 중량 (색: 안정여유)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="NSGA-II 선형 최적화 (M5b-2)")
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--payload", type=float, required=True)
    parser.add_argument("--endurance", type=float, default=4.0)
    parser.add_argument("--pop", type=int, default=24)
    parser.add_argument("--gen", type=int, default=20)
    parser.add_argument("--out", default="outputs/pareto")
    args = parser.parse_args(argv)

    goal = GoalSpec(target_speed_ms=args.speed, payload_kg=args.payload,
                    purpose="survey", endurance_h=args.endurance)
    print(f"NSGA-II 시작: 개체 {args.pop} × 세대 {args.gen} "
          f"(평가 ~{args.pop * args.gen}회, 물리 직접 — 수 분 소요)")
    df = optimize_design(goal, pop_size=args.pop, n_gen=args.gen, verbose=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "pareto.csv", index=False)
    plot_pareto(df, out / "pareto.png")

    # 상위 후보 3척 STL (ELO 쌍대비교용): 저항 최소·중량 최소·안정여유 최대
    picks = {
        "min_resistance": df["resistance_n"].idxmin(),
        "min_mass": df["total_mass_kg"].idxmin(),
        "max_stability": df["stability_margin"].idxmax(),
    }
    for tag, idx in picks.items():
        row = df.loc[idx]
        dims = dims_from_vector(np.array([row["loa"], row["lb"],
                                          row["bt"], row["cb"]]))
        generate_hull_mesh(dims).export(out / f"candidate_{tag}.stl")

    print(f"파레토 후보 {len(df)}척 → {out}/pareto.csv, pareto.png, "
          f"candidate_*.stl (3종)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
