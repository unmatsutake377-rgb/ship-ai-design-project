"""Ship-D 대규모 선별 → 실선형 파레토 (spec §7 M5b-2 2차, 2026-07-28).

접근: 45차원 직접 생성이 아니라 유효성 보증된 3만 척의 **선별** —
무작위 45파라미터 벡터는 대부분 자기교차 선체라 (원저자 제약 49개의
존재 이유), 생성 최적화는 대리모델(M5b-1)로 평가가 빨라진 뒤가 적기.

평가 사슬: scaled_mesh(상사 축소) → 설계 나선(메쉬형 Michell — 이중검증)
→ 정역학 필터 → (저항, 중량, 안정여유). Wigley 파레토(optimize.py)와
같은 목적 3축이라 직접 비교 가능.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from data import shipd_loader
from src.core.regime import FN_DISPLACEMENT_MAX, froude_length
from src.core.types import GoalSpec, MainDimensions
from src.optimize import GM_BAND
from src.physics.resistance import total_resistance_mesh
from src.pipeline import design_spiral

DEPTH_OVER_DRAFT_FALLBACK = 1.6  # dims.depth 미사용 경로용 형식값


def evaluate_shipd_hull(vector: np.ndarray, goal: GoalSpec,
                        target_loa: float, hull_id: int = -1) -> dict:
    """Ship-D 선체 1척 평가. 실패 = feasible False + 사유."""
    base = {"hull_id": hull_id}
    try:
        if froude_length(goal.target_speed_ms, target_loa) \
                >= FN_DISPLACEMENT_MAX:
            raise ValueError("반배수량 영역")
        mesh = shipd_loader.scaled_mesh(vector, target_loa)
        beam = float(mesh.extents[1])
        depth = float(mesh.bounds[1][2])
        # 설계 나선용 형식 치수 (평가는 전부 메쉬 기반 — cb는 미사용 경로)
        dims = MainDimensions(loa=target_loa, beam=beam, depth=depth,
                              draft_design=depth / DEPTH_OVER_DRAFT_FALLBACK,
                              cb=0.5)
        weights, hydro, resist, motors, batt_kg, _ = design_spiral(
            mesh, dims, goal,
            resistance_fn=lambda m_, d_, s_, w_=None:
                total_resistance_mesh(m_, target_loa, d_, s_),
        )
        if not hydro.passed:
            raise ValueError(f"필터 불합격: {hydro.checks}")
        # 다구획 공간 검사 (2026-08-03 재개방 — 스펙 multibay-maxbox):
        # 단일 상자 보류(과혹)를 다구획+예약 절단으로 대체. 라벨 호환을
        # 위해 feasible(중량·안정)은 그대로 두고 space_ok를 별도 열로 —
        # 기존 1만 라벨과 의미가 섞이지 않게 (파레토는 둘 다 요구).
        from src.physics.cargo_hold import (
            bays_from_hold,
            cargo_kg_interval,
            gm_band_reachable,
            multibay_hold,
            reserved_volume_for,
        )
        from src.physics.maxbox import payload_volume_for

        pv, _ = payload_volume_for(goal.payload_kg, goal.purpose)
        hold = multibay_hold(mesh, depth,
                             reserved_volume_for(weights.propulsion_mass),
                             stern="xmax")   # Ship-D 관례: 뱃머리 x=0
        # 2단계 구간 판정: 짐을 실제 구획에 부었을 때 도달 가능한
        # KG 구간이 GM 밴드와 겹치나 — 합격 배치의 존재 판정.
        # (기존 stability_margin은 "짐 VCG=0.6D 가정" 한 점 평가 —
        #  라벨 호환 위해 보존, 구간 판정은 별도 열)
        interval = cargo_kg_interval(bays_from_hold(hold), goal.payload_kg,
                                     goal.payload_kg / max(pv, 1e-9))
        if interval is not None:
            a = weights.assumptions
            fixed = (weights.structure_mass * a["vcg_structure"]
                     + weights.propulsion_mass * a["vcg_propulsion"])
            gm_alloc_ok, gm_alloc_margin = gm_band_reachable(
                interval[0], interval[1], km=hydro.kb + hydro.bm,
                fixed_moment=fixed, cargo_mass=goal.payload_kg,
                total_mass=weights.total_mass, beam=beam, band=GM_BAND)
        else:
            gm_alloc_ok, gm_alloc_margin = False, float("nan")
        gmb = hydro.gm / beam
        margin = min(gmb - GM_BAND[0], GM_BAND[1] - gmb)
        return {**base, "beam": beam, "draft": hydro.draft,
                "resistance_n": resist.total,
                "total_mass_kg": weights.total_mass,
                "stability_margin": margin, "feasible": True,
                "space_ok": bool(hold.total_volume >= pv),
                "hold_volume_m3": hold.total_volume,
                "n_bays": len(hold.boxes),
                "gm_alloc_ok": bool(gm_alloc_ok),
                "gm_alloc_margin": gm_alloc_margin, "reason": ""}
    except Exception as exc:
        return {**base, "beam": float("nan"), "draft": float("nan"),
                "resistance_n": float("nan"),
                "total_mass_kg": float("nan"),
                "stability_margin": float("nan"),
                "feasible": False, "space_ok": False,
                "hold_volume_m3": float("nan"), "n_bays": 0,
                "gm_alloc_ok": False, "gm_alloc_margin": float("nan"),
                "reason": str(exc)[:80]}


def _mark_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """3목적 (저항↓, 중량↓, −안정여유↓) 비지배 후보 표시."""
    f = np.column_stack([df["resistance_n"], df["total_mass_kg"],
                         -df["stability_margin"]])
    n = len(f)
    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and (f[j] <= f[i]).all() and (f[j] < f[i]).any():
                pareto[i] = False
                break
    out = df.copy()
    out["pareto"] = pareto
    return out


def screen(goal: GoalSpec, target_loa: float, n_samples: int = 300,
           seed: int = 1, verbose: bool = False) -> pd.DataFrame:
    """무작위 표본 전수 평가 → feasible 행 + 파레토 표시 DataFrame."""
    vectors, _ = shipd_loader.load_vectors()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vectors), n_samples, replace=False)
    rows = []
    for k, i in enumerate(idx):
        rows.append(evaluate_shipd_hull(vectors[i], goal, target_loa,
                                        hull_id=int(i)))
        if verbose and (k + 1) % 25 == 0:
            ok = sum(r["feasible"] for r in rows)
            print(f"  {k + 1}/{n_samples} 평가 — feasible {ok}")
    df = pd.DataFrame(rows)
    # 파레토 후보 = 중량·안정 AND 공간 AND 배치 존재 (2026-08-03)
    candidate = df["feasible"] & df["space_ok"] & df["gm_alloc_ok"]
    feasible = df[candidate].reset_index(drop=True)
    if feasible.empty:
        return df.assign(pareto=False)
    marked = _mark_pareto(feasible)
    # 탈락 행도 보존 (pareto=False) — 대리모델 타당성 분류 학습에 필수.
    # (초기 버전은 feasible만 저장 → 분류기가 '전부 통과'를 학습하는
    #  결손 라벨 버그가 있었음, 2026-07-28)
    rejected = df[~candidate].assign(pareto=False)
    return pd.concat([marked, rejected], ignore_index=True)


def plot_screen(df: pd.DataFrame, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=110)
    rest = df[~df["pareto"]]
    front = df[df["pareto"]]
    ax.scatter(rest["resistance_n"], rest["total_mass_kg"], s=25,
               color="#94a3b8", alpha=0.5, label=f"통과 {len(rest)}척")
    sc = ax.scatter(front["resistance_n"], front["total_mass_kg"],
                    c=front["stability_margin"], cmap="viridis", s=80,
                    edgecolor="#1e293b", linewidth=0.6,
                    label=f"파레토 {len(front)}척")
    fig.colorbar(sc, label="안정여유")
    ax.set_xlabel("전저항 @ 목표속도 [N]")
    ax.set_ylabel("전체 중량 [kg]")
    ax.set_title("Ship-D 선별 파레토 — 실선형 3만 척 표본")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Ship-D 선별 파레토 (M5b-2 2차)")
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--payload", type=float, required=True)
    parser.add_argument("--loa", type=float, required=True,
                        help="목표 선체 길이 [m] (상사 축소 스케일)")
    parser.add_argument("--endurance", type=float, default=4.0)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--out", default="outputs/shipd_pareto")
    args = parser.parse_args(argv)

    goal = GoalSpec(target_speed_ms=args.speed, payload_kg=args.payload,
                    purpose="survey", endurance_h=args.endurance)
    print(f"Ship-D 선별 시작: 표본 {args.n}척 (척당 ~3초 — 수십 분)")
    df = screen(goal, args.loa, n_samples=args.n, verbose=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "screen.csv", index=False)
    ok = df[df["feasible"]]
    if ok.empty:
        print("feasible 0척 — 조건 완화 필요")
        return 2
    plot_screen(ok, out / "pareto.png")

    front = ok[ok["pareto"]].sort_values("resistance_n")
    vectors, _ = shipd_loader.load_vectors()
    for rank, (_, row) in enumerate(front.head(3).iterrows(), 1):
        mesh = shipd_loader.scaled_mesh(vectors[int(row.hull_id)], args.loa)
        mesh.export(out / f"pareto_{rank}_hull{int(row.hull_id)}.stl")

    print(f"평가 {len(df)}척 (통과 {len(ok)}) / 파레토 "
          f"{int(ok['pareto'].sum())}척 → {out}/")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
