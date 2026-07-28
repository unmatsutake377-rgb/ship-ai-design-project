"""가상 전수 선별 (spec §7 M5b-1 활용, 2026-07-28).

흐름: 실측 평가분(screen.csv) 학습 → 3만 척 전수 예측(밀리초/척)
→ 예측-타당 중 예측-파레토 상위 K척 → **진짜 물리 재검증**
→ 기존 실측 feasible과 합산한 최종 파레토.

정직 원칙: 대리모델은 후보 추천만 — 결과 CSV의 모든 수치는 실물리 산출.
대리모델 검증 지표(분류 정확도·R²)를 함께 저장.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import shipd_loader
from src.ai.surrogate import train_surrogate
from src.core.types import GoalSpec
from src.screen_shipd import _mark_pareto, evaluate_shipd_hull


def load_training_data(screen_csv: str | Path
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray]:
    """screen.csv(실측) + Ship-D 벡터 결합 → (X, y_feas, Y_obj, hull_ids).

    주의: screen.csv는 feasible 행만 저장돼 있던 초기 버전도 있으므로
    reason 열 유무와 무관하게 hull_id 기준으로 결합한다.
    """
    df = pd.read_csv(screen_csv)
    vectors, _ = shipd_loader.load_vectors()
    ids = df["hull_id"].to_numpy(dtype=int)
    x = vectors[ids]
    y_feas = df["feasible"].to_numpy(dtype=float)
    y_obj = df[["resistance_n", "total_mass_kg",
                "stability_margin"]].to_numpy(dtype=float)
    return x, y_feas, y_obj, ids


def virtual_screen(goal: GoalSpec, target_loa: float,
                   screen_csv: str | Path, top_k: int = 80,
                   epochs: int = 400, seed: int = 0,
                   verbose: bool = False) -> tuple[pd.DataFrame, dict]:
    """대리모델 학습 → 전수 예측 → 상위 K 실물리 재검증 → 합산 파레토."""
    x, y_feas, y_obj, known_ids = load_training_data(screen_csv)
    model, metrics = train_surrogate(x, y_feas, y_obj,
                                     epochs=epochs, seed=seed)
    if verbose:
        print(f"대리모델 지표: {metrics}")

    vectors, _ = shipd_loader.load_vectors()
    feas_p, obj_p = model.predict(vectors)

    # 예측-타당 & 미평가 선체 중 예측-파레토 근사: 저항 예측 순 상위 후보
    # (3목적 예측 파레토 전체는 후보 폭이 넓어짐 — 저항·중량·여유 각 축
    #  상위를 섞어 다양성 확보)
    candidate = np.where((feas_p > 0.5)
                         & ~np.isin(np.arange(len(vectors)), known_ids))[0]
    if len(candidate) == 0:
        raise RuntimeError("예측-타당 후보 0척 — 대리모델/데이터 점검 필요")
    per_axis = max(1, top_k // 3)
    picks: list[int] = []
    for k, ascending in ((0, True), (1, True), (2, False)):
        order = candidate[np.argsort(obj_p[candidate, k])]
        if not ascending:
            order = order[::-1]
        picks.extend(order[:per_axis])
    picks = list(dict.fromkeys(picks))[:top_k]

    if verbose:
        print(f"재검증 대상 {len(picks)}척 (예측-타당 {len(candidate)}척 중)")
    rows = []
    for k, i in enumerate(picks):
        rows.append(evaluate_shipd_hull(vectors[i], goal, target_loa,
                                        hull_id=int(i)))
        if verbose and (k + 1) % 20 == 0:
            ok = sum(r["feasible"] for r in rows)
            print(f"  재검증 {k + 1}/{len(picks)} — 실측 통과 {ok}")

    new_df = pd.DataFrame(rows)
    new_ok = new_df[new_df["feasible"]].reset_index(drop=True)
    metrics["reverify_pass_rate"] = float(len(new_ok) / max(1, len(picks)))

    old = pd.read_csv(screen_csv)
    old_ok = old[old["feasible"]][["hull_id", "beam", "draft", "resistance_n",
                                   "total_mass_kg", "stability_margin",
                                   "feasible", "reason"]]
    combined = pd.concat([old_ok, new_ok[old_ok.columns]],
                         ignore_index=True)
    combined = _mark_pareto(combined)
    combined["source"] = np.where(combined["hull_id"].isin(known_ids),
                                  "screen300", "surrogate_pick")
    return combined, metrics


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="가상 전수 선별 (M5b-1)")
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--payload", type=float, required=True)
    parser.add_argument("--loa", type=float, required=True)
    parser.add_argument("--endurance", type=float, default=4.0)
    parser.add_argument("--screen-csv", default="outputs/shipd_pareto/screen.csv")
    parser.add_argument("--topk", type=int, default=80)
    parser.add_argument("--out", default="outputs/virtual_screen")
    args = parser.parse_args(argv)

    goal = GoalSpec(target_speed_ms=args.speed, payload_kg=args.payload,
                    purpose="survey", endurance_h=args.endurance)
    combined, metrics = virtual_screen(goal, args.loa, args.screen_csv,
                                       top_k=args.topk, verbose=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out / "combined.csv", index=False)
    with open(out / "surrogate_metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    front = combined[combined["pareto"]]
    new_on_front = front[front["source"] == "surrogate_pick"]
    print(f"합산 통과 {len(combined)}척 / 파레토 {len(front)}척 "
          f"(그중 대리모델 발굴 {len(new_on_front)}척)")
    print(f"저항 최저 {combined.resistance_n.min():.1f} N · "
          f"중량 최저 {combined.total_mass_kg.min():.1f} kg")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
