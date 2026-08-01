"""파레토 추천 가중 (#25 2단계, 스펙 2026-08-01).

파레토 전선은 "어느 쪽으로도 지지 않는" 후보 여러 척 — 최종 선택은
사람 몫이지만, 용도를 알면 기본 추천 1척을 자동 표시할 수 있다.

가중 출처 (오너 결정, 혼합): w = (1−α)·용도 프리셋 + α·ELO 취향,
α = min(1, 대결수/10) — "데이터 적으면 프리셋을 믿고, 쌓이면 사람을
믿는다". ELO 대결은 목적 대표선수전(min_mass/max_stability/
min_resistance 이름 규약)이라 대표들의 ELO 점수를 소프트맥스로
확률화하면 그대로 취향 가중치가 된다. 원안 6단계 "사람 피드백 학습"
고리가 추천 체계에 연결되는 지점.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

OBJECTIVES = ("resistance", "mass", "stability")

# 용도 프리셋 (개략 판단 — 실무 근거 없음, ELO가 쌓이며 희석되는 설계):
# survey 안정 우선(계측은 흔들림이 적), patrol 저항 우선(순항 효율),
# workboat 중량 우선(선체가 가벼울수록 적재 여유).
_PRESETS = {
    "survey": {"resistance": 0.3, "mass": 0.2, "stability": 0.5},
    "patrol": {"resistance": 0.5, "mass": 0.2, "stability": 0.3},
    "workboat": {"resistance": 0.2, "mass": 0.5, "stability": 0.3},
}

# 대결 후보 이름 → 목적 (ELO 이력의 이름 규약)
_NAME_TO_OBJ = {"min_resistance": "resistance", "min_mass": "mass",
                "max_stability": "stability"}

ELO_K = 32          # src/hitl ELO와 동일 관례
ELO_TEMPERATURE = 100.0  # 소프트맥스 온도 — ELO 점수차 100 ≈ 확률 64:36
ALPHA_FULL_AT = 10  # 대결 몇 건이면 ELO가 완전 주도하나


def preset_weights(purpose: str) -> dict:
    return dict(_PRESETS.get(purpose, _PRESETS["survey"]))


def _elo_ratings(csv_path: Path) -> tuple[dict, int]:
    """대결 이력 → 목적 대표 3인의 ELO 점수. 반환 (점수, 대결 수)."""
    ratings = {obj: 1000.0 for obj in OBJECTIVES}
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return ratings, 0
    df = pd.read_csv(csv_path)
    n = 0
    for _, row in df.iterrows():
        w_obj = next((o for k, o in _NAME_TO_OBJ.items()
                      if k in str(row.winner)), None)
        l_obj = next((o for k, o in _NAME_TO_OBJ.items()
                      if k in str(row.loser)), None)
        if w_obj is None or l_obj is None or w_obj == l_obj:
            continue
        expected_w = 1.0 / (1.0 + 10 ** ((ratings[l_obj] - ratings[w_obj])
                                         / 400.0))
        ratings[w_obj] += ELO_K * (1.0 - expected_w)
        ratings[l_obj] -= ELO_K * (1.0 - expected_w)
        n += 1
    return ratings, n


def elo_weights(csv_path: Path) -> tuple[dict, int]:
    """ELO 점수 → 소프트맥스 가중치. 이력 없으면 균등."""
    ratings, n = _elo_ratings(csv_path)
    exps = {o: math.exp(r / ELO_TEMPERATURE) for o, r in ratings.items()}
    total = sum(exps.values())
    return {o: e / total for o, e in exps.items()}, n


def blend(preset: dict, elo: dict, n_duels: int) -> dict:
    alpha = min(1.0, n_duels / ALPHA_FULL_AT)
    return {o: (1 - alpha) * preset[o] + alpha * elo[o] for o in OBJECTIVES}


def recommend(front: pd.DataFrame, purpose: str,
              elo_csv: Path = Path("data/user_comparisons.csv")
              ) -> tuple[int, dict, str]:
    """전선 후보 중 추천 1척. 반환 (행 인덱스, 최종 가중, 이유 문장).

    목적값을 0~1 정규화 (모두 '클수록 좋다' 방향으로 통일:
    저항·중량은 뒤집음) → 가중합 최대."""
    elo, n = elo_weights(elo_csv)
    w = blend(preset_weights(purpose), elo, n)

    def norm(series, lower_is_better):
        lo, hi = series.min(), series.max()
        if hi == lo:
            return series * 0 + 1.0
        x = (series - lo) / (hi - lo)
        return 1.0 - x if lower_is_better else x

    score = (w["resistance"] * norm(front["resistance_n"], True)
             + w["mass"] * norm(front["total_mass_kg"], True)
             + w["stability"] * norm(front["stability_margin"], False))
    idx = int(score.idxmax())
    labels = {"resistance": "저항", "mass": "중량", "stability": "안정"}
    top = max(w, key=w.get)
    alpha = min(1.0, n / ALPHA_FULL_AT)
    why = (f"{purpose} 가중 (저항 {w['resistance']:.2f} / 중량 "
           f"{w['mass']:.2f} / 안정 {w['stability']:.2f} — "
           f"{labels[top]} 우선, ELO {n}건 반영률 {alpha:.0%})")
    return idx, w, why
