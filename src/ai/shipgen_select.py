"""ShipGen 조건부 선별 — 조건→문법 연속 매핑 (스펙 2단계, 2026-08-05).

오너 철학: "크기와 속도에 따라 전반적인 파라미터가 변경되는 형식" —
문법을 규칙으로 박지 않고, goal 조건(용도·속도·짐)으로 Ship-D
30,000척을 실측 선별하면 그 조건에 맞는 문법이 데이터에서 저절로
나온다 (빠른 조건 → fine 선형, 무거운 짐 → 풍만 — 4중 게이트와
저항이 심판).

라이선스 자세: 로컬 실행 전용, 생성 형상 커밋·공개 금지 (스펙 §1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.types import GoalSpec


@dataclass(frozen=True)
class ShipGenPick:
    hull_id: int
    vector: np.ndarray      # 45파라미터 (Ship-D 원벡터)
    row: dict               # evaluate_shipd_hull 결과 (4중 게이트 포함)
    n_evaluated: int
    n_passed: int


def select_hull(goal: GoalSpec, target_loa: float,
                pool_size: int = 400, seed: int = 3,
                ) -> ShipGenPick | None:
    """goal 조건으로 Ship-D 표본을 실측 평가 → 4중 게이트 통과 중
    용도 가중 최선을 반환. 통과 0이면 None (정직 거절은 호출측).

    가중: 용도 프리셋과 동일 계보 (recommend와 정합) — 저항·중량·
    안정을 표본 내 정규화해 합산."""
    from data import shipd_loader
    from src.screen_shipd import evaluate_shipd_hull

    vectors, _ = shipd_loader.load_vectors()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vectors), pool_size, replace=False)

    passed = []
    for hid in idx:
        r = evaluate_shipd_hull(vectors[hid], goal, target_loa, int(hid))
        if (r["feasible"] and r.get("space_ok") and r.get("gm_alloc_ok")
                and r.get("trim_ok")):
            passed.append(r)
    if not passed:
        return ShipGenPick(-1, np.array([]), {}, len(idx), 0) \
            if False else None

    res = np.array([r["resistance_n"] for r in passed])
    mas = np.array([r["total_mass_kg"] for r in passed])
    stb = np.array([r["stability_margin"] for r in passed])

    def z(a):
        return (a - a.mean()) / (a.std() + 1e-9)

    # survey 가중 계보 (recommend.py): 안정 우선, 저항·중량 균형
    score = -0.28 * z(res) - 0.27 * z(mas) + 0.45 * z(stb)
    best = passed[int(np.argmax(score))]
    return ShipGenPick(hull_id=int(best["hull_id"]),
                       vector=vectors[int(best["hull_id"])],
                       row=best, n_evaluated=len(idx),
                       n_passed=len(passed))
