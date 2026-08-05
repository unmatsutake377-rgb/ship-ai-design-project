"""ShipGen 45파라미터 NSGA-II (스펙 3단계, 2026-08-05).

기존 Wigley 4변수 최적화(src/optimize.py)의 Ship-D 판:
- 탐색 공간: 45파라미터, 경계 = InputVectors_30k 실값 분포의
  열별 [min, max] (하드코딩 경계 금지 — 데이터가 정함)
- 초기 개체군: 실척 3만 척에서 무작위 시드 — 무작위 45차원 점이
  아니라 제약을 이미 만족하는 실선 분포에서 출발 (부담 완화 1호)
- 제약: 원저자 input_Constraints 49개 (≤0 만족) — 위반은 물리 평가
  없이 사망 페널티 (부담 완화 2호)
- 평가: evaluate_shipd_hull (설계 나선 + 4중 게이트) — 불합격도
  사망 페널티, 최종 전선은 전부 실물리 재평가 (optimize.py 관례)

가짜 전멸 교훈(worklog 08-05): 평가 실패를 조용히 삼키지 않고
사유별 카운터를 집계해 반환 — 인프라 사망이 물리로 위장 못 하게.

라이선스 자세: 로컬 실행 전용, 생성 형상 커밋·공개 금지 (스펙 §1).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.core.types import GoalSpec

DEATH_PENALTY = (1e6, 1e6, 1e6)   # (저항, 중량, -안정) 최악값

# 부분공간: 주요치수(0~6)·중앙단면(7~10) — Ship-D 벡터 앞 11개.
# 0(LOA)은 10 고정 정규화라 실질 10차원. 선수 9·선미 11·벌브 14는
# 기준 척 값 고정 (1차 성적표: 45차원 자유 탐색은 제약 사망 66%로
# 예산 낭비 — worklog 08-05 3단계).
SUBSPACE_MAIN_MID = tuple(range(11))


def dataset_bounds() -> tuple[np.ndarray, np.ndarray]:
    """탐색 경계 = 3만 척 실값 분포의 열별 [min, max]."""
    from data import shipd_loader

    vectors, _ = shipd_loader.load_vectors()
    return vectors.min(axis=0), vectors.max(axis=0)


def constraints_ok(vector: np.ndarray) -> bool:
    """원저자 대수 제약 49개 (≤0 만족). 형상 생성 없이 검사 가능."""
    from data.shipd_loader import _hull_parameterization

    HP = _hull_parameterization()
    try:
        c = HP(np.asarray(vector, dtype=np.float64)).input_Constraints()
        return bool(np.all(np.asarray(c, dtype=float) <= 0.0))
    except Exception:
        return False


def optimize_shipgen(goal: GoalSpec, target_loa: float,
                     pop_size: int = 32, n_gen: int = 12,
                     seed: int = 1, verbose: bool = False,
                     free_idx: tuple[int, ...] | None = None,
                     base_vector: np.ndarray | None = None,
                     ) -> pd.DataFrame:
    """NSGA-II (Ship-D 파라미터) → 최종 세대 비지배 전선 DataFrame.

    free_idx 주면 부분공간 탐색: 그 인덱스만 자유, 나머지는
    base_vector(기본 = 시드 첫 척) 값 고정. None이면 45 전차원.

    반환 열: evaluate_shipd_hull 결과 + vector_json (45파라미터 —
    재현·메쉬 재생성용). df.attrs["death_stats"]에 사망 사유 집계."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize

    from data import shipd_loader
    from src.screen_shipd import evaluate_shipd_hull

    vectors, _ = shipd_loader.load_vectors()
    xl45, xu45 = dataset_bounds()
    rng = np.random.default_rng(seed)
    seeds = vectors[rng.choice(len(vectors), pop_size, replace=False)]

    if free_idx is None:
        free = np.arange(45)
        base = None
    else:
        free = np.asarray(free_idx, int)
        base = np.asarray(base_vector if base_vector is not None
                          else seeds[0], float)

    def _full(x: np.ndarray) -> np.ndarray:
        if base is None:
            return np.asarray(x, float)
        v = base.copy()
        v[free] = x
        return v

    stats = {"constraint": 0, "gate": 0, "alive": 0}

    def _objectives(x: np.ndarray) -> tuple[float, float, float] | None:
        v = _full(x)
        if not constraints_ok(v):
            stats["constraint"] += 1
            return None
        r = evaluate_shipd_hull(v, goal, target_loa)
        if not (r["feasible"] and r.get("space_ok")
                and r.get("gm_alloc_ok") and r.get("trim_ok")):
            stats["gate"] += 1
            return None
        stats["alive"] += 1
        return (r["resistance_n"], r["total_mass_kg"],
                -r["stability_margin"])

    class ShipGenProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(free), n_obj=3,
                             xl=xl45[free], xu=xu45[free])

        def _evaluate(self, x, out, *args, **kwargs):
            f = _objectives(np.asarray(x, float))
            out["F"] = list(f) if f is not None else list(DEATH_PENALTY)

    res = minimize(ShipGenProblem(),
                   NSGA2(pop_size=pop_size, sampling=seeds[:, free]),
                   ("n_gen", n_gen), seed=seed, verbose=verbose)

    # 최종 전선 실물리 재평가 + 게이트 재확인 (optimize.py 관례)
    rows = []
    for x in np.atleast_2d(res.X):
        v = _full(x)
        if not constraints_ok(v):
            continue
        r = evaluate_shipd_hull(v, goal, target_loa)
        if (r["feasible"] and r.get("space_ok") and r.get("gm_alloc_ok")
                and r.get("trim_ok")):
            r["vector_json"] = json.dumps(np.asarray(v, float).tolist())
            rows.append(r)
    df = pd.DataFrame(rows).reset_index(drop=True)
    df.attrs["death_stats"] = dict(stats)
    if verbose:
        print(f"사망 집계: 제약 {stats['constraint']} / 게이트 "
              f"{stats['gate']} / 생존 {stats['alive']}")
    return df
