"""ELO 대결 매치메이킹 — 비슷한 형태끼리 (오너 결정 2026-08-04).

배경: 극단끼리(경량 1.8 m 통통 vs 순항 3.2 m 날씬) 붙이면 "사용
방식·속도가 아예 달라 비교 불가" (오너, 4R 카드1 기각). 사람의
선호 신호는 **비교 가능한 쌍**에서만 정보가 된다 — 실선 평가도
같은 급끼리 비교하는 것과 같은 원리.

매칭 규칙: 파레토 전선 안에서 형태 거리(정규화 L·L/B·Cb 유클리드)가
가까운 쌍부터 — 형태는 비슷한데 목적값(저항/중량/안정)이 갈리는
쌍이 "미세 트레이드오프 판정"이라는 ELO의 본령.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORM_COLS = ("loa", "lb", "cb")   # 형태·체급 축
OBJ_COLS = ("resistance_n", "total_mass_kg", "stability_margin")


# obj_min 실측 캘리브레이션 (2026-08-04 신 ELO 1차전): z거리
# ~0.3 쌍을 오너가 '차이 사실상 없음' 무승부 판정 — 하한을
# 그 위로 (구분 가능한 트레이드오프만 매칭)
def similar_pairs(df: pd.DataFrame, n_pairs: int = 3,
                  obj_min: float = 0.8,
                  ) -> list[tuple[pd.Series, pd.Series, float]]:
    """전선에서 "형태는 비슷, 목적은 갈리는" 쌍 n개.

    1판 함정 (2026-08-04 실측): 순수 형태 최근접은 전선의 **준중복
    개체끼리** 붙임 (거리 0.00 = 같은 배 대결 — 판정 불능). 조건 =
    형태거리 최소 **AND** 목적거리(정규화) ≥ obj_min — 눈에 보이는
    트레이드오프가 있어야 사람 선호가 정보가 된다.

    반환: (행A, 행B, 형태거리) 목록, 형태거리 오름차순·선체 중복 없음."""
    d = df.reset_index(drop=True)

    def znorm(cols):
        a = d[list(cols)].to_numpy(dtype=float)
        return (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-12)

    zf, zo = znorm(FORM_COLS), znorm(OBJ_COLS)
    n = len(d)
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            obj_d = float(np.linalg.norm(zo[i] - zo[j]))
            if obj_d < obj_min:
                continue                    # 준중복 — 대결 무의미
            cand.append((float(np.linalg.norm(zf[i] - zf[j])), i, j))
    cand.sort()
    used: set[int] = set()
    pairs = []
    for dist, i, j in cand:
        if i in used or j in used:
            continue
        pairs.append((d.iloc[i], d.iloc[j], dist))
        used.update((i, j))
        if len(pairs) >= n_pairs:
            break
    return pairs
