"""내항 기준 — 횡요 고유주기 (내항성 3단계 선행 조각).

IMO IS Code 2008 (MSC.267(85)) Weather Criterion의 규정 공식:
  T_roll = 2·c·B / √GM   [s]
  c = 0.373 + 0.023·(B/T) − 0.043·(Lwl/100)
A급 근거 (국제 규정 명문) — 횡동요 회전반경 근사가 내장된 경험식.

용도: ① 파도 공명 회피 판정의 기초 (파주기 ≈ T_roll이면 위험)
② GM 밴드의 물리적 의미 보완 (GM 크면 주기 짧음 = 뻣뻣·멀미).
"""
from __future__ import annotations

import math


def imo_roll_c(beam: float, draft: float, lwl: float) -> float:
    """IMO 계수 c — 규정식 그대로."""
    return 0.373 + 0.023 * (beam / draft) - 0.043 * (lwl / 100.0)


def roll_natural_period(beam: float, draft: float, lwl: float,
                        gm: float) -> float:
    """횡요 고유주기 [s] — IMO IS Code Weather Criterion 공식."""
    if gm <= 0:
        return float("inf")   # 복원력 없음 — 주기 정의 불가 (발산)
    return 2.0 * imo_roll_c(beam, draft, lwl) * beam / math.sqrt(gm)
