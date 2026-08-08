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


# 용도별 내항 운용 한계 (유의 진폭) — NORDFORSK 1987 일반 운용 한계
# 계보의 개략 (C급 — 정확 표값 확보 시 승급): 계측(survey)은 엄격,
# 작업·순찰·화물은 일반 한계.
SEAKEEPING_LIMITS = {
    "survey":   {"roll_deg": 4.0, "pitch_deg": 2.5, "heave_over_hs": 0.8},
    "patrol":   {"roll_deg": 6.0, "pitch_deg": 3.0, "heave_over_hs": 1.0},
    "workboat": {"roll_deg": 6.0, "pitch_deg": 3.0, "heave_over_hs": 1.0},
    "cargo":    {"roll_deg": 6.0, "pitch_deg": 3.0, "heave_over_hs": 1.0},
}

# 용도별 설계 해상 상태 (Hs[m], Tz[s]) — 운용 환경 개략 (C급):
# 소형 USV = 연안 (SS2~3), cargo = 외항 (SS5 근방)
DESIGN_SEA_STATE = {
    "survey": (0.5, 3.0), "patrol": (0.75, 3.5),
    "workboat": (0.5, 3.0), "cargo": (3.5, 8.5),
}


def seakeeping_gate(mesh, draft: float, mass: float, iyy: float,
                    beam: float, lwl: float, gm: float,
                    purpose: str) -> dict:
    """5번째 게이트 — 설계 해상에서 유의 응답 vs 용도 한계 합불.

    반환: 성적 + 항목별 판정 + passed. 기준·해상 상태는 C급 개략
    명시 (수집·문헌 확보로 승급 예정)."""
    from src.physics.seakeeping.waves import seakeeping_report

    hs, tz = DESIGN_SEA_STATE.get(purpose, DESIGN_SEA_STATE["survey"])
    lim = SEAKEEPING_LIMITS.get(purpose, SEAKEEPING_LIMITS["survey"])
    rep = seakeeping_report(mesh, draft, mass, iyy, beam, lwl, gm,
                            hs, tz, n_freq=5)
    checks = {
        "roll": rep["sig_roll_deg"] <= lim["roll_deg"],
        "pitch": rep["sig_pitch_deg"] <= lim["pitch_deg"],
        "heave": rep["sig_heave_m"] <= lim["heave_over_hs"] * hs,
    }
    return {**rep, "limits": lim, "checks": {k: bool(v) for k, v
                                             in checks.items()},
            "passed": bool(all(checks.values())),
            "grade_note": "기준·해상 상태 = NORDFORSK 계보 개략 (C급)"}
