"""파랑 스펙트럼 + 불규칙파 유의 응답 (내항성 3단계, 스펙 §4).

ITTC/ISSC (Bretschneider) 2-파라미터 스펙트럼 [표준 공식, B급]:
  S(ω) = 172.75·Hs²/(Tz⁴·ω⁵) · exp(−691/(Tz⁴·ω⁴))   [m²·s]

자기 검증 항등식 (시험 앵커): m0 = ∫S dω = Hs²/16 — 유의파고
정의와의 정합 (Hs = 4√m0)이 공식 계수의 심판.

유의 응답: m0_r = ∫ RAO²(ω)·S(ω) dω → 유의 진폭 = 2·√m0_r
(선형계 — 응답도 파고에 비례).

roll: 1자유도 공진 모델 — 파면 경사 기진, 관성·복원은 IMO 고유주기
(criteria.py 재사용), 감쇠비 κ = 점성 지배 (Ikeda 정식은 후속 —
빌지킬 없는 나선체 통상 0.03~0.08, 기본 0.06 C급 표기).
  RAO_roll(ω) = 1/√[(1−r²)² + (2κr)²],  r = ω/ωn
공진 해석값 (앵커): RAO(ωn) = 1/(2κ).
"""
from __future__ import annotations

import math

import numpy as np

ROLL_DAMPING_RATIO_DEFAULT = 0.06   # 점성 지배 통상 (C급 — Ikeda 후속)


def ittc_spectrum(omega: float, hs: float, tz: float) -> float:
    """ITTC/ISSC 스펙트럼 S(ω) [m²·s]."""
    if omega <= 0.0:
        return 0.0
    return (172.75 * hs * hs / (tz ** 4 * omega ** 5)
            * math.exp(-691.0 / (tz ** 4 * omega ** 4)))


def spectral_moment(hs: float, tz: float, order: int = 0,
                    rao=None, n: int = 400) -> float:
    """mₙ = ∫ ωⁿ·RAO²(ω)·S(ω) dω (RAO 생략 = 파 자체)."""
    oms = np.linspace(0.15, 6.0, n)
    s = np.array([ittc_spectrum(o, hs, tz) for o in oms])
    if rao is not None:
        s = s * np.array([rao(o) for o in oms]) ** 2
    return float(np.trapezoid(oms ** order * s, oms))


def significant_amplitude(hs: float, tz: float, rao) -> float:
    """유의 응답 진폭 = 2√m0 (RAO 함수 주입)."""
    return 2.0 * math.sqrt(spectral_moment(hs, tz, 0, rao))


def roll_rao(omega: float, roll_period_s: float,
             kappa: float = ROLL_DAMPING_RATIO_DEFAULT) -> float:
    """roll RAO (파면 경사 대비, 횡파 1자유도 공진 모델)."""
    wn = 2.0 * math.pi / roll_period_s
    r = omega / wn
    return 1.0 / math.sqrt((1.0 - r * r) ** 2 + (2.0 * kappa * r) ** 2)


def significant_roll_deg(hs: float, tz: float, roll_period_s: float,
                         kappa: float = ROLL_DAMPING_RATIO_DEFAULT
                         ) -> float:
    """유의 roll 진폭 [deg] — 파면 경사 스펙트럼 × roll RAO.

    파면 경사 진폭 = k·ζa → 경사 스펙트럼 = (ω²/g)²·S(ω)."""
    def slope_response(o):
        return (o * o / 9.81) * roll_rao(o, roll_period_s, kappa)

    m0 = spectral_moment(hs, tz, 0, slope_response)
    return math.degrees(2.0 * math.sqrt(m0))


def seakeeping_report(mesh, draft: float, mass: float, iyy: float,
                      beam: float, lwl: float, gm: float,
                      hs: float, tz: float,
                      n_freq: int = 7) -> dict:
    """내항 성적표 — 지정 해상 (Hs, Tz)에서 유의 응답 3종.

    heave·pitch: 스트립 RAO (성긴 주파수 격자 + 보간 — 스트립 계산
    비용 절충, 격자는 스펙트럼 에너지 대역을 덮음).
    roll: IMO 고유주기 + 공진 모델 (횡파 보수 — 선수파 항해 시 실제
    roll은 이보다 작음, 정직 표기)."""
    from src.physics.seakeeping.criteria import roll_natural_period
    from src.physics.seakeeping.strip import heave_pitch_rao

    wm = 2.0 * math.pi / tz                    # 평균 주파수 근방
    oms = np.linspace(0.45 * wm, 2.2 * wm, n_freq)
    raos = heave_pitch_rao(mesh, draft, mass, iyy, list(oms),
                           n_stations=9, contour_n=10)
    h_grid = np.array([r.heave_rao for r in raos])
    p_grid = np.array([r.pitch_rao for r in raos])

    def h_rao(o):
        return float(np.interp(o, oms, h_grid, left=h_grid[0], right=0.0))

    def p_rao(o):
        # pitch RAO는 파면 경사 대비 → 절대 각도 = RAO·k·ζa
        k = o * o / 9.81
        return float(np.interp(o, oms, p_grid,
                               left=p_grid[0], right=0.0)) * k

    t_roll = roll_natural_period(beam, draft, lwl, gm)
    return {
        "sea_state": {"hs_m": hs, "tz_s": tz},
        "sig_heave_m": significant_amplitude(hs, tz, h_rao),
        "sig_pitch_deg": math.degrees(
            significant_amplitude(hs, tz, p_rao)),
        "sig_roll_deg": significant_roll_deg(hs, tz, t_roll),
        "roll_period_s": t_roll,
        "note": "roll = 횡파 보수 (공진 모델·κ 0.06 C급 — Ikeda 후속)"
                "; heave·pitch = 선수파 스트립 (V=0)",
    }
