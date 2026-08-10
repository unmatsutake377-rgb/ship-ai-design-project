"""조종 프록시 — 정상 선회 닫힌 해법 (NSGA fast 게이트용).

원리: 전체 게이트의 시간 적분(RK4 수천 스텝) 대신 **같은 MMG
derivatives**의 정상상태 (u̇,v̇,ṙ)=0 3원 연립을 fsolve로 푼다
(함수 평가 수십 회). 새 물리·새 계수 0 — 프록시 오차원은 "과도
선회 대 정상 선회" 차이 하나라 보정 계수 1개로 흡수.

정상 선회 반경: R = U_s / r_s (U_s = √(u²+v²) 정상 속력),
선회지름 프록시 = 2R/L × TACTICAL_OVER_STEADY.

TACTICAL_OVER_STEADY 1.15: 추정 사슬 선박군 시뮬 대조 실측 —
합성 100 m Cb 0.47/0.55/0.60에서 비 1.13/1.18/1.20, 실전선
v3 슬렌더 비 1.16 (경계 근방 = 슬렌더 쪽이라 그쪽 정밀 우선).
통통 선형(L/B~3.7)은 비 ~1.5까지 커져 프록시가 **관대** — 통통
쪽 오탈락 없음 방향이라 fast 프록시로 안전 (최종 확정은 full
재검, 2단 구도). 캘리브레이션 전수 실측 (2026-08-11): v2 합격
97척 프록시 D_T 최대 3.12 (오탈락 0) vs v3 불합격 표본 21척
최소 6.41 (오생존 0) — 한계 5.0 양쪽 완전 분리.
IMO advance는 과도기 지배라 프록시 없음 — D_T가 v3 실측에서
더 크게 위반된 지배 항목이라 D_T만 관문 (정직 각주).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import fsolve

from src.physics.maneuvering.mmg import MMGShip, derivatives

TACTICAL_OVER_STEADY = 1.15     # 선박군 시뮬 대조 (모듈 docstring)


def steady_turning_state(ship: MMGShip, u0: float,
                         delta_deg: float = 35.0) -> dict:
    """타각 고정 정상 선회 (u, v, r) — derivatives 상위 3성분 = 0."""
    delta = math.radians(delta_deg)

    lpp = ship.par.lpp

    def resid(x):
        u, v, r = x
        # u≤0 특이점 가드 (백지 리뷰 지적): fsolve가 탐색 중
        # u=0을 밟으면 J_P=0 나눗셈 — 양의 u로 밀어내는 페널티.
        if u <= 1e-3:
            return np.array([1e3 * (1e-3 - u) + 1.0, v, r * lpp])
        s = np.array([u, v, r, 0.0, 0.0, 0.0])
        d = derivatives(ship, s, delta)[:3]
        # ṙ×Lpp: 잔차 3성분 단위 정합 (m/s²) — 대형선 조건수 개선
        return np.array([d[0], d[1], d[2] * lpp])

    guess = np.array([0.6 * u0, -0.15 * u0, 0.3 * u0 / lpp])
    sol, _info, ier, _msg = fsolve(resid, guess, full_output=True)
    u_s, v_s, r_s = (float(sol[0]), float(sol[1]), float(sol[2]))
    big_u = math.hypot(u_s, v_s)
    converged = bool(ier == 1 and u_s > 0.0 and abs(r_s) > 1e-9)
    return {
        "u": u_s, "v": v_s, "r": r_s,
        "speed": big_u,
        "radius_over_l": (big_u / (abs(r_s) * lpp)
                          if converged else float("inf")),
        "converged": converged,
    }


def turning_proxy(ship: MMGShip, u0: float) -> dict:
    """35° 정상 선회 → 선회지름 프록시 (L 배수) + IMO 5.0L 판정."""
    st = steady_turning_state(ship, u0, delta_deg=35.0)
    dt_proxy = 2.0 * st["radius_over_l"] * TACTICAL_OVER_STEADY
    return {
        "steady_radius_over_l": st["radius_over_l"],
        "tactical_diameter_proxy_over_l": float(dt_proxy),
        "converged": st["converged"],
        "passed": bool(st["converged"] and dt_proxy <= 5.0),
        "note": "정상 선회 프록시 — advance 미판정 (과도기 지배), "
                "full 재검이 IMO 전 항목 확정",
    }
