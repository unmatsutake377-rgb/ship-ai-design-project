"""MMG 3자유도 조종 모델 (조종성 1단계, 스펙 2026-08-09 §2).

원전: Yasukawa & Yoshimura (2015) — 힘 = 선체(H) + 프로펠러(P) +
러더(R) 분해, 미드십 원점·xG 오프셋 운동방정식 (Eq 4, p3 원전
대조). 무차원: 힘 ½ρLdU², 모멘트 ½ρL²dU² (v'=v_m/U, r'=rL/U).

계수 주입형 — 배(KVLCC2·우리 배)와 모델이 분리. 2단계 estimation
이 같은 인터페이스로 우리 배 계수를 만든다.

정직 표기: 직진 저항 R0(u) = r0_n·(u/u0)² 근사 (원전은 Schoenherr
곡선 — 접근 속도 근방에서 u² 스케일 충분). βP의 x'P = −0.48
(프로펠러 위치 관례값 — 원전 Eq 15는 정의만, 수치 명기 없음).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.physics.maneuvering.kvlcc2 import MMGCoeffs, ShipParticulars

X_P_PRIME = -0.48      # βP = β − x'P·r' 프로펠러 위치 (관례값, C급)


@dataclass(frozen=True)
class MMGShip:
    par: ShipParticulars
    co: MMGCoeffs
    r0_n: float            # 직진 저항 at 접근 속도 u0 [N]
    n_p: float             # 프로펠러 회전수 [1/s] (자항 평형)
    u0: float              # 접근 속도 [m/s] (R0 정의 속도)

    @property
    def mass(self) -> float:
        return self.par.rho * self.par.displacement_m3

    @property
    def izz(self) -> float:
        return self.mass * (0.25 * self.par.lpp) ** 2


def _kt(co: MMGCoeffs, j_p: float) -> float:
    return co.k0 + co.k1 * j_p + co.k2 * j_p * j_p


def solve_self_propulsion(par: ShipParticulars, co: MMGCoeffs,
                          u0: float, r0_n: float) -> float:
    """직진 평형 nP — (1−tP)·ρnP²DP⁴·KT(JP) = R0 이분법."""
    def net(n_p: float) -> float:
        j_p = u0 * (1.0 - co.w_p0) / max(n_p * par.dp, 1e-9)
        t = par.rho * n_p ** 2 * par.dp ** 4 * _kt(co, j_p)
        return (1.0 - co.t_p) * t - r0_n
    lo, hi = 0.05, 100.0
    if net(hi) < 0.0:
        raise ValueError(
            "자항 평형 브래킷 밖 — 저항이 최대 추력 초과 "
            f"(nP {hi} rps)")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if net(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _wake_and_beta(par, co, u, v_m, r) -> tuple[float, float]:
    """조종 중 유효 반류 wP (원전 Eq 16) + 사항각 β."""
    big_u = max(math.hypot(u, v_m), 1e-9)
    beta = math.atan2(-v_m, max(u, 1e-9))
    rp = r * par.lpp / big_u
    beta_p = beta - X_P_PRIME * rp
    c2 = co.c2_plus if beta_p >= 0.0 else co.c2_minus
    ratio = 1.0 + (1.0 - math.exp(-co.c1 * abs(beta_p))) * (c2 - 1.0)
    w_p = 1.0 - ratio * (1.0 - co.w_p0)
    return w_p, beta


def _hull_forces(par, co, u, v_m, r) -> tuple[float, float, float]:
    """선체 유체력 (원전 Eq 6~7) — R0 제외 (호출측 합산)."""
    big_u = max(math.hypot(u, v_m), 1e-9)
    vp = v_m / big_u
    rp = r * par.lpp / big_u
    q = 0.5 * par.rho * par.lpp * par.draft * big_u ** 2
    xh = q * (co.xvv * vp ** 2 + co.xvr * vp * rp + co.xrr * rp ** 2
              + co.xvvvv * vp ** 4)
    yh = q * (co.yv * vp + co.yr * rp + co.yvvv * vp ** 3
              + co.yvvr * vp ** 2 * rp + co.yvrr * vp * rp ** 2
              + co.yrrr * rp ** 3)
    nh = q * par.lpp * (co.nv * vp + co.nr * rp + co.nvvv * vp ** 3
                        + co.nvvr * vp ** 2 * rp
                        + co.nvrr * vp * rp ** 2 + co.nrrr * rp ** 3)
    return xh, yh, nh


def _rudder_forces(ship: MMGShip, u, v_m, r,
                   delta) -> tuple[float, float, float]:
    """러더 힘 (원전 Eq 18~25) — 프로펠러 후류 증속 + 상호작용."""
    par, co = ship.par, ship.co
    big_u = max(math.hypot(u, v_m), 1e-9)
    w_p, beta = _wake_and_beta(par, co, u, v_m, r)
    rp = r * par.lpp / big_u
    j_p = u * (1.0 - w_p) / max(ship.n_p * par.dp, 1e-9)
    kt = max(_kt(co, j_p), 0.0)
    eta = par.dp / par.hr
    inner = 1.0 + co.kappa * (math.sqrt(1.0 + 8.0 * kt
                                        / (math.pi * j_p ** 2)) - 1.0)
    u_r = (co.eps * u * (1.0 - w_p)
           * math.sqrt(eta * inner ** 2 + (1.0 - eta)))
    beta_r = beta - co.ell_r_p * rp
    gamma = co.gamma_r_plus if beta_r >= 0.0 else co.gamma_r_minus
    v_r = big_u * gamma * beta_r
    alpha_r = delta - math.atan2(v_r, max(u_r, 1e-9))
    f_n = (0.5 * par.rho * par.ar * (u_r ** 2 + v_r ** 2)
           * co.f_alpha * math.sin(alpha_r))
    x_r = -(1.0 - co.t_r) * f_n * math.sin(delta)
    y_r = -(1.0 + co.a_h) * f_n * math.cos(delta)
    n_r = -(co.x_r_p + co.a_h * co.x_h_p) * par.lpp \
        * f_n * math.cos(delta)
    return x_r, y_r, n_r


def derivatives(ship: MMGShip, state: np.ndarray,
                delta: float) -> np.ndarray:
    """상태 미분 — state = [u, v_m, r, x0, y0, psi] (원전 Eq 4)."""
    par, co = ship.par, ship.co
    u, v_m, r, _x0, _y0, psi = state
    m = ship.mass
    md = 0.5 * par.rho * par.lpp ** 2 * par.draft
    mx = co.mx_p * md
    my = co.my_p * md
    jz = co.jz_p * 0.5 * par.rho * par.lpp ** 4 * par.draft

    xh, yh, nh = _hull_forces(par, co, u, v_m, r)
    w_p, _ = _wake_and_beta(par, co, u, v_m, r)
    j_p = u * (1.0 - w_p) / max(ship.n_p * par.dp, 1e-9)
    x_p = (1.0 - co.t_p) * par.rho * ship.n_p ** 2 * par.dp ** 4 \
        * max(_kt(co, j_p), 0.0)
    r0 = ship.r0_n * (u / ship.u0) ** 2
    x_r, y_r, n_r = _rudder_forces(ship, u, v_m, r, delta)

    x = xh - r0 + x_p + x_r
    y = yh + y_r
    n = nh + n_r                      # 미드십 기준 (원전 관례)

    xg = par.xg
    a11 = m + my
    a12 = xg * m
    a21 = xg * m
    a22 = ship.izz + xg ** 2 * m + jz
    b1 = y - (m + mx) * u * r
    b2 = n - xg * m * u * r
    det = a11 * a22 - a12 * a21
    v_dot = (b1 * a22 - b2 * a12) / det
    r_dot = (b2 * a11 - b1 * a21) / det
    u_dot = (x + (m + my) * v_m * r + xg * m * r ** 2) / (m + mx)

    x0_dot = u * math.cos(psi) - v_m * math.sin(psi)
    y0_dot = u * math.sin(psi) + v_m * math.cos(psi)
    return np.array([u_dot, v_dot, r_dot, x0_dot, y0_dot, r])
