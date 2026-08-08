"""Frank close-fit — 임의 단면 2D heave 계수 (소스 분포 경계요소).

원전: Journée & Massie 7.3.4 (Frank 1967 계보) — 식 7.113~7.142
(구현 인덱스: 내항성 스펙 §9 보강). 좌표: x 폭방향, y 상방
(수면 y=0, 수중 y<0). 진동 점원 Green 함수 (복소형, ν = ω²/g):

  G(z,ζ) = (1/2π)[ln(z−ζ) − ln(z−ζ̄)] + (1/π)·PV∫0∞ e^{−ik(z−ζ̄)}/(ν−k) dk
           − i·e^{−iν(z−ζ̄)}          (ζ̄ = 자유표면 미러)

시간 규약: 물리량 = Re{ (·)·e^{−iωt} }. 운동 y = A e^{−iωt} (A=1).

이산화: **전체 윤곽** (수선 좌현 → 킬 → 수선 우현) 상수 소스
세그먼트 — 대칭 미러 조립 대신 전 윤곽 (버그 표면적 최소).
자기항 규약은 소스 유출량 항등식(∮∂G/∂n ds = 1)으로 고정.

검증 사다리 (격리 실험 문화): ① ln 소스 유출량 = 1 ② G 자유표면
조건 νG + ∂G/∂y = 0 ③ 반원에서 Ursell 대조 (최종 심판).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

RHO_SEAWATER = 1025.0
G_ACC = 9.81


def _pv_wave_integral(w: complex, nu: float) -> complex:
    """PV∫0∞ e^{−ikw}/(ν−k) dk — Im(w) < 0 (지수 감쇠)."""
    decay = -w.imag
    f_nu = np.exp(-1j * nu * w)

    def f(k: float) -> complex:
        return np.exp(-1j * k * w)

    def reg(k: float) -> complex:
        if abs(k - nu) < 1e-10 * max(nu, 1e-12):
            return 1j * w * f_nu
        return (f(k) - f_nu) / (nu - k)

    re1, _ = quad(lambda k: reg(k).real, 0.0, 2.0 * nu, limit=200)
    im1, _ = quad(lambda k: reg(k).imag, 0.0, 2.0 * nu, limit=200)
    upper = 2.0 * nu + 40.0 / max(decay, 1e-9)
    re2, _ = quad(lambda k: (f(k) / (nu - k)).real, 2.0 * nu, upper,
                  limit=200)
    im2, _ = quad(lambda k: (f(k) / (nu - k)).imag, 2.0 * nu, upper,
                  limit=200)
    return complex(re1 + re2, im1 + im2)


def _parts(z: complex, zeta: complex, nu: float):
    """원전 7.123 구조: G = Re{A} − i·Re{B} — A·B와 도함수 반환.

    A = (1/2π)[ln(z−ζ) − ln(w)] + (1/π)PV,  B = e^{−iνw} (w = z−ζ̄).
    y=0에서 Re{ln쌍} = 0 (켤레) — Re 구조가 자유표면 조건의 본질."""
    zeta_m = zeta.conjugate()
    w = z - zeta_m
    a_val = (np.log(z - zeta) - np.log(w)) / (2.0 * math.pi)         + _pv_wave_integral(w, nu) / math.pi
    pv = _pv_wave_integral(w, nu)
    full = 1.0 / (1j * w)
    a_der = (1.0 / (z - zeta) - 1.0 / w) / (2.0 * math.pi)         + (-1j) * (nu * pv - full) / math.pi
    b_val = np.exp(-1j * nu * w)
    b_der = -1j * nu * b_val
    return a_val, a_der, b_val, b_der


def green(z: complex, zeta: complex, nu: float) -> complex:
    """G = Re{A} − i·Re{B} (원전 7.123 그대로)."""
    a_val, _, b_val, _ = _parts(z, zeta, nu)
    return complex(a_val.real, -b_val.real)


def normal_derivative(z: complex, zeta: complex, nu: float,
                      n: complex) -> complex:
    """∂G/∂n — Re 구조 유지: ∂Re{A}/∂x = Re{A'}, ∂Re{A}/∂y = −Im{A'}."""
    _, a_der, _, b_der = _parts(z, zeta, nu)
    gx = complex(a_der.real, -b_der.real)
    gy = complex(-a_der.imag, b_der.imag)
    return n.real * gx + n.imag * gy


@dataclass(frozen=True)
class Frank2D:
    added_mass: float
    damping: float


def _contour(points):
    """윤곽 → (중점, 길이, 외향 법선). 순서: 수선(−B/2,0) → 킬 →
    수선(+B/2,0). 외향(물 쪽) 법선 = 접선×(+90° 회전: t·(+1j)) —
    이 순서에서 물은 윤곽의 왼쪽이 아니라 오른쪽 아래… 방향은
    반원 검증으로 고정: 원점 기준 바깥(원심) 방향이 외향."""
    pts = [complex(x, y) for x, y in points]
    n_seg = len(pts) - 1
    mids, lens, normals = [], [], []
    for i in range(n_seg):
        a, b = pts[i], pts[i + 1]
        mid = (a + b) / 2.0
        ln = abs(b - a)
        t = (b - a) / ln
        nrm = t * (-1j)               # 시계/반시계는 검증이 고정
        # 물 쪽(선체 바깥) 보정: 단면 내부 참조점(윤곽 도심 위쪽)에서
        # 멀어지는 방향이 외향
        ref = sum(pts) / len(pts)
        if ((mid - ref).real * nrm.real + (mid - ref).imag * nrm.imag) < 0:
            nrm = -nrm
        mids.append(mid)
        lens.append(ln)
        normals.append(nrm)
    return pts, mids, lens, normals


def heave_coefficients_frank(points, omega: float, gauss_n: int = 4,
                             rho: float = RHO_SEAWATER) -> Frank2D:
    """전체 윤곽 (수선 좌현 → 킬 → 수선 우현) → heave 2D 계수."""
    if gauss_n % 2 == 1:
        gauss_n += 1        # 홀수 절점 = 자기 세그먼트 중점 겹침 (0나눗셈)
    pts, mids, lens, normals = _contour(points)
    n_seg = len(mids)
    nu = omega * omega / G_ACC
    gx, gw = np.polynomial.legendre.leggauss(gauss_n)

    amat = np.zeros((n_seg, n_seg), dtype=complex)
    gmat = np.zeros((n_seg, n_seg), dtype=complex)
    for i in range(n_seg):
        zi, ni = mids[i], normals[i]
        for j in range(n_seg):
            a, b = pts[j], pts[j + 1]
            if i == j:
                # ln(z−ζ) 자기항: 법선미분 = +1/2 (외부 콜로케이션,
                # 유출량 항등식 시험이 규약 고정) · 포텐셜 = 해석값
                s = lens[j]
                a_self = 0.5
                g_self = s * (math.log(s / 2.0) - 1.0) / (2.0 * math.pi)
                # 나머지 항 (미러·파동 — 정칙): 가우스 적분, Re 구조
                d_reg = 0.0 + 0.0j
                g_reg = 0.0 + 0.0j
                for xg, wg in zip(gx, gw):
                    zeta = a + (b - a) * (0.5 + 0.5 * xg)
                    a_val, a_der, b_val, b_der = _parts(zi, zeta, nu)
                    # 자기 특이 ln(z−ζ) 성분 차감 (해석 처리했으므로)
                    sing = np.log(zi - zeta) / (2.0 * math.pi)
                    sing_d = 1.0 / (zi - zeta) / (2.0 * math.pi)
                    av = a_val - sing
                    ad = a_der - sing_d
                    gxc = complex(ad.real, -b_der.real)
                    gyc = complex(-ad.imag, b_der.imag)
                    d_reg += wg * (ni.real * gxc + ni.imag * gyc)
                    g_reg += wg * complex(av.real, -b_val.real)
                amat[i, j] = a_self + d_reg * lens[j] * 0.5
                gmat[i, j] = g_self + g_reg * lens[j] * 0.5
            else:
                dv = 0.0 + 0.0j
                gv = 0.0 + 0.0j
                for xg, wg in zip(gx, gw):
                    zeta = a + (b - a) * (0.5 + 0.5 * xg)
                    dv += wg * normal_derivative(zi, zeta, nu, ni)
                    gv += wg * green(zi, zeta, nu)
                amat[i, j] = dv * lens[j] * 0.5
                gmat[i, j] = gv * lens[j] * 0.5

    # heave 경계조건 (A=1): 몸체 속도 = d/dt e^{−iωt} ĵ = −iω ĵ
    # → vn = −iω·ny
    rhs = np.array([-1j * omega * normals[i].imag for i in range(n_seg)],
                   dtype=complex)
    q = np.linalg.solve(amat, rhs)

    phi = gmat @ q
    # 압력 p = −ρ ∂Φ/∂t = iωρ·φ (e^{−iωt} 계수)
    # 몸에 작용하는 힘: F_y = −∮ p·n_y ds (n = 물 쪽 외향 — 압력은
    # 몸을 안쪽으로 밈. 부호는 반원 Ursell 대조로 확정, 08-09)
    force = -sum(phi[i] * normals[i].imag * lens[i] for i in range(n_seg))
    fc = 1j * omega * rho * force
    # F = (M ω² + i ω N)·e^{−iωt}  (y = e^{−iωt}: −Mÿ−Nẏ)
    return Frank2D(added_mass=fc.real / omega ** 2,
                   damping=fc.imag / omega)
