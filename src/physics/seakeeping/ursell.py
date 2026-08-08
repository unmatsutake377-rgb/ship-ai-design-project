"""Ursell (1949) 반원 heave 2D 계수 — 주파수 의존 부가질량·조파감쇠.

원전: Journée & Massie, Offshore Hydromechanics (TU Delft OCW),
7장 §7.3.1 — 식 7.55~7.82 (구현 인덱스: 내항성 스펙 §8).
references/OffshoreHydromechanics_Journee_Massie.pdf 로컬 보존본.

정식화 (heave, 반경 r0, 좌표: y 아래 양·θ는 y축부터):
- 표면 스트림함수 (α=0):  ψA'_2m(θ) = sin(2mθ) + ξr·sin((2m−1)θ)/(2m−1)
  ψB0c = π e^{−ky} sin(kx),  ψB0s = −π e^{−ky} cos(kx) + H(x,y)
  H = ∫0∞ [k cos(νy) + ν sin(νy)]/(ν²+k²)·e^{−νx} dν   (x ≥ 0)
- 경계조건 (식 7.66): Ψ0 = −ẏ·r0 sinθ + C(t) → 시간 성분 분해:
  ψB0c + Σ P2m ψA'_2m = a·sinθ + c1
  ψB0s + Σ Q2m ψA'_2m = b·sinθ + c2      (collocation 최소자승)
  a = πω²r0·(ya/ηa)·sinδ/g,  b = 같은 꼴 cosδ — 진폭비·위상 동시 산출
- 압력 (7.79) → 하중 적분 (7.82) → M33'·N33' 분해.

검증 앵커: ① 고주파 극한 M33' → ρπr0²/2 (Lewis 해석값과 접점)
② N33' ≥ 0 ③ 에너지 항등식 N33' = ρg²(ηa/ya)²/ω³ (방사 감쇠 정확
관계 — 급수 결과의 내부 일관성 심판).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

RHO_SEAWATER = 1025.0
G = 9.81


@dataclass(frozen=True)
class Ursell2D:
    xi_r: float          # 무차원 주파수² = ω²r0/g
    added_mass: float    # M33' [kg/m]
    damping: float       # N33' [N·s/m per m]
    amp_ratio: float     # ηa/ya (방사파/운동 진폭비)
    phase: float         # δ [rad]


def _h_integral(x: float, y: float, k: float) -> float:
    """H(x,y) = ∫0∞ [k cos(νy)+ν sin(νy)]/(ν²+k²)·e^{−νx} dν."""
    def f(v):
        return (k * math.cos(v * y) + v * math.sin(v * y)) \
            / (v * v + k * k) * math.exp(-v * x)
    # x>0이면 지수 감쇠 — 분할 적분으로 안정
    if x > 1e-9:
        val, _ = quad(f, 0.0, 50.0 * k + 50.0 / max(x, 1e-3), limit=200)
        return val
    # x=0: 진동 수렴 — 두 항 분해 (weight 적분)
    v1, _ = quad(lambda v: k / (v * v + k * k), 0.0, np.inf,
                 weight="cos", wvar=y, limit=200)
    v2, _ = quad(lambda v: v / (v * v + k * k), 0.0, np.inf,
                 weight="sin", wvar=y, limit=200)
    return v1 + v2


def _g_integral(x: float, y: float, k: float) -> float:
    """G(x,y) = ∫0∞ [k sin(νy) − ν cos(νy)]/(ν²+k²)·e^{−νx} dν (φBs)."""
    def f(v):
        return (k * math.sin(v * y) - v * math.cos(v * y)) \
            / (v * v + k * k) * math.exp(-v * x)
    if x > 1e-9:
        val, _ = quad(f, 0.0, 50.0 * k + 50.0 / max(x, 1e-3), limit=200)
        return val
    v1, _ = quad(lambda v: k / (v * v + k * k), 0.0, np.inf,
                 weight="sin", wvar=y, limit=200)
    v2, _ = quad(lambda v: v / (v * v + k * k), 0.0, np.inf,
                 weight="cos", wvar=y, limit=200)
    return v1 - v2


def heave_coefficients(r0: float, omega: float, n_terms: int = 8,
                       n_colloc: int = 24,
                       rho: float = RHO_SEAWATER) -> Ursell2D:
    """반원 heave 주파수 의존 계수 (Ursell 급수 + collocation)."""
    k = omega * omega / G
    xi_r = k * r0

    thetas = np.linspace(0.03, math.pi / 2.0, n_colloc)

    def psi_a(m, th):
        return math.sin(2 * m * th) + xi_r * math.sin((2 * m - 1) * th) \
            / (2 * m - 1)

    def phi_a(m, th):
        return math.cos(2 * m * th) + xi_r * math.cos((2 * m - 1) * th) \
            / (2 * m - 1)

    # 표면 좌표 (α=0): x = r0 sinθ, y = r0 cosθ
    xs = r0 * np.sin(thetas)
    ys = r0 * np.cos(thetas)
    psi_bc = np.array([math.pi * math.exp(-k * y) * math.sin(k * x)
                       for x, y in zip(xs, ys)])
    psi_bs = np.array([-math.pi * math.exp(-k * y) * math.cos(k * x)
                       + _h_integral(x, y, k)
                       for x, y in zip(xs, ys)])

    # 최소자승: [ψA'_2m ... , −sinθ, −1]·[P.., a, c1] = −ψB0c (이항)
    ncols = n_terms + 2
    amat = np.zeros((n_colloc, ncols))
    for j in range(n_terms):
        amat[:, j] = [psi_a(j + 1, th) for th in thetas]
    amat[:, n_terms] = -np.sin(thetas)
    amat[:, n_terms + 1] = -1.0
    sol_c, *_ = np.linalg.lstsq(amat, -psi_bc, rcond=None)
    sol_s, *_ = np.linalg.lstsq(amat, -psi_bs, rcond=None)
    p2m, a_coef = sol_c[:n_terms], sol_c[n_terms]
    q2m, b_coef = sol_s[:n_terms], sol_s[n_terms]

    # 진폭비·위상: a = πω²r0(ya/ηa)sinδ/g, b = 같은 꼴 cosδ
    scale = math.pi * omega * omega * r0 / G
    ya_over_eta = math.hypot(a_coef, b_coef) / scale
    if ya_over_eta < 1e-12:
        return Ursell2D(xi_r, float("nan"), float("nan"),
                        float("nan"), float("nan"))
    delta = math.atan2(a_coef, b_coef)
    amp_ratio = 1.0 / ya_over_eta          # ηa/ya

    # 하중 적분: I = ∫0^{π/2} φ'(θ) cosθ dθ (전 급수 합성)
    def phi_c_total(th):
        x, y = r0 * math.sin(th), r0 * math.cos(th)
        return math.pi * math.exp(-k * y) * math.cos(k * x) \
            + sum(p2m[j] * phi_a(j + 1, th) for j in range(n_terms))

    def phi_s_total(th):
        x, y = r0 * math.sin(th), r0 * math.cos(th)
        return math.pi * math.exp(-k * y) * math.sin(k * abs(x)) \
            + _g_integral(abs(x), y, k) \
            + sum(q2m[j] * phi_a(j + 1, th) for j in range(n_terms))

    i_p, _ = quad(lambda th: phi_c_total(th) * math.cos(th),
                  0.0, math.pi / 2.0, limit=100)
    i_q, _ = quad(lambda th: phi_s_total(th) * math.cos(th),
                  0.0, math.pi / 2.0, limit=100)

    # Fy' = −∫p dx0 성분 → cos·sin 계수 (ya=1 정규화, ηa = amp_ratio)
    eta_a = amp_ratio                       # ya = 1 기준
    fc = (2.0 * r0 * rho * G * eta_a / math.pi) * i_q
    fs = -(2.0 * r0 * rho * G * eta_a / math.pi) * i_p
    m33 = (fc * math.cos(delta) - fs * math.sin(delta)) / (omega * omega)
    n33 = (fc * math.sin(delta) + fs * math.cos(delta)) / omega
    return Ursell2D(xi_r=xi_r, added_mass=m33, damping=n33,
                    amp_ratio=amp_ratio, phase=delta)
