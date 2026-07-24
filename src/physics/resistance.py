"""저항 추정 (spec §2.3, M3.5).

마찰: ITTC-57 상관선 + 형상계수 (1+k).
조파: Michell 박선이론 적분.
Holtrop-Mennen 금지 — 20~300 m 상선 회귀식, USV 스케일 무효 (spec §2.3).
"""
from __future__ import annotations

import math

import numpy as np
import trimesh

from src.physics.hydrostatics import immersed_mesh, waterplane_properties

RHO_SEAWATER = 1025.0
NU_SEAWATER = 1.19e-6   # 해수 15°C 동점성계수 [m²/s]
G = 9.81
FORM_FACTOR = 0.10      # 형상계수 k (세장 선형 개략, 점성압력저항 보정)


def reynolds(speed: float, length: float, nu: float = NU_SEAWATER) -> float:
    return speed * length / nu


def ittc_cf(re: float) -> float:
    """ITTC-57 모형선-실선 상관선: Cf = 0.075 / (log10(Re) - 2)²."""
    return 0.075 / (math.log10(re) - 2.0) ** 2


def wetted_surface(mesh: trimesh.Trimesh, draft: float) -> float:
    """침수 표면적. slice가 만든 수선면 캡(들)은 물에 닿지 않으므로 제외."""
    below = immersed_mesh(mesh, draft)
    aw, _ = waterplane_properties(mesh, draft)
    return float(below.area) - aw


def frictional_resistance(speed: float, length: float, wetted_area: float,
                          rho: float = RHO_SEAWATER,
                          form_factor: float = FORM_FACTOR) -> float:
    """Rf = ½·ρ·V²·S·Cf·(1+k)."""
    cf = ittc_cf(reynolds(speed, length))
    return 0.5 * rho * speed ** 2 * wetted_area * cf * (1.0 + form_factor)


def michell_wave_resistance(loa: float, beam: float, draft_design: float,
                            n: float, m: float, draft: float, speed: float,
                            rho: float = RHO_SEAWATER,
                            n_u: int = 120, n_x: int = 160,
                            n_z: int = 80) -> float:
    """Michell 박선이론 조파저항 [N].

    Rw = (4ρg²)/(πV²) ∫₁^∞ (P²+Q²) λ²/√(λ²−1) dλ
    P+iQ = ∫∫ (∂y/∂x)·exp(−k₀λ²·d)·exp(ik₀λx) dx dz  (d: 수면하 깊이)
    λ = cosh(u) 치환: dλ/√(λ²−1) = du → 특이점 제거.
    일반화 Wigley 분리형 y=(B/2)·f(x)·h(z) → x/z 적분 분리.
    draft: 실제(평형) 흘수 — 침수 부분만 적분.
    """
    k0 = G / speed ** 2

    # x 방향: f'(x) = -(2/L)·n·|u|^(n-1)·sign(u), 반폭 계수 B/2 포함
    xs = np.linspace(-loa / 2, loa / 2, n_x)
    ux = 2.0 * xs / loa
    fprime = -(beam / 2.0) * (2.0 / loa) * n * np.abs(ux) ** (n - 1) * np.sign(ux)

    # z 방향: h(z) = 1 − ((T_d−z)/T_d)^m, 수면하 깊이 d = draft − z
    zs = np.linspace(0.0, draft, n_z)
    h = 1.0 - ((draft_design - zs) / draft_design) ** m
    d_below = draft - zs

    us = np.linspace(1e-4, 4.0, n_u)  # λ = cosh(u) ∈ [1, ~27]
    lam = np.cosh(us)

    z_int = np.trapezoid(
        h[None, :] * np.exp(-k0 * lam[:, None] ** 2 * d_below[None, :]),
        zs, axis=1,
    )
    phase = k0 * lam[:, None] * xs[None, :]
    p = np.trapezoid(fprime[None, :] * np.cos(phase), xs, axis=1)
    q = np.trapezoid(fprime[None, :] * np.sin(phase), xs, axis=1)

    integrand = (p ** 2 + q ** 2) * z_int ** 2 * lam ** 2
    return float(4.0 * rho * G ** 2 / (np.pi * speed ** 2)
                 * np.trapezoid(integrand, us))
