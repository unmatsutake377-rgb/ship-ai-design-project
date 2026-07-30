"""저항 추정 (spec §2.3, M3.5).

마찰: ITTC-57 상관선 + 형상계수 (1+k).
조파: Michell 박선이론 적분.
Holtrop-Mennen 금지 — 20~300 m 상선 회귀식, USV 스케일 무효 (spec §2.3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh

from src.core.types import MainDimensions
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


def hull_offsets(mesh: trimesh.Trimesh, draft: float, n_x: int = 60,
                 n_z: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """메쉬에서 수선하 반폭표 추출: (xs, zs, y_half[n_x, n_z]).

    각 스테이션에서 선체를 종단면으로 잘라, 깊이별 최대 |y|를 반폭으로.
    임의 형상(Ship-D 포함)용 — 해석형 Wigley 가정 없음.
    """
    below = immersed_mesh(mesh, draft)
    x_min, x_max = float(below.bounds[0][0]), float(below.bounds[1][0])
    span = x_max - x_min
    xs = np.linspace(x_min + 1e-4 * span, x_max - 1e-4 * span, n_x)
    zs = np.linspace(0.0, draft, n_z)
    y_half = np.zeros((n_x, n_z))
    for i, x in enumerate(xs):
        section = below.section(plane_origin=[x, 0, 0],
                                plane_normal=[1, 0, 0])
        if section is None:
            continue
        for loop in section.discrete:
            pts = np.asarray(loop)
            # 스캔라인: 각 z에서 단면 다각형 변과의 교차 y를 보간해 최대 반폭.
            # (bin 방식은 격자를 좁히면 꼭짓점 사이 빈틈에 떨어져 0으로 샘 —
            #  해상도 올릴수록 오차가 커지던 버그의 원인이었음, 2026-07-27)
            z1, z2 = pts[:-1, 2], pts[1:, 2]
            y1, y2 = pts[:-1, 1], pts[1:, 1]
            dz = z2 - z1
            valid = np.abs(dz) > 1e-12
            for j, z in enumerate(zs):
                crossing = valid & (np.minimum(z1, z2) <= z) \
                    & (z <= np.maximum(z1, z2))
                if crossing.any():
                    t = (z - z1[crossing]) / dz[crossing]
                    y_at = y1[crossing] + t * (y2[crossing] - y1[crossing])
                    y_half[i, j] = max(y_half[i, j], float(np.abs(y_at).max()))
    return xs, zs, y_half


def michell_wave_resistance_mesh(mesh: trimesh.Trimesh, draft: float,
                                 speed: float, rho: float = RHO_SEAWATER,
                                 n_u: int = 120, n_x: int = 120,
                                 n_z: int = 60) -> float:
    """임의 메쉬용 Michell 조파저항 [N] — 해석형(Wigley 전용)의 일반화.

    반폭표를 수치로 뽑아 ∂y/∂x를 유한차분, 이중적분을 격자합으로.
    검증: 같은 Wigley 선체에서 해석형 michell_wave_resistance와 교차 대조
    (테스트 test_mesh_michell_matches_analytic_wigley).
    """
    xs, zs, y_half = hull_offsets(mesh, draft, n_x=n_x, n_z=n_z)
    dydx = np.gradient(y_half, xs, axis=0)          # (n_x, n_z)
    depth_below = draft - zs                          # (n_z,)

    k0 = G / speed ** 2
    us = np.linspace(1e-4, 4.0, n_u)
    lam = np.cosh(us)

    # z 적분: 각 (λ, x)에 대해 ∫ dydx·exp(−k0λ²d) dz → (n_u, n_x)
    expo = np.exp(-k0 * lam[:, None] ** 2 * depth_below[None, :])  # (n_u, n_z)
    zint = np.trapezoid(dydx[None, :, :] * expo[:, None, :], zs, axis=2)

    phase = k0 * lam[:, None] * xs[None, :]                        # (n_u, n_x)
    p = np.trapezoid(zint * np.cos(phase), xs, axis=1)
    q = np.trapezoid(zint * np.sin(phase), xs, axis=1)

    integrand = (p ** 2 + q ** 2) * lam ** 2
    return float(4.0 * rho * G ** 2 / (np.pi * speed ** 2)
                 * np.trapezoid(integrand, us))


@dataclass(frozen=True)
class ResistanceReport:
    speed: float
    froude: float
    reynolds: float
    wetted_area: float
    cf: float
    form_factor: float
    rf: float               # 마찰저항 [N]
    rw: float               # 조파저항 [N]
    total: float            # 전저항 = 소요 추력 [N]
    effective_power: float  # Pe = R·V [W]


def _assemble_report(speed: float, loa: float, s_wet: float, rf: float,
                     rw: float) -> ResistanceReport:
    total = rf + rw
    return ResistanceReport(
        speed=speed, froude=speed / math.sqrt(G * loa),
        reynolds=reynolds(speed, loa),
        wetted_area=s_wet, cf=ittc_cf(reynolds(speed, loa)),
        form_factor=FORM_FACTOR,
        rf=rf, rw=rw, total=total, effective_power=total * speed,
    )


def total_resistance(mesh: trimesh.Trimesh, dims: MainDimensions,
                     n: float, m: float, draft: float,
                     speed: float, rho: float = RHO_SEAWATER) -> ResistanceReport:
    """전저항 — Wigley 경로 (조파는 해석형 Michell)."""
    s_wet = wetted_surface(mesh, draft)
    rf = frictional_resistance(speed, dims.loa, s_wet, rho)
    rw = michell_wave_resistance(dims.loa, dims.beam, dims.draft_design,
                                 n, m, draft, speed, rho)
    return _assemble_report(speed, dims.loa, s_wet, rf, rw)


def total_resistance_mesh(mesh: trimesh.Trimesh, loa: float, draft: float,
                          speed: float,
                          rho: float = RHO_SEAWATER) -> ResistanceReport:
    """전저항 — 임의 메쉬 경로 (조파는 이중검증된 메쉬형 Michell).

    Ship-D 등 해석형 가정이 없는 형상용 (spec §7 2차)."""
    s_wet = wetted_surface(mesh, draft)
    rf = frictional_resistance(speed, loa, s_wet, rho)
    rw = michell_wave_resistance_mesh(mesh, draft, speed, rho)
    return _assemble_report(speed, loa, s_wet, rf, rw)


# 환기 트랜섬 기저저항 계수 (Phase C-1): 트랜섬 뒤 박리 사수역의 압력 결손.
# Hoerner 기저항력 계열 개략 상수 — 정량 문헌 벤치마크 없이 자릿수 정합
# 목적임을 명시 (반배수량 동적 부상·트림 미모델과 함께 Phase C 한계)
TRANSOM_BASE_DRAG_COEF = 0.10


def transom_drag(speed: float, transom_area: float,
                 rho: float = RHO_SEAWATER) -> float:
    """트랜섬 기저저항 [N] = ½·ρ·V²·A_t·C_bt."""
    return 0.5 * rho * speed ** 2 * transom_area * TRANSOM_BASE_DRAG_COEF


def total_resistance_semi(mesh: trimesh.Trimesh, loa: float, draft: float,
                          speed: float, transom_area: float,
                          rho: float = RHO_SEAWATER) -> ResistanceReport:
    """반배수량 전저항 (Phase C-1) = ITTC 마찰 + 메쉬 Michell + 트랜섬항.

    한계 (명시): 동적 부상·주행 트림 미모델 — 정적 흘수 근사.
    Michell은 Fn 0.4~1.0에서 근사 유효 (박선 가정 내)."""
    s_wet = wetted_surface(mesh, draft)
    rf = frictional_resistance(speed, loa, s_wet, rho)
    rw = michell_wave_resistance_mesh(mesh, draft, speed, rho) \
        + transom_drag(speed, transom_area, rho)
    return _assemble_report(speed, loa, s_wet, rf, rw)
