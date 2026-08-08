"""스트립 이론 — heave·pitch 연성 RAO (내항성 2단계, 스펙 §3).

원전: Journée & Massie 8.3 (Ordinary Strip Theory — 식 8.21~8.46,
Modified 박스 항 제외). 선수파 (μ=180°), 심수.

사슬: 메쉬 → 스테이션 (sections) → 단면 2D 계수 (Frank) →
길이 적분 (8.24~8.25) → 기진력 (FK 유효진폭 C3, 8.41) →
2×2 복소 운동방정식 → RAO.

검증 앵커 (해석 극한): ① 장파 kL→0에서 heave RAO→1·pitch
RAO/k→1 (배가 파면을 탐) ② 단파 RAO→0 ③ 공진 대역 존재.
Wigley 문헌 실측 대조는 데이터 확보 후 (백로그).

V(전진속도) 항: 식 그대로 스테이션 차분(dM'/dx)으로 수치 조립 —
V=0이면 두 스트립 이론 동치 (원전 8-14 명시).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

RHO_SEAWATER = 1025.0
G_ACC = 9.81


@dataclass(frozen=True)
class StripRAO:
    omega: float          # 파 주파수 (=조우 주파수, V=0) [rad/s]
    heave_rao: float      # |ẑ|/ζa
    pitch_rao: float      # |θ̂|/(k·ζa)  (파면 기울기 무차원)
    heave_phase: float
    pitch_phase: float


def _station_contour(sec, n: int = 16):
    """LewisSection → Frank 입력 윤곽 (수선 좌현→킬→우수선).

    Lewis 사상 곡선 재사용 — 물리·형상 동일 소스."""
    from src.physics.seakeeping.lewis import section_points

    pts = section_points(sec, n=n)          # (y≥0, z: 킬0→수선T)
    # Frank 좌표 (x 폭, y 상방·수면0): y_f = z − T
    right = [(y, z - sec.draft) for y, z in pts]      # 킬→우수선
    left = [(-y, z) for y, z in reversed(right)]      # 좌수선→킬
    return left + right[1:]


def heave_pitch_rao(mesh, draft: float, mass: float, iyy: float,
                    omegas, speed: float = 0.0,
                    n_stations: int = 11, contour_n: int = 12,
                    rho: float = RHO_SEAWATER) -> list[StripRAO]:
    """heave·pitch RAO(ω) — 선수파, G≈수선 중앙 가정 (대칭 선형)."""
    from src.physics.seakeeping.frank import heave_coefficients_frank
    from src.physics.seakeeping.sections import extract_stations

    stations = extract_stations(mesh, draft, n_stations=n_stations)
    if len(stations) < 5:
        raise ValueError("유효 스테이션 부족 — 흘수·메쉬 확인")
    xmid = 0.5 * (mesh.bounds[0][0] + mesh.bounds[1][0])
    xs = np.array([x - xmid for x, _ in stations])     # G 기준
    secs = [s for _, s in stations]
    yw = np.array([s.beam / 2.0 for s in secs])        # 수선 반폭

    out = []
    for omega in omegas:
        k = omega * omega / G_ACC
        we = omega + k * speed              # 선수파 조우 주파수
        # 단면 2D 계수 (조우 주파수에서)
        m2d = np.zeros(len(secs))
        n2d = np.zeros(len(secs))
        for i, s in enumerate(secs):
            f = heave_coefficients_frank(_station_contour(s, contour_n),
                                         we, gauss_n=2)   # 짝수 — 자기 세그먼트 중점 겹침 방지
            m2d[i], n2d[i] = f.added_mass, f.damping
        dmdx = np.gradient(m2d, xs)
        dndx = np.gradient(n2d, xs)

        def tr(arr):
            return float(np.trapezoid(arr, xs))

        v = speed
        a33 = tr(m2d)
        b33 = tr(n2d - v * dmdx)
        c33 = 2.0 * rho * G_ACC * tr(yw)
        a35 = -tr(m2d * xs) - v / we ** 2 * tr(n2d - v * dmdx)
        b35 = -tr((n2d - v * dmdx) * xs) + 2.0 * v * tr(m2d)
        c35 = -2.0 * rho * G_ACC * tr(yw * xs)
        a53 = -tr(m2d * xs)
        b53 = -tr((n2d - v * dmdx) * xs)
        c53 = -2.0 * rho * G_ACC * tr(yw * xs)
        a55 = tr(m2d * xs ** 2) + v / we ** 2 * tr((n2d - v * dmdx) * xs)
        b55 = tr((n2d - v * dmdx) * xs ** 2)
        c55 = 2.0 * rho * G_ACC * tr(yw * xs ** 2)

        # 기진력 (선수파 μ=180°: 위상 e^{+ikx}) — FK 유효진폭 C3
        # (8.41 심수): C3 = 1 − (k/yw)∫e^{kz}·y(z)dz ≈ e^{−kT*}
        # Lewis 단면 적분 (사상 곡선)
        from src.physics.seakeeping.lewis import section_points
        x3 = np.zeros(len(secs), dtype=complex)
        for i, s in enumerate(secs):
            pts = section_points(s, n=30)
            zs = np.array([p[1] - s.draft for p in pts])   # 0(수선)→−T(킬)
            ys = np.array([p[0] for p in pts])
            integ = float(np.trapezoid(ys * np.exp(k * zs), zs))
            # zs 킬→수선 순서 주의: section_points는 킬(−T)→수선(0)
            c3 = 1.0 - (k / max(yw[i], 1e-9)) * abs(integ)
            c3 = max(c3, 0.0)
            zeta = c3 * np.exp(1j * k * xs[i])       # ζ*(x) 복소 진폭
            # ζ̈* = −kg·ζ*, ζ̇* = d/dt → −iωe·(−kg/ωe²)? 원전 8.45:
            # ζ̈ = −kg ζ* cosθ, ζ̇ = −(kg/ωe) ζ* sinθ, θ = ωet+kx
            # 복소 (cosθ = Re e^{iθ}): ζ̈̂ = −kg ζ̂, ζ̇̂ = i(kg/ωe) ζ̂
            acc = -k * G_ACC * zeta
            vel = 1j * (k * G_ACC / we) * zeta
            x3[i] = m2d[i] * acc + n2d[i] * vel \
                + 2.0 * rho * G_ACC * yw[i] * zeta
        xw3 = complex(np.trapezoid(x3, xs))
        xw5 = complex(np.trapezoid(-x3 * xs, xs))

        # 운동방정식: [−ωe²(M+A) + iωe B + C]·u = F
        lhs = np.array([
            [-we ** 2 * (mass + a33) + 1j * we * b33 + c33,
             -we ** 2 * a35 + 1j * we * b35 + c35],
            [-we ** 2 * a53 + 1j * we * b53 + c53,
             -we ** 2 * (iyy + a55) + 1j * we * b55 + c55]])
        rhs = np.array([xw3, xw5])
        z_hat, th_hat = np.linalg.solve(lhs, rhs)
        out.append(StripRAO(
            omega=omega,
            heave_rao=abs(z_hat),
            pitch_rao=abs(th_hat) / k,
            heave_phase=float(np.angle(z_hat)),
            pitch_phase=float(np.angle(th_hat))))
    return out
