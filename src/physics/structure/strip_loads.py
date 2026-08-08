"""스트립 동적 굽힘 모멘트 RAO (구조 강도 1단계 — 3중 교차검증 축).

원리: 자유 운동 해 (ẑ, θ̂)에서 스테이션별 내부 수직 하중 밀도
q(x) = 관성 − 유체력 = μ(x)·s̈(x) − [기진 − (부가질량·s̈ + 감쇠·ṡ
+ 복원·s)] 를 두 번 적분 → V(x)·M(x) 복소 진폭.

앵커: ① 폐합 항등식 (양끝 V·M ≈ 0 — 운동방정식 총평형의 내부
버전) ② 구속 모드 = 준정적 표준파 교차 (λ=L) ③ 자유 < 구속
(파면 타기 하중 경감).
부호: |M| 진폭만 반환 (위상별 호깅/새깅은 IACS·준정적 축이 판정).
"""
from __future__ import annotations

import numpy as np

from src.physics.seakeeping.strip import (
    G_ACC,
    RHO_SEAWATER,
    sectional_coeffs,
    sectional_excitation,
    sectional_setup,
)
from src.physics.structure.loads import _cumtrapz, weight_linear_density


def wave_bending_rao(mesh, draft: float, mass: float, iyy: float,
                     blocks, omegas, n_stations: int = 21,
                     contour_n: int = 12,
                     restrained: bool = False) -> list[dict]:
    """규칙파 ζa=1 기준 미드십 굽힘 모멘트 진폭 [N·m/m]."""
    xs, secs, yw = sectional_setup(mesh, draft,
                                   n_stations=n_stations,
                                   contour_n=contour_n)
    xmid_g = 0.5 * (mesh.bounds[0][0] + mesh.bounds[1][0])
    # 중량 밀도 μ(x) [kg/m] — 블록을 G 기준 좌표로 이동, 질량 폐합
    blocks_g = [(m, x0 - xmid_g, x1 - xmid_g) for m, x0, x1 in blocks]
    mu = weight_linear_density(xs, blocks_g) / G_ACC
    integ_mu = float(np.trapezoid(mu, xs))
    if integ_mu > 0:
        mu *= mass / integ_mu

    out = []
    for omega in omegas:
        k = omega * omega / G_ACC
        we = omega                                  # V=0
        m2d, n2d = sectional_coeffs(secs, we, contour_n=contour_n)
        x3 = sectional_excitation(secs, xs, yw, k, we, m2d, n2d)

        if restrained:
            s = np.zeros(len(xs), dtype=complex)
        else:
            # 자유 운동 해 — 기존 RAO 조립과 동일 계수 (V=0 항)
            def tr(arr):
                return float(np.trapezoid(arr, xs))
            a33, b33 = tr(m2d), tr(n2d)
            c33 = 2.0 * RHO_SEAWATER * G_ACC * tr(yw)
            a35 = -tr(m2d * xs)
            b35 = -tr(n2d * xs)
            c35 = -2.0 * RHO_SEAWATER * G_ACC * tr(yw * xs)
            a55 = tr(m2d * xs ** 2)
            b55 = tr(n2d * xs ** 2)
            c55 = 2.0 * RHO_SEAWATER * G_ACC * tr(yw * xs ** 2)
            xw3 = complex(np.trapezoid(x3, xs))
            xw5 = complex(np.trapezoid(-x3 * xs, xs))
            lhs = np.array([
                [-we ** 2 * (mass + a33) + 1j * we * b33 + c33,
                 -we ** 2 * a35 + 1j * we * b35 + c35],
                [-we ** 2 * a35 + 1j * we * b35 + c35,
                 -we ** 2 * (iyy + a55) + 1j * we * b55 + c55]])
            z_hat, th_hat = np.linalg.solve(lhs, np.array([xw3, xw5]))
            s = z_hat - xs * th_hat

        acc = -we ** 2 * s
        vel = 1j * we * s
        f_hydro = x3 - (m2d * acc + n2d * vel
                        + 2.0 * RHO_SEAWATER * G_ACC * yw * s)
        q = mu * acc - f_hydro
        shear = _cumtrapz(q, xs)
        moment = _cumtrapz(shear, xs)
        bal_v = float(abs(shear[-1])
                      / max(np.max(np.abs(shear)), 1e-12))
        bal_m = float(abs(moment[-1])
                      / max(np.max(np.abs(moment)), 1e-12))
        ramp = (xs - xs[0]) / (xs[-1] - xs[0])
        moment = moment - moment[-1] * ramp
        m_mid = abs(complex(np.interp(0.0, xs, moment.real),
                            np.interp(0.0, xs, moment.imag)))
        out.append({"omega": float(omega),
                    "m_mid_per_amp_nm": float(m_mid),
                    "balance_v": bal_v, "balance_m": bal_m})
    return out
