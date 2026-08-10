"""치수 → MMGShip 조립 헬퍼 (게이트·프록시 공용).

pipeline._maneuvering_gate에 있던 구성 사슬을 추출 — 러더 DNV
면적·종횡비 1.5(C급), Kt(J) 2차 3점 정합, Kijima+Yoshimura 계수
추정, 자항 평형 nP. 대역 밖(Cb<0.40)은 EstimationRangeError를
그대로 올림 (호출측 정직 처리 몫).

관례 명기 (백지 리뷰 지적 반영): 인자 loa는 **Lpp로 사용**
(프로젝트 전반의 dims.loa = 수선간장 근사 관례 — LOA/Lpp 구분
없는 개략, 3~5% 계통 오차 C급). xg=0.0은 미드십 CG 개략 (기존
게이트 계보 그대로 — KG 분포모델의 세로판은 백로그).
"""
from __future__ import annotations

import math

from src.physics.maneuvering.kvlcc2 import ShipParticulars
from src.physics.maneuvering.mmg import MMGShip, solve_self_propulsion


def mmg_ship_from_dims(loa: float, beam: float, draft: float,
                       cb: float, displacement_m3: float,
                       d_prop: float, resistance_n: float,
                       speed_ms: float, ear: float, z: int,
                       pitch_ratio: float) -> tuple[MMGShip, dict]:
    """치수·저항·프로펠러 → (MMGShip, 추정 노트)."""
    from src.physics.maneuvering.estimation import estimate_mmg_coeffs
    from src.physics.propeller import kt_kq, thrust_deduction, wake_fraction
    from src.sim_adapters.rudder import rudder_area_dnv

    ar = rudder_area_dnv(loa, beam, draft)
    hr = math.sqrt(1.5 * ar)              # 종횡비 1.5 통상 (C급)
    w0 = wake_fraction(cb)
    t0 = thrust_deduction(cb)
    kt0 = kt_kq(0.0, pitch_ratio, ear, z)[0]
    kt5 = kt_kq(0.5, pitch_ratio, ear, z)[0]
    kt1 = kt_kq(1.0, pitch_ratio, ear, z)[0]
    k0 = kt0
    k2 = 2.0 * (kt1 + kt0 - 2.0 * kt5)
    k1 = kt1 - kt0 - k2
    co, notes = estimate_mmg_coeffs(
        loa=loa, beam=beam, draft=draft, cb=cb,
        displacement_m3=displacement_m3, xg=0.0,
        dp=d_prop, hr=hr, ar=ar, w_p0=w0, t_p=t0,
        k0=k0, k1=k1, k2=k2)
    par = ShipParticulars(
        lpp=loa, beam=beam, draft=draft,
        displacement_m3=displacement_m3, xg=0.0, cb=cb,
        dp=d_prop, hr=hr, ar=ar, rho=1025.0)
    n_p = solve_self_propulsion(par, co, speed_ms, resistance_n)
    return MMGShip(par=par, co=co, r0_n=resistance_n, n_p=n_p,
                   u0=speed_ms), notes
