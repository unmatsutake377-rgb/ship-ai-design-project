"""Wageningen B-시리즈 프로펠러 설계 (3단계 2차, 스펙 2026-08-06 §4).

Kt·Kq 다항: Oosterveld & van Oossanen (1975) 회귀 — 계수는
data/wageningen_b_coeffs.csv (출처 헤더 참조, A급 학술 데이터).
유효 대역: Z 2~7, AE/A0 0.30~1.05, P/D 0.5~1.4, Rn 2e6 (모형 기준
— Reynolds 수정은 후속, 예비 설계 목적엔 2차 효과).

설계 루프: 소요 추력 T → (n, P/D) 스윕 → T 달성점 중 효율 최대 →
ηD = η0·ηH·ηR → 제동동력. 반류·추력감소는 화물선 통상 개략
(w 0.30, t 0.20 — C급, Holtrop 부속식으로 승급 예정).
캐비테이션: Keller 최소 전개면적비 검사.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

RHO_SEAWATER = 1025.0


def wake_fraction(cb: float) -> float:
    """Taylor 반류 근사 (단축선): w = 0.5·Cb − 0.05 (고전 회귀, B급).

    Cb 연동 — 풍만할수록 선미 반류 큼. Holtrop 정밀 부속식은
    공식 전문 확보 시 승급 (기억 하드코딩 금지 관례)."""
    return max(0.10, min(0.45, 0.5 * cb - 0.05))


def thrust_deduction(cb: float) -> float:
    """추력감소 근사: t ≈ 0.6·w (단축선 통상비, B급)."""
    return 0.6 * wake_fraction(cb)
ETA_RELATIVE_ROT = 1.00    # ηR 개략 (0.98~1.02 대역)
P_ATM_PA = 101_325.0
P_VAPOR_PA = 2_340.0       # 15°C 해수 증기압 근방

_COEFFS = Path(__file__).resolve().parents[2] / "data/wageningen_b_coeffs.csv"


def _load_coeffs():
    kq, kt = [], []
    with open(_COEFFS, newline="") as f:
        for row in f:
            if row.startswith("#") or row.startswith("cq"):
                continue
            v = [float(x) for x in row.split(",")]
            if v[0] != 0.0:
                kq.append((v[0], v[1], v[2], v[3], v[4]))
            if v[5] != 0.0:
                kt.append((v[5], v[6], v[7], v[8], v[9]))
    return kt, kq


_KT, _KQ = _load_coeffs()


def kt_kq(j: float, pd: float, ear: float, z: int) -> tuple[float, float]:
    """개수(open-water) 추력·토크 계수 (Kt, Kq)."""
    kt = sum(c * j ** s * pd ** t * ear ** u * z ** v
             for c, s, t, u, v in _KT)
    kq = sum(c * j ** s * pd ** t * ear ** u * z ** v
             for c, s, t, u, v in _KQ)
    return kt, kq


def keller_min_ear(thrust_n: float, diameter: float, z: int,
                   depth_shaft: float, rho: float = RHO_SEAWATER) -> float:
    """Keller 캐비테이션 최소 전개면적비 (단축 상선 k=0.20)."""
    p0 = P_ATM_PA + rho * 9.81 * depth_shaft
    return ((1.3 + 0.3 * z) * thrust_n
            / (diameter ** 2 * (p0 - P_VAPOR_PA))) + 0.20


@dataclass(frozen=True)
class PropellerDesign:
    diameter: float
    pitch_ratio: float
    rpm: float
    ear: float
    z: int
    j: float
    kt: float
    kq: float
    eta0: float
    eta_d: float           # 준추진효율 (η0·ηH·ηR)
    thrust_n: float
    brake_power_kw: float
    cavitation_ok: bool
    ear_min_keller: float


class PropellerDesignError(ValueError):
    """설계 실패 — 소요 추력을 유효 대역 안에서 달성 불가."""


def design_propeller(resistance_n: float, speed_ms: float,
                     diameter_max: float, z: int = 4, ear: float = 0.55,
                     shaft_depth: float | None = None,
                     rho: float = RHO_SEAWATER,
                     cb: float = 0.70,
                     w_override: float | None = None,
                     t_override: float | None = None,
                     eta_r: float = ETA_RELATIVE_ROT) -> PropellerDesign:
    """소요 추력을 채우는 (P/D, rpm) 중 개수 효율 최대점 선택.

    diameter_max: 흘수 제한 직경 (통상 D ≈ 0.65~0.75T) — 큰 직경이
    저회전·고효율이라 상한을 그대로 채택 (실무 관례).
    cb: 반류·추력감소 Taylor 근사 입력 (배 풍만도 연동, B급).
    w_override/t_override: Holtrop 정밀 부속식 주입 (2단 설계 —
    1차 Taylor로 직경 확정 후 정밀 w·t로 재설계)."""
    w = w_override if w_override is not None else wake_fraction(cb)
    td = t_override if t_override is not None else thrust_deduction(cb)
    t_req = resistance_n / (1.0 - td)
    va = speed_ms * (1.0 - w)
    d = diameter_max
    if shaft_depth is None:
        shaft_depth = d          # 축 몰수깊이 개략 (≈D)
    best = None
    for pd_i in [0.5 + 0.02 * i for i in range(46)]:      # P/D 0.5~1.4
        # J 스윕으로 T(J)=T_req 교점 탐색 (n = Va/(J·D))
        for ji in range(120, 4, -1):
            j = ji / 100.0                               # J 1.20→0.05
            kt, kq = kt_kq(j, pd_i, ear, z)
            if kt <= 0.0 or kq <= 0.0:
                continue
            n = va / (j * d)
            thrust = rho * n ** 2 * d ** 4 * kt
            if thrust >= t_req:
                eta0 = j * kt / (2.0 * math.pi * kq)
                if best is None or eta0 > best[0]:
                    best = (eta0, j, pd_i, kt, kq, n, thrust)
                break                                    # 이 P/D 완료
    if best is None:
        raise PropellerDesignError(
            f"소요 추력 {t_req / 1e3:.0f} kN을 D {d:.1f} m·B{z}-{ear:.2f} "
            "대역(P/D 0.5~1.4)에서 달성 불가 — 직경·날개수 재검토 필요.")
    eta0, j, pd_i, kt, kq, n, thrust = best
    eta_h = (1.0 - td) / (1.0 - w)
    eta_d = eta0 * eta_h * eta_r
    pb_kw = resistance_n * speed_ms / eta_d / 1000.0
    ear_min = keller_min_ear(t_req, d, z, shaft_depth, rho)
    return PropellerDesign(
        diameter=d, pitch_ratio=pd_i, rpm=n * 60.0, ear=ear, z=z,
        j=j, kt=kt, kq=kq, eta0=eta0, eta_d=eta_d, thrust_n=thrust,
        brake_power_kw=pb_kw, cavitation_ok=ear >= ear_min,
        ear_min_keller=ear_min)


def holtrop_wake_thrust(lpp: float, beam: float, draft: float,
                        d_prop: float, cp: float, lcb_pct: float,
                        wetted_surface: float, k1: float,
                        speed: float, cstern: float = 0.0,
                        nu: float = 1.19e-6) -> tuple[float, float, float]:
    """Holtrop-Mennen (1982) 단축선 w·t·ηR 정밀 부속식.

    출처: 원논문 회귀 — 공개 구현(aliozdenisik/kcs-resistance-
    propulsion, Schneekluth·Birk·Bertram 교차 검증 표기) 대조 이식.
    lcb_pct: %Lpp (+선수). 반환 (w, t, eta_r). Taylor 근사의 승급판."""
    from src.physics.resistance import ittc_cf

    re = speed * lpp / nu
    cv = (1.0 + k1) * ittc_cf(re)
    s = wetted_surface
    ta = draft
    if beam / ta < 5.0:
        c8 = beam * s / (lpp * d_prop * ta)
    else:
        c8 = s * (7.0 * beam / ta - 25.0) / (lpp * d_prop
                                             * (beam / ta - 3.0))
    c9 = c8 if c8 < 28.0 else 32.0 - 16.0 / (c8 - 24.0)
    c11 = (ta / d_prop if ta / d_prop < 2.0
           else 0.0833333 * (ta / d_prop) ** 3 + 1.33333)
    cp1 = 1.45 * cp - 0.315 - 0.0225 * lcb_pct
    c19 = 0.0015 * cstern
    c20 = 1.0 + 0.015 * cstern
    w = (c9 * c20 * cv * (lpp / ta)
         * (0.050776 + 0.93405 * c11 * cv / (1.0 - cp1))
         + 0.27915 * c20 * math.sqrt(beam / (lpp * (1.0 - cp1)))
         + c19 * c20)
    t = (0.25014 * (beam / lpp) ** 0.28956
         * (math.sqrt(beam * draft) / d_prop) ** 0.2624
         / (1.0 - cp + 0.0225 * lcb_pct) ** 0.01762
         + 0.0015 * cstern)
    eta_r = 0.9922 - 0.05908 * 0.55 + 0.07424 * (cp - 0.0225 * lcb_pct)
    return w, t, eta_r
