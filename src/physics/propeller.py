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
WAKE_FRACTION = 0.30       # 화물선 통상 개략 (C급)
THRUST_DEDUCTION = 0.20    # 동상
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
                     rho: float = RHO_SEAWATER) -> PropellerDesign:
    """소요 추력을 채우는 (P/D, rpm) 중 개수 효율 최대점 선택.

    diameter_max: 흘수 제한 직경 (통상 D ≈ 0.65~0.75T) — 큰 직경이
    저회전·고효율이라 상한을 그대로 채택 (실무 관례)."""
    t_req = resistance_n / (1.0 - THRUST_DEDUCTION)
    va = speed_ms * (1.0 - WAKE_FRACTION)
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
    eta_h = (1.0 - THRUST_DEDUCTION) / (1.0 - WAKE_FRACTION)
    eta_d = eta0 * eta_h * ETA_RELATIVE_ROT
    pb_kw = resistance_n * speed_ms / eta_d / 1000.0
    ear_min = keller_min_ear(t_req, d, z, shaft_depth, rho)
    return PropellerDesign(
        diameter=d, pitch_ratio=pd_i, rpm=n * 60.0, ear=ear, z=z,
        j=j, kt=kt, kq=kq, eta0=eta0, eta_d=eta_d, thrust_n=thrust,
        brake_power_kw=pb_kw, cavitation_ok=ear >= ear_min,
        ear_min_keller=ear_min)
