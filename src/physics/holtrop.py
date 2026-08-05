"""Holtrop-Mennen 저항 추정 (상선급 실무 표준 경험식).

출처: Holtrop & Mennen, "An Approximate Power Prediction Method",
International Shipbuilding Progress 29 (1982) + Holtrop (1984) 재추정.
회귀 기반: 수조 모형시험 다수(배수량형 상선 계열)의 통계 —
유효 대역은 배수량 상선 (Fn < 0.45, Cp 0.55~0.85 근방).

전 크기 개방 1단계 (스펙 2026-08-06-all-size): Michell(얇은 배
가정)이 상선급 풍만 선형에서 깨지는 대역의 담당 후보. 채택 여부는
KCS 벤치마크 실측 A/B가 결정 (규칙이 아니라 실측이 심판 — 관례).

구현 범위 (1차): 마찰(ITTC-57)×형상계수 + 조파 + 구상선수 +
트랜섬 + 모형-실선 상관(CA). 부가물(RAPP)은 나선 경로 밖 — 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

RHO_SEAWATER = 1025.0
G = 9.81


@dataclass(frozen=True)
class HoltropInput:
    """Holtrop 입력 — 전부 실선(또는 모형) 실측 치수 [m, m², m³].

    lcb_frac: LCB 위치 (Lpp 중앙 기준 %/100, +선수쪽 — Holtrop 규약).
    abt: 구상선수 횡단면적 [m²] (없으면 0). hb: 구상선수 단면 중심
    높이 [m]. at: 트랜섬 침수 단면적 [m²] (없으면 0)."""
    lwl: float
    beam: float
    draft: float
    volume: float
    wetted_surface: float
    cb: float          # Lwl 기준
    cm: float
    cwp: float
    lcb_frac: float = 0.0
    abt: float = 0.0
    hb: float = 0.0
    at: float = 0.0
    cstern: float = 0.0   # 선미 형상 계수 (0=보통, +U형, -V형)


def run_length(h: HoltropInput) -> float:
    """선미 흐름부 길이 LR (Holtrop 부속식)."""
    cp = h.cb / h.cm
    return h.lwl * (1.0 - cp + 0.06 * cp * (100.0 * h.lcb_frac)
                    / (4.0 * cp - 1.0))


def form_factor(h: HoltropInput) -> float:
    """형상계수 1+k1 — 점성 저항의 3차원 증폭."""
    cp = h.cb / h.cm
    lr = run_length(h)
    c14 = 1.0 + 0.011 * h.cstern
    return (0.93 + 0.487118 * c14
            * (h.beam / h.lwl) ** 1.06806
            * (h.draft / h.lwl) ** 0.46106
            * (h.lwl / lr) ** 0.121563
            * (h.lwl ** 3 / h.volume) ** 0.36486
            * (1.0 - cp) ** (-0.604247))


def half_entrance_angle(h: HoltropInput) -> float:
    """수선 입사반각 iE [deg] (회귀식 — 실측 없을 때)."""
    cp = h.cb / h.cm
    lr = run_length(h)
    return 1.0 + 89.0 * math.exp(
        -((h.lwl / h.beam) ** 0.80856)
        * (1.0 - h.cwp) ** 0.30484
        * (1.0 - cp - 0.0225 * 100.0 * h.lcb_frac) ** 0.6367
        * (lr / h.beam) ** 0.34574
        * (100.0 * h.volume / h.lwl ** 3) ** 0.16302)


def wave_resistance(h: HoltropInput, speed: float,
                    rho: float = RHO_SEAWATER) -> float:
    """조파 저항 RW [N] (Fn ≤ 0.4 회귀 — 배수량 상선 대역)."""
    fn = speed / math.sqrt(G * h.lwl)
    if fn <= 0.0:
        return 0.0
    cp = h.cb / h.cm
    b_l = h.beam / h.lwl
    if b_l < 0.11:
        c7 = 0.229577 * b_l ** 0.33333
    elif b_l < 0.25:
        c7 = b_l
    else:
        c7 = 0.5 - 0.0625 * h.lwl / h.beam
    ie = half_entrance_angle(h)
    c1 = (2223105.0 * c7 ** 3.78613
          * (h.draft / h.beam) ** 1.07961
          * (90.0 - ie) ** (-1.37565))
    # 구상선수 감쇠 c2 (없으면 1)
    if h.abt > 0.0:
        c3 = (0.56 * h.abt ** 1.5
              / (h.beam * h.draft * (0.31 * math.sqrt(h.abt)
                                     + h.draft - h.hb)))
        c2 = math.exp(-1.89 * math.sqrt(c3))
    else:
        c2 = 1.0
    c5 = 1.0 - 0.8 * h.at / (h.beam * h.draft * h.cm)
    l3_v = h.lwl ** 3 / h.volume
    if cp < 0.80:
        c16 = 8.07981 * cp - 13.8673 * cp ** 2 + 6.984388 * cp ** 3
    else:
        c16 = 1.73014 - 0.7067 * cp
    m1 = (0.0140407 * h.lwl / h.draft
          - 1.75254 * h.volume ** (1.0 / 3.0) / h.lwl
          - 4.79323 * h.beam / h.lwl - c16)
    if l3_v < 512.0:
        c15 = -1.69385
    elif l3_v < 1726.91:
        c15 = -1.69385 + (h.lwl / h.volume ** (1.0 / 3.0) - 8.0) / 2.36
    else:
        c15 = 0.0
    m4 = c15 * 0.4 * math.exp(-0.034 * fn ** (-3.29))
    lam = (1.446 * cp - 0.03 * h.lwl / h.beam
           if h.lwl / h.beam < 12.0 else 1.446 * cp - 0.36)
    return (c1 * c2 * c5 * h.volume * rho * G
            * math.exp(m1 * fn ** (-0.9) + m4 * math.cos(lam * fn ** (-2))))


def correlation_allowance(h: HoltropInput) -> float:
    """모형-실선 상관수정 CA (실선 예측용 — 모형 대조 시 0으로)."""
    t_l = h.draft / h.lwl
    c4 = t_l if t_l <= 0.04 else 0.04
    if h.abt > 0.0:
        c3 = (0.56 * h.abt ** 1.5
              / (h.beam * h.draft * (0.31 * math.sqrt(h.abt)
                                     + h.draft - h.hb)))
        c2 = math.exp(-1.89 * math.sqrt(c3))
    else:
        c2 = 1.0
    return (0.006 * (h.lwl + 100.0) ** (-0.16) - 0.00205
            + 0.003 * math.sqrt(h.lwl / 7.5) * h.cb ** 4 * c2
            * (0.04 - c4))


def total_resistance_holtrop(h: HoltropInput, speed: float,
                             rho: float = RHO_SEAWATER,
                             nu: float = 1.19e-6,
                             include_ca: bool = True) -> dict:
    """전 저항 [N] — 성분 분해 dict 반환 (rf·rw·ra·total·ct 등).

    include_ca=False: 모형 스케일 대조용 (CA는 실선 상관 보정)."""
    from src.physics.resistance import ittc_cf

    re = speed * h.lwl / nu
    cf = ittc_cf(re)
    k1 = form_factor(h)
    q = 0.5 * rho * speed ** 2 * h.wetted_surface
    rf = q * cf
    rv = rf * k1                      # 점성 (마찰×형상)
    rw = wave_resistance(h, speed, rho)
    ca = correlation_allowance(h) if include_ca else 0.0
    ra = q * ca
    total = rv + rw + ra
    return {"rf": rf, "form_factor": k1, "rv": rv, "rw": rw,
            "ra": ra, "total": total,
            "ct": total / q if q > 0 else float("nan"),
            "cf": cf, "reynolds": re,
            "froude": speed / math.sqrt(G * h.lwl)}
