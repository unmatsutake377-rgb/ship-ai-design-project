"""우리 배 → MMG 계수 추정 (조종성 2단계, 스펙 §2·§5-2).

원전: Yoshimura & Masumoto (MARSIM 2012 — JASNAOE 14 (2011) 영역판,
references/Yoshimura_MMG_database.pdf, 저자 공식 researchmap 배포):
- 선형 = Kijima (1990) 계보 Eq 11 (p5 이미지 판독): Y'β0 = 0.5πk +
  1.4·Cb/(L/B), (Y'r−m'x)0 = 0.5·Cb/(L/B), N'β0 = k, N'r0 =
  −0.54k+k² (k = 2d/L) — 원전 명시: 풀선형에도 선형 미계수는 유효
- 비선형 = Eq 14~16 (p6), 상호작용 = Eq 17~18 (p7): tR 0.39 (τ'=0),
  aH = 3.6·Cb/(L/B) (상선), x'H = −0.4, ℓ'R = −0.9,
  γR = 2.06·Cb/(L/B) + 0.14, ε = 2.26 − 1.82(1−wP0), κ = 0.55/ε
- fα = Fujii 식 6.13Λ/(Λ+2.25) (Yasukawa 2015 Eq 38)

부호 변환: 원전은 β-r' 관례 (β ≈ −v') — 홀수 β 차수 항은 부호
반전해 v'-r' 관례(MMGCoeffs)로 사상.

적용 대역 (원전 p9): L/B 2.6~7.1, d/B 0.25~0.46, Cb 0.51~0.65.
- Cb 0.65~0.85 (풀선형): 선형은 유효(원전 명시), 비선형·상호작용은
  외삽 — C급 정직 표기 (notes["band"]="extrapolated_full")
- Cb 0.40~0.51 (슬렌더 — NSGA 전선 대역): 완만 외삽 C급
  (notes["band"]="extrapolated_slender") — Kijima 선형은 반이론
  (0.5πk 저종횡비 날개 항)이라 연속, 풀선형 반대편 외삽이 자기
  대조 비 1.02로 관용성 실증 (2026-08-10 확장, 마지막 각주 해소)
- Cb < 0.40: 정직 거절 (EstimationRangeError)
부가질량 m'x·m'y·J'z: Motora 계보 미확보 — KVLCC2 실측 무차원값
(0.022/0.223/0.011)을 대표값으로 채용 (C급, 자릿수 목적).
C1·C2 반류 변화도 KVLCC2 값 (C급).
"""
from __future__ import annotations

import math

from src.physics.maneuvering.kvlcc2 import MMGCoeffs


class EstimationRangeError(ValueError):
    """회귀 대역 밖 — 소형·특수 선형은 계수 추정 불가 (정직 거절)."""


def fujii_lift_gradient(hr: float, ar: float) -> float:
    """Fujii 러더 양력 기울기 fα = 6.13Λ/(Λ+2.25), Λ = HR²/AR."""
    lam = hr * hr / max(ar, 1e-9)
    return 6.13 * lam / (lam + 2.25)


def estimate_mmg_coeffs(loa: float, beam: float, draft: float,
                        cb: float, displacement_m3: float, xg: float,
                        dp: float, hr: float, ar: float,
                        w_p0: float, t_p: float,
                        k0: float, k1: float, k2: float,
                        rho: float = 1025.0
                        ) -> tuple[MMGCoeffs, dict]:
    """치수·풍만도 → MMGCoeffs + 등급 노트."""
    lb = loa / beam
    db = draft / beam
    if cb < 0.40:
        raise EstimationRangeError(
            f"Cb {cb:.2f} < 0.40 — 회귀 외삽 한계 밖 (소형·극날씬"
            " 선형). 조종 성적표는 기존 민첩성 지표로.")
    if 0.51 <= cb <= 0.65 and 2.6 <= lb <= 7.1 and 0.25 <= db <= 0.46:
        band = "in"
    elif cb < 0.51:
        band = "extrapolated_slender"
    else:
        band = "extrapolated_full"

    k = 2.0 * draft / loa
    cb_lb = cb / lb

    # 부가질량 (KVLCC2 대표값 — C급)
    mx_p, my_p, jz_p = 0.022, 0.223, 0.011
    xg_p = xg / loa

    # 선형 (Kijima Eq 11, β→v 부호 변환: Y'v=−Y'β, N'v=−N'β)
    yv = -(0.5 * math.pi * k + 1.4 * cb_lb)
    yr = 0.5 * cb_lb + mx_p          # (Y'r − m'x)0 = 0.5·Cb/(L/B)
    nv = -k
    nr = -0.54 * k + k * k

    # 비선형 (Eq 14~16, τ'=0 — 홀수 β 차수 부호 반전)
    xvv = 1.15 * cb_lb - 0.18
    xvr = -((-1.91 * cb_lb + 0.08) + my_p)      # X'βr − m'y 회귀
    xrr = (-0.085 * cb_lb + 0.008) - xg_p * my_p
    xvvvv = -6.68 * cb_lb + 1.10
    yvvv = -(0.185 * lb + 0.48)
    yvvr = -0.75
    yvrr = -(0.26 * (1.0 - cb) * lb + 0.11)
    yrrr = -0.051
    nvvv = -(-0.69 * cb + 0.66)
    nvvr = 1.55 * cb_lb - 0.76
    nvrr = -(0.075 * (1.0 - cb) * lb - 0.098)
    nrrr = 0.25 * cb_lb - 0.056

    # 상호작용 (Eq 17~18, 상선 갈래)
    t_r = 0.39                        # 1−tR = 0.61 (τ'=0)
    a_h = 3.6 * cb_lb
    eps = 2.26 - 1.82 * (1.0 - w_p0)
    gamma = 2.06 * cb_lb + 0.14

    co = MMGCoeffs(
        xvv=xvv, xvr=xvr, xrr=xrr, xvvvv=xvvvv,
        yv=yv, yr=yr, yvvv=yvvv, yvvr=yvvr, yvrr=yvrr, yrrr=yrrr,
        nv=nv, nr=nr, nvvv=nvvv, nvvr=nvvr, nvrr=nvrr, nrrr=nrrr,
        mx_p=mx_p, my_p=my_p, jz_p=jz_p,
        t_p=t_p, w_p0=w_p0, k0=k0, k1=k1, k2=k2,
        c1=2.0, c2_plus=1.6, c2_minus=1.1,
        t_r=t_r, a_h=a_h, x_h_p=-0.4,
        eps=eps, kappa=0.55 / max(eps, 1e-9),
        f_alpha=fujii_lift_gradient(hr, ar),
        gamma_r_plus=gamma, gamma_r_minus=gamma,
        ell_r_p=-0.9)
    band_note = {
        "in": "Yoshimura 회귀 대역 내",
        "extrapolated_full": "풀선형 외삽 — 선형 Kijima 유효(원전 "
                             "명시), 비선형·상호작용 C급",
        "extrapolated_slender": "슬렌더 외삽 (Cb 0.40~0.51) — 반이론"
                                " 선형식 연속 외삽 C급, 자기 대조"
                                " 관용성 계보",
    }
    notes = {
        "band": band,
        "grade": "B" if band == "in" else "C",
        "note": band_note[band],
        "added_mass": "KVLCC2 대표값 (C급)",
    }
    return co, notes
