"""KVLCC2 원전 데이터 (Yasukawa & Yoshimura 2015, JMST 20:37-52).

references/Yasukawa2015_MMG.pdf — Table 1 (p5, 0-기준) 주요목,
Table 3 (p12) 유체력 계수, Table 4 (p12) 선회 실측 앵커.
A급: MMG 표준법 원저 (오픈액세스) + SIMMAN2008 공개 실측 계보.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShipParticulars:
    lpp: float
    beam: float
    draft: float
    displacement_m3: float
    xg: float              # 미드십 기준 +선수 [m]
    cb: float
    dp: float              # 프로펠러 직경
    hr: float              # 러더 높이
    ar: float              # 러더 가동부 면적
    rho: float = 1000.0    # 수조 담수


@dataclass(frozen=True)
class MMGCoeffs:
    # 선체 미계수 (Table 3, SNAME prime)
    xvv: float
    xvr: float
    xrr: float
    xvvvv: float
    yv: float
    yr: float
    yvvv: float
    yvvr: float
    yvrr: float
    yrrr: float
    nv: float
    nr: float
    nvvv: float
    nvvr: float
    nvrr: float
    nrrr: float
    # 부가질량 (Motora 도표 계보)
    mx_p: float
    my_p: float
    jz_p: float
    # 프로펠러
    t_p: float
    w_p0: float
    k0: float
    k1: float
    k2: float
    c1: float              # 조종 중 반류 변화
    c2_plus: float         # βP > 0
    c2_minus: float        # βP < 0
    # 러더
    t_r: float
    a_h: float
    x_h_p: float
    eps: float
    kappa: float
    f_alpha: float
    gamma_r_plus: float    # βR > 0
    gamma_r_minus: float   # βR < 0
    ell_r_p: float
    x_r_p: float = -0.5    # 러더 위치 (미드십 기준 −L/2)


KVLCC2_L7 = ShipParticulars(
    lpp=7.00, beam=1.27, draft=0.46, displacement_m3=3.27,
    xg=0.25, cb=0.810, dp=0.216, hr=0.345, ar=0.0539)

KVLCC2_COEFFS = MMGCoeffs(
    xvv=-0.040, xvr=0.002, xrr=0.011, xvvvv=0.771,
    yv=-0.315, yr=0.083, yvvv=-1.607, yvvr=0.379,
    yvrr=-0.391, yrrr=0.008,
    nv=-0.137, nr=-0.049, nvvv=-0.030, nvvr=-0.294,
    nvrr=0.055, nrrr=-0.013,
    mx_p=0.022, my_p=0.223, jz_p=0.011,
    t_p=0.220, w_p0=0.40,
    k0=0.2931, k1=0.2753, k2=-0.1385,
    c1=2.0, c2_plus=1.6, c2_minus=1.1,
    t_r=0.387, a_h=0.312, x_h_p=-0.464,
    eps=1.09, kappa=0.50, f_alpha=2.747,
    gamma_r_plus=0.640, gamma_r_minus=0.395,
    ell_r_p=-0.710)

PAPER_ANCHORS = {
    "turning_advance_cal": 3.31, "turning_advance_exp": 3.25,
    "turning_tactical_cal": 3.36, "turning_tactical_exp": 3.34,
    "note": "Table 4 (p12) L7 모델 δ=35° — 계산/실측, L 배수",
}
