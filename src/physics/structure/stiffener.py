"""보강재(stiffener) 설계 + 좌굴 게이트 (KS V ISO 12215-5).

요구 단면계수 Z를 충족하면서 웨브 국부 좌굴(세장비)을 피하는
플랫바 보강재를 설계한다. 원전 KS V ISO 12215-5 표 B.2 §1:
웨브 세장비 h_w/t_w ≤ 0.50·√(E/σ_yw) (실제응력=설계응력 조건).

세장비 변수는 **σ_yw(용접 항복)** — 표 B.2 §1 공식이 문자 그대로
0.50·√(E/σ_yw)이고, §2 사전계산값(알루 5083=12)을 역산하면
σ=125=σ_yw (설계응력 σ_d 88 아님, 백지 리뷰 σ_d 제안 기각 근거).

T/L바 (design_tee_stiffener): 웨브는 **전단 좌굴** 1.29·√(E/τ_yw),
τ_yw=σ_yw/√3 (§2 강 T웨브 50·알루 40 — 플랫바보다 슬렌더 허용
커 경량), 플랜지 돌출 0.50·√(E/σ_yw). 등두께·b_f=0.4·h_w 비례
에선 웨브가 지배하고 플랜지는 항상 통과(stocky — 비구속, 의도).

플랫바 근사 (C급 정직): 바 단독 단면계수 Z = t_w·h_w²/6 [mm³]
= t_w·h_w²/6000 [cm³] (부착판 유효폭 무시 → 보수적, 실제 T/L
프로파일은 더 가벼움). 웨브 두께를 최소값부터 올려 세장비
한계 안에서 Z를 만족하는 최소 플랫바를 찾는다 — h_w/t_w는 t_w
증가 시 t_w^1.5로 감소해 **통상 Z(≲수천 cm³)·웨브 상한 60mm
안에서 수렴** (거대 Z는 60mm로도 미달 = 정직 불합격 반환).
"""
from __future__ import annotations

import math

T_WEB_MIN_MM = 5.0
T_WEB_STEP_MM = 1.0
T_WEB_MAX_MM = 60.0
FLANGE_RATIO = 0.4        # T 플랜지폭 b_f = 0.4·h_w (통상 비례 C급)


def stiffener_web_slenderness_limit(e_nmm2: float,
                                    sigma_yw_nmm2: float) -> float:
    """플랫바 웨브 세장비 상한 h_w/t_w ≤ 0.50·√(E/σ_yw) (표 B.2 §1)."""
    return 0.50 * (e_nmm2 / sigma_yw_nmm2) ** 0.5


def stiffener_web_shear_slenderness_limit(e_nmm2: float,
                                          sigma_yw_nmm2: float
                                          ) -> float:
    """T/L 웨브 세장비 상한 h_w/t_w ≤ 1.29·√(E/τ_yw), τ_yw=σ_yw/√3
    (표 B.2 §1 — 전단 좌굴). 플랜지 지지로 플랫바보다 크게 허용
    (강 T웨브 50·알루 5083 40). 실선이 T/L 쓰는 이유."""
    tau_yw = sigma_yw_nmm2 / (3.0 ** 0.5)
    return 1.29 * (e_nmm2 / tau_yw) ** 0.5


def stiffener_flange_slenderness_limit(e_nmm2: float,
                                       sigma_yw_nmm2: float) -> float:
    """T/L 플랜지 돌출 세장비 상한 (d/t_f ≤ 0.50·√(E/σ_yw), 표
    B.2 §1). 플랫바 웨브와 동일 계수."""
    return 0.50 * (e_nmm2 / sigma_yw_nmm2) ** 0.5


def tee_section_modulus(h_w_mm: float, t_w_mm: float,
                        b_f_mm: float, t_f_mm: float) -> float:
    """T바 단면계수 [cm³] — 바 단독(부착판 무시 보수적), 웨브 끝
    극한섬유 기준. 웨브 밑 y=0 기준 중립축·관성 합성."""
    a_web = h_w_mm * t_w_mm
    a_fl = b_f_mm * t_f_mm
    a = a_web + a_fl
    y_web = h_w_mm / 2.0
    y_fl = h_w_mm + t_f_mm / 2.0
    ybar = (a_web * y_web + a_fl * y_fl) / a
    i = (t_w_mm * h_w_mm ** 3 / 12.0 + a_web * (ybar - y_web) ** 2
         + b_f_mm * t_f_mm ** 3 / 12.0 + a_fl * (y_fl - ybar) ** 2)
    c = max(ybar, h_w_mm + t_f_mm - ybar)
    return (i / c) / 1000.0                # mm³ → cm³


def _solve_tee_height(z_req_cm3: float, t_mm: float) -> float:
    """등두께 t·플랜지 0.4·h_w T바가 Z_req를 내는 웨브 높이 (이분)."""
    lo, hi = 1.0, 5000.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        z = tee_section_modulus(mid, t_mm, FLANGE_RATIO * mid, t_mm)
        if z < z_req_cm3:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def design_tee_stiffener(z_req_cm3: float, sigma_yw_nmm2: float,
                         e_nmm2: float,
                         t_min_mm: float = T_WEB_MIN_MM) -> dict:
    """요구 Z + 웨브 전단 세장비 + 플랜지 세장비 만족 최소 T바.

    등두께 t (웨브=플랜지)·플랜지폭 b_f=0.4·h_w. t를 최소부터 올려
    h_w를 Z 만족하게 잡고 세장비 두 한계 검사. T웨브는 전단 기준
    (플랫바보다 슬렌더 허용 커 경량)."""
    web_lam = stiffener_web_shear_slenderness_limit(e_nmm2,
                                                    sigma_yw_nmm2)
    fl_lam = stiffener_flange_slenderness_limit(e_nmm2, sigma_yw_nmm2)
    t = max(t_min_mm, T_WEB_MIN_MM)
    passed = False
    while True:
        h_w = _solve_tee_height(z_req_cm3, t)
        b_f = FLANGE_RATIO * h_w
        web_sl = h_w / t
        fl_sl = (b_f / 2.0) / t
        if web_sl <= web_lam and fl_sl <= fl_lam:
            passed = True
            break
        if t >= T_WEB_MAX_MM:
            break
        t = min(t + T_WEB_STEP_MM, T_WEB_MAX_MM)
    b_f = FLANGE_RATIO * h_w
    return {
        "profile": "tee",
        "web_height_mm": float(h_w), "t_web_mm": float(t),
        "flange_width_mm": float(b_f), "t_flange_mm": float(t),
        "web_slenderness": float(h_w / max(t, 1e-9)),
        "web_slenderness_max": float(web_lam),
        "flange_slenderness": float((b_f / 2.0) / max(t, 1e-9)),
        "flange_slenderness_max": float(fl_lam),
        "area_cm2": float((h_w * t + b_f * t) / 100.0),
        "passed": bool(passed),
        "note": "T바 (웨브 전단·플랜지 세장비 KS V ISO 12215-5 표 "
                "B.2; 부착판 무시 보수적 C급). 플랫바보다 경량",
    }


def design_flat_bar_stiffener(z_req_cm3: float, sigma_yw_nmm2: float,
                              e_nmm2: float,
                              t_web_min_mm: float = T_WEB_MIN_MM
                              ) -> dict:
    """요구 Z + 세장비 한계 만족 최소 플랫바 → 치수·판정 반환.

    바 단독 Z = t_w·h_w²/6000 [cm³] → h_w = √(6000·Z/t_w) [mm].
    웨브를 t_web_min부터 증육해 h_w/t_w ≤ 한계 될 때까지."""
    lam_max = stiffener_web_slenderness_limit(e_nmm2, sigma_yw_nmm2)
    t_w = max(t_web_min_mm, T_WEB_MIN_MM)
    passed = False
    # 웨브 상한까지 증육 — 성립 시 그 치수, 미성립 시 상한(60mm)
    # 치수로 정직 반환 (실패 경로 치수 오염 방지 — 백지 리뷰 수리).
    while True:
        h_w = math.sqrt(6000.0 * z_req_cm3 / t_w)
        if h_w / t_w <= lam_max:
            passed = True
            break
        if t_w >= T_WEB_MAX_MM:
            break                         # 상한 도달·미성립 (t_w 고정)
        t_w = min(t_w + T_WEB_STEP_MM, T_WEB_MAX_MM)
    return {
        "web_height_mm": float(h_w),
        "t_web_mm": float(t_w),
        "slenderness": float(h_w / max(t_w, 1e-9)),
        "slenderness_max": float(lam_max),
        "area_cm2": float(t_w * h_w / 100.0),
        "passed": bool(passed),
        "note": "플랫바 근사 (부착판 무시 보수적 C급; T/L은 경량) "
                "— KS V ISO 12215-5 표 B.2 웨브 세장비",
    }
