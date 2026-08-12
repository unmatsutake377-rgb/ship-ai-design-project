"""보강재(stiffener) 설계 + 좌굴 게이트 (KS V ISO 12215-5).

요구 단면계수 Z를 충족하면서 웨브 국부 좌굴(세장비)을 피하는
플랫바 보강재를 설계한다. 원전 KS V ISO 12215-5 표 B.2 §1:
웨브 세장비 h_w/t_w ≤ 0.50·√(E/σ_yw) (실제응력=설계응력 조건).

세장비 변수는 **σ_yw(용접 항복)** — 표 B.2 §1 공식이 문자 그대로
0.50·√(E/σ_yw)이고, §2 사전계산값(알루 5083=12)을 역산하면
σ=125=σ_yw (설계응력 σ_d 88 아님, 백지 리뷰 σ_d 제안 기각 근거).

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


def stiffener_web_slenderness_limit(e_nmm2: float,
                                    sigma_yw_nmm2: float) -> float:
    """웨브 세장비 상한 h_w/t_w ≤ 0.50·√(E/σ_yw) (표 B.2 §1)."""
    return 0.50 * (e_nmm2 / sigma_yw_nmm2) ** 0.5


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
