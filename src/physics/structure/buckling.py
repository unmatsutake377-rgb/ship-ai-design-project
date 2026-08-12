"""판 좌굴 강도 — IACS UR S11.5 (Rev.9, 압축 판 전용).

원전: references/IACS_UR_S11.pdf (페이지 이미지 판독 관례 —
p8 S11.5.2.1·p11 S11.5.3.1·p13 S11.5.5.1):
- 탄성 좌굴 (판·압축):
    σ_E = 0.9·m·E·(t_b/(1000·s))²  [N/mm²]
    m = 8.4/(ψ+1.1)  (종보강 판, 0≤ψ≤1; ψ=압축응력 최소/최대비)
- 임계 좌굴 (Johnson-Ostenfeld 소성 보정):
    σ_c = σ_E                    (σ_E ≤ σ_F/2)
    σ_c = σ_F·(1 − σ_F/(4·σ_E))  (σ_E > σ_F/2)
- 판정: σ_c ≥ β·σ_a  (β=1 판)

정직 범위 (C급 각주): **압축 판 좌굴만** 구현. 보강재 기둥·비틀림
좌굴(S11.5.2.2 — St Venant·극·섹토리얼 관성)과 전단 좌굴은 미구현
(백로그). 순두께 t_b는 표준 공제(S11.5.2.1.1 표: 0.05~0.10t)
대신 부식 여유 tk 재사용 근사 (C급).
"""
from __future__ import annotations

E_STEEL_NMM2 = 2.06e5           # 강 탄성계수 (원전 p8)


def plate_slenderness_m(psi: float = 1.0) -> float:
    """종보강 판 계수 m = 8.4/(ψ+1.1) (원전 p8, 0≤ψ≤1).

    ψ는 원전 적용대역 [0,1]로 클램프 — 이 프로젝트 용례는 종강도
    균일 압축(ψ≈1)뿐이라 방어용. ψ<0(응력 반전) 별도식은 미사용
    이나, 클램프는 m을 낮춰 σ_E를 과소평가 = 좌굴강도상 보수적
    (백지 리뷰 지적 반영 — 침묵 클램프의 방향성 명시)."""
    return 8.4 / (max(0.0, min(psi, 1.0)) + 1.1)


def elastic_plate_buckling_stress(t_net_mm: float, s_m: float,
                                  psi: float = 1.0,
                                  e_nmm2: float = E_STEEL_NMM2
                                  ) -> float:
    """탄성 좌굴 응력 σ_E [N/mm²] — 압축 판 (원전 S11.5.2.1)."""
    if s_m <= 0.0:
        raise ValueError(f"판 간격 s_m > 0 이어야 함 (받음 {s_m})")
    m = plate_slenderness_m(psi)
    return 0.9 * m * e_nmm2 * (t_net_mm / (1000.0 * s_m)) ** 2


def critical_buckling_stress(sigma_e_nmm2: float,
                             sigma_f_nmm2: float) -> float:
    """임계 좌굴 응력 σ_c — Johnson-Ostenfeld (원전 S11.5.3.1)."""
    if sigma_e_nmm2 <= sigma_f_nmm2 / 2.0:
        return sigma_e_nmm2
    return sigma_f_nmm2 * (1.0 - sigma_f_nmm2 / (4.0 * sigma_e_nmm2))


def stiffener_web_slenderness_max(e_nmm2: float,
                                  sigma_yw_nmm2: float) -> float:
    """보강재 플랫바 웨브 세장비 상한 h/t_w ≤ 0.50·√(E/σ_yw).

    원전 KS V ISO 12215-5 표 B.2 §1 (금속 보강재 국부 좌굴 방지 —
    실제응력=설계응력 σ_d 조건). 웨브가 이 비를 넘으면 좌굴.
    사전계산값(§2): 강 E24 15·알루 5083 12. (T·top hat 웨브는
    전단 τ_yw 기준 1.29 계수 — 필요 시 별도. C급: 압축 웨브만.)
    """
    return 0.50 * (e_nmm2 / sigma_yw_nmm2) ** 0.5


def plate_buckling_check(t_net_mm: float, s_m: float,
                         sigma_a_nmm2: float, sigma_f_nmm2: float,
                         psi: float = 1.0, beta: float = 1.0,
                         e_nmm2: float = E_STEEL_NMM2) -> dict:
    """압축 판 좌굴 판정 σ_c ≥ β·σ_a (원전 S11.5.5.1).

    e_nmm2: 재료 탄성계수 — 반드시 재료값 전달 (알루 7.0e4·강
    2.06e5). σ_E ∝ E라 강 기본값으로 알루를 계산하면 2.94배 과대
    = 비보수 오답 (백지 리뷰 [상] 검거)."""
    se = elastic_plate_buckling_stress(t_net_mm, s_m, psi, e_nmm2)
    sc = critical_buckling_stress(se, sigma_f_nmm2)
    demand = beta * sigma_a_nmm2
    return {
        "sigma_e_nmm2": float(se),
        "sigma_c_nmm2": float(sc),
        "sigma_a_nmm2": float(sigma_a_nmm2),
        "margin_ratio": float(sc / max(demand, 1e-9)),
        "passed": bool(sc >= demand),
    }
