"""종강도 판정 + 증육 수렴 (구조 강도 3단계, 스펙 §2).

판정: σ = M/Z ≤ σ_allow = 175·f1 N/mm² (UR S11 p5 허용 굽힘 응력
— k = 1/f1 관계). 요구 단면계수 Z_req = |M_total|/(σ_allow·1000)
[M kN·m → Z m³ 단위 환산].

수렴: 규칙 두께(2단계 scantlings)로 시작 → Z 부족 시 지배 위치
(갑판/선저) 판을 0.5mm씩 증육 반복 — 실무 설계 나선의 최소형.
구조 중량은 병기 전용 (Watson 정본 유지 — 스펙 §4 이중 계산 금지).
"""
from __future__ import annotations

from src.physics.structure.materials import Material
from src.physics.structure.midship import assemble_midship
from src.physics.structure.scantlings import (
    default_spacing_m,
    design_pressure_bottom,
    design_pressure_deck,
    design_pressure_side,
    min_thickness_mm,
    plate_thickness_mm,
)

SIGMA_BASE_NMM2 = 175.0        # UR S11 허용 굽힘 응력 (연강 k=1)
T_STEP_MM = 0.5


def longitudinal_strength(loa: float, beam: float, depth: float,
                          draft: float, m_still_knm: float,
                          m_wave_hog_knm: float, m_wave_sag_knm: float,
                          material: Material,
                          spacing_m: float | None = None,
                          max_iter: int = 20) -> dict:
    """종강도 판정 — 합격 두께·단면계수·수렴 기록 반환."""
    f1 = material.f1
    # 허용응력: ISO 설계응력 σ_d 있으면 그것 (알루 5083 — KS V ISO
    # 12215-5 표 B.2 정본, 용접 항복 125를 허용처럼 쓰던 과대 정정),
    # 없으면 IACS UR S11 프레임 175·f1 (강 대형 상선).
    sigma_allow = (material.design_stress_nmm2
                   if material.design_stress_nmm2 is not None
                   else SIGMA_BASE_NMM2 * f1)
    sigma_local = 120.0 * f1               # 판 국부 허용 (원전 p89)
    s = spacing_m if spacing_m is not None else default_spacing_m(loa)
    span = max(2.5 * s, 0.5)

    pb = design_pressure_bottom(loa, beam, draft)
    ps = design_pressure_side(loa, beam, draft, depth)
    pd = design_pressure_deck(loa)
    tk = 1.5 if material.name in ("mild_steel", "ah36") else 0.5
    tb = max(plate_thickness_mm(pb, s, span, sigma_local, tk),
             min_thickness_mm(loa, f1, "bottom"))
    ts = max(plate_thickness_mm(ps, s, span, sigma_local, tk),
             min_thickness_mm(loa, f1, "side"))
    td = max(plate_thickness_mm(pd, s, span, sigma_local, tk),
             min_thickness_mm(loa, f1, "deck"))

    # 설계 모멘트: 호깅/새깅 두 조합 중 절대값 최대
    m_hog = m_still_knm + m_wave_hog_knm
    m_sag = m_still_knm + m_wave_sag_knm
    m_design = max(abs(m_hog), abs(m_sag))
    z_req = m_design / (sigma_allow * 1000.0)     # [m³]
    # 좌굴 압축판: 새깅=갑판 압축·호깅=선저 압축 (표준 보 부호).
    from src.physics.structure.buckling import plate_buckling_check
    sigma_f = material.yield_nmm2

    n_long = max(int(beam / s), 2)
    a_long = max(10.0, 0.3 * loa)                 # 종늑골 면적 cm² 개략

    iterations = 0
    passed = False
    buck_deck = buck_bottom = None
    for _ in range(max_iter + 1):
        sec = assemble_midship(beam, depth, tb, ts, td,
                               n_bottom_long=n_long,
                               n_deck_long=n_long,
                               long_area_cm2=a_long)
        # 항복: 두 판 모두 Z_req 이상
        yield_deck = sec.z_deck_m3 >= z_req
        yield_keel = sec.z_keel_m3 >= z_req
        # 좌굴: 압축판만. 표준 보 굽힘 부호 — 호깅(∩, M>0)은 갑판
        # 인장·선저 압축, 새깅(∪, M<0)은 갑판 압축·선저 인장.
        # 따라서 갑판 좌굴 = 새깅 모멘트, 선저 좌굴 = 호깅 모멘트
        # (백지 리뷰 [상] 부호 반전 검거 반영). 순두께 = 총 − tk.
        sa_deck = abs(m_sag) / (sec.z_deck_m3 * 1000.0)
        sa_bottom = abs(m_hog) / (sec.z_keel_m3 * 1000.0)
        buck_deck = plate_buckling_check(max(td - tk, 0.5), s,
                                         sa_deck, sigma_f,
                                         e_nmm2=material.e_nmm2)
        buck_bottom = plate_buckling_check(max(tb - tk, 0.5), s,
                                           sa_bottom, sigma_f,
                                           e_nmm2=material.e_nmm2)
        deck_ok = yield_deck and buck_deck["passed"]
        keel_ok = yield_keel and buck_bottom["passed"]
        if deck_ok and keel_ok:
            passed = True
            break
        # 불합격 판 증육 (항복·좌굴 공통 지렛대 = 두께↑ → Z↑·σ_E↑).
        # 스텝: 항복 부족비 + 좌굴 부족비 중 큰 쪽 비례 (최소 0.5).
        if not deck_ok:
            yd = z_req / max(sec.z_deck_m3, 1e-9)
            bd = sa_deck / max(buck_deck["sigma_c_nmm2"], 1e-9)
            td += max(T_STEP_MM, td * (max(yd, bd) - 1.0) * 0.5)
        if not keel_ok:
            yk = z_req / max(sec.z_keel_m3, 1e-9)
            bk = sa_bottom / max(buck_bottom["sigma_c_nmm2"], 1e-9)
            tb += max(T_STEP_MM, tb * (max(yk, bk) - 1.0) * 0.5)
        iterations += 1

    governing = "deck" if sec.z_deck_m3 <= sec.z_keel_m3 else "keel"
    mass_per_m = sec.area_m2 * material.density_kgm3   # 종부재만
    note = ("합격" if passed
            else f"수렴 한도({max_iter}회) 초과 — 치수·재료 재검토")
    if loa < 90.0:
        note += "; 소형 하중 = 준정적 표준파 (IACS 범위 밖, C급)"

    # 보강재 좌굴 (KS V ISO 12215-5 표 B.2 §1) — 선저 국부 압력의
    # 요구 단면계수를 만족하며 웨브 세장비 h_w/t_w ≤ 0.50·√(E/σ_yw)
    # 인 플랫바를 설계·판정. 하드 게이트 연동 (플랫바는 웨브 증육
    # 으로 항상 수렴하나, 설계 성립 여부·최종 치수를 확정). girder
    # 단면 모델(a_long)과 분리 — NSGA 파급 없이 국부 보강재만.
    from src.physics.structure.scantlings import stiffener_modulus_cm3
    from src.physics.structure.stiffener import design_flat_bar_stiffener
    z_stiff = stiffener_modulus_cm3(pb, s, span, f1)   # 선저 요구 [cm³]
    stiff = design_flat_bar_stiffener(
        z_stiff, material.yield_nmm2, material.e_nmm2)

    # 판 좌굴 (IACS S11.5, 압축 판) + 보강재 좌굴 (ISO) — 하드 게이트.
    # 갑판=새깅 압축·선저=호깅 압축 (압축판만). 정직 C급: 압축 판·
    # 플랫바 웨브 세장비 (T/L·전단·기둥 좌굴 미구현). 항복 Z는 총
    # 두께·좌굴 σ_E는 순두께 사용 (IACS 부식 후 관례, 기준 불일치
    # 명시 — 백지 리뷰 지적).
    buckling = {
        "deck": buck_deck, "bottom": buck_bottom, "stiffener": stiff,
        "passed": bool(buck_deck["passed"] and buck_bottom["passed"]
                       and stiff["passed"]),
        "note": ("판+보강재 좌굴 하드 게이트 (S11.5 압축 판 + KS V "
                 "ISO 12215-5 플랫바 웨브 세장비; T/L·전단 C급). "
                 "갑판=새깅·선저=호깅 압축"),
    }
    # 보강재 좌굴 불성립(웨브 상한 초과) 시 종합 판정에도 반영
    passed = bool(passed and stiff["passed"])

    return {
        "passed": passed,
        "buckling": buckling,
        "z_required_m3": z_req,
        "z_deck_m3": sec.z_deck_m3,
        "z_keel_m3": sec.z_keel_m3,
        "governing": governing,
        "t_bottom_mm": tb, "t_side_mm": ts, "t_deck_mm": td,
        "iterations": iterations,
        "sigma_allow_nmm2": sigma_allow,
        "structure_mass_per_m_kgm": mass_per_m,
        "material": material.name,
        "m_design_knm": m_design,
        "note": note,
    }
