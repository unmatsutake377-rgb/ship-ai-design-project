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
    sigma_allow = SIGMA_BASE_NMM2 * f1
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

    n_long = max(int(beam / s), 2)
    a_long = max(10.0, 0.3 * loa)                 # 종늑골 면적 cm² 개략

    iterations = 0
    passed = False
    for _ in range(max_iter + 1):
        sec = assemble_midship(beam, depth, tb, ts, td,
                               n_bottom_long=n_long,
                               n_deck_long=n_long,
                               long_area_cm2=a_long)
        if sec.z_deck_m3 >= z_req and sec.z_keel_m3 >= z_req:
            passed = True
            break
        governing = "deck" if sec.z_deck_m3 < sec.z_keel_m3 else "keel"
        # 적응 스텝: 부족비 비례 점프 (큰 부족 = 큰 걸음) + 최소 0.5
        deficit = z_req / max(min(sec.z_deck_m3, sec.z_keel_m3), 1e-9)
        if governing == "deck":
            step = max(T_STEP_MM, td * (deficit - 1.0) * 0.6)
            td += step
        else:
            step = max(T_STEP_MM, tb * (deficit - 1.0) * 0.6)
            tb += step
        iterations += 1

    governing = "deck" if sec.z_deck_m3 <= sec.z_keel_m3 else "keel"
    mass_per_m = sec.area_m2 * material.density_kgm3   # 종부재만
    note = ("합격" if passed
            else f"수렴 한도({max_iter}회) 초과 — 치수·재료 재검토")
    if loa < 90.0:
        note += "; 소형 하중 = 준정적 표준파 (IACS 범위 밖, C급)"
    return {
        "passed": passed,
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
