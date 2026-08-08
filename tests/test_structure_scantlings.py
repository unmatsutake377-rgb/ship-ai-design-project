"""DNV 계보 국부 스캔틀링 — 실선 두께 대역 앵커 (스펙 §3)."""
import pytest


def test_plate_thickness_formula_anchor():
    """원전 식 손계산: p=100 kN/m², s=0.7m, σ=120, ka=1 (s/l 0.4↓)
    → t = 15.8·0.7·√(100/120) + 1.5 = 11.60 mm."""
    from src.physics.structure.scantlings import plate_thickness_mm
    t = plate_thickness_mm(100.0, 0.7, 2.5, 120.0, tk_mm=1.5)
    assert t == pytest.approx(15.8 * 0.7 * (100.0 / 120.0) ** 0.5
                              + 1.5, rel=1e-6)


def test_plate_thickness_aspect_correction():
    """정사각 패널(s/l=1) ka=0.72 하한 — 좁고 긴 패널보다 얇게 허용."""
    from src.physics.structure.scantlings import plate_thickness_mm
    t_long = plate_thickness_mm(100.0, 0.7, 2.5, 120.0)   # s/l 0.28
    t_sq = plate_thickness_mm(100.0, 0.7, 0.7, 120.0)     # s/l 1.0
    assert t_sq < t_long


def test_100m_cargo_bottom_band():
    """실선 sanity: 100m 화물선 선저 8~16mm 대역 (문헌 통상)."""
    from src.physics.structure.scantlings import (
        design_pressure_bottom,
        default_spacing_m,
        min_thickness_mm,
        plate_thickness_mm,
    )
    p = design_pressure_bottom(100.0, 15.0, 5.5)
    s = default_spacing_m(100.0)
    t = max(plate_thickness_mm(p, s, 2.5, 120.0),
            min_thickness_mm(100.0, 1.0, "bottom"))
    assert 8.0 < t < 16.0


def test_small_alu_band():
    """소형 알루 2~9mm 대역 (실선 USV 관행) — 전 크기 유효 확인."""
    from src.physics.structure.materials import MATERIALS
    from src.physics.structure.scantlings import (
        design_pressure_bottom,
        default_spacing_m,
        min_thickness_mm,
        plate_thickness_mm,
    )
    al = MATERIALS["al5083"]
    p = design_pressure_bottom(3.0, 1.2, 0.3)
    s = default_spacing_m(3.0)
    t = max(plate_thickness_mm(p, s, 0.5, 120.0 * al.f1, tk_mm=0.5),
            min_thickness_mm(3.0, al.f1, "bottom"))
    assert 2.0 < t < 9.0


def test_stiffener_modulus_positive_scaling():
    """늑골 단면계수 — 스팬 제곱·압력 비례 (원전 구조)."""
    from src.physics.structure.scantlings import stiffener_modulus_cm3
    z1 = stiffener_modulus_cm3(100.0, 0.7, 2.0, 1.0)
    z2 = stiffener_modulus_cm3(100.0, 0.7, 4.0, 1.0)
    assert z2 == pytest.approx(4.0 * z1, rel=1e-6)
    assert z1 > 0


def test_100m_section_modulus_vs_iacs_requirement():
    """2단계 통합: 규칙 두께로 조립한 단면계수 vs UR S11 요구치
    Z_req = M_total/σ(175) — 같은 자릿수 (3단계 판정 예고편).

    단위 사고 다발 지점: 1 N/mm² = 1000 kN/m² →
    Z[m³] = M[kN·m] / (σ[N/mm²]·1000)."""
    from src.physics.structure.midship import assemble_midship
    from src.physics.structure.scantlings import (
        design_pressure_bottom,
        design_pressure_deck,
        design_pressure_side,
        default_spacing_m,
        min_thickness_mm,
        plate_thickness_mm,
    )
    from src.physics.structure.wave_loads import iacs_wave_bending_knm

    loa, beam, depth, draft, cb = 100.0, 15.0, 8.0, 5.5, 0.75
    s = default_spacing_m(loa)
    pb = design_pressure_bottom(loa, beam, draft)
    ps = design_pressure_side(loa, beam, draft, depth)
    pd = design_pressure_deck(loa)
    tb = max(plate_thickness_mm(pb, s, 2.5, 120.0),
             min_thickness_mm(loa, 1.0, "bottom"))
    ts = max(plate_thickness_mm(ps, s, 2.5, 120.0),
             min_thickness_mm(loa, 1.0, "side"))
    td = max(plate_thickness_mm(pd, s, 2.5, 120.0),
             min_thickness_mm(loa, 1.0, "deck"))
    sec = assemble_midship(beam, depth, tb, ts, td,
                           n_bottom_long=int(beam / s),
                           n_deck_long=int(beam / s),
                           long_area_cm2=30.0)
    hog, _ = iacs_wave_bending_knm(loa, beam, cb)
    m_total_knm = hog * 1.5          # 정수 성분 개략 가산 (자릿수용)
    z_req_m3 = m_total_knm / (175.0 * 1000.0)
    print(f"\n[2단계] Z_deck {sec.z_deck_m3:.3f} / Z_keel "
          f"{sec.z_keel_m3:.3f} / Z_req {z_req_m3:.3f} m³ "
          f"(tb {tb:.1f} ts {ts:.1f} td {td:.1f} mm)")
    assert 0.2 < sec.z_deck_m3 / z_req_m3 < 5.0
