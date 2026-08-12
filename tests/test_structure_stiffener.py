"""보강재 좌굴 게이트 — 플랫바 설계 + 세장비 (KS V ISO 12215-5).

원전: KS V ISO 12215-5 표 B.2 §1 (보강재 웨브 세장비 h/t_w ≤
0.50·√(E/σ_yw), 국부 좌굴 방지) + 요구 단면계수(scantlings).
플랫바 근사(바 단독 Z = t_w·h_w²/6, 부착판 무시 = 보수적 C급).
"""
import pytest


def test_flat_bar_meets_modulus_and_slenderness():
    """요구 Z 충족 + 세장비 한계 통과하는 플랫바 설계 (강)."""
    from src.physics.structure.stiffener import design_flat_bar_stiffener
    d = design_flat_bar_stiffener(z_req_cm3=100.0, sigma_yw_nmm2=235.0,
                                  e_nmm2=210000.0)
    assert d["passed"]
    # 세장비 = h_w/t_w ≤ 한계 0.50·√(E/σyw) ≈ 14.9
    assert d["slenderness"] <= d["slenderness_max"] + 1e-6
    # 제공 단면계수 ≥ 요구 (플랫바 바 단독 Z = t_w·h_w²/6000 cm³)
    z_provided = d["t_web_mm"] * d["web_height_mm"] ** 2 / 6000.0
    assert z_provided >= 100.0 - 1e-6


def test_aluminum_needs_thicker_web():
    """알루는 E 낮아(70000<210000) 세장비 한계 엄격 → 같은 Z에
    웨브가 강보다 두꺼워짐 (물리 방향)."""
    from src.physics.structure.stiffener import design_flat_bar_stiffener
    steel = design_flat_bar_stiffener(100.0, 235.0, 210000.0)
    alu = design_flat_bar_stiffener(100.0, 125.0, 70000.0)
    assert alu["slenderness_max"] < steel["slenderness_max"]
    assert alu["t_web_mm"] >= steel["t_web_mm"]


def test_slenderness_max_matches_iso():
    """세장비 한계 = 표 B.2 값 재현 (강 E24 ≈15·알루 5083 ≈12)."""
    from src.physics.structure.stiffener import design_flat_bar_stiffener
    assert design_flat_bar_stiffener(
        50.0, 235.0, 210000.0)["slenderness_max"] == pytest.approx(
        15.0, abs=0.5)
    assert design_flat_bar_stiffener(
        50.0, 125.0, 70000.0)["slenderness_max"] == pytest.approx(
        12.0, abs=0.5)


def test_gate_reports_stiffener_buckling():
    """구조 게이트 — 리포트에 보강재 좌굴 하위 검사 병기 + 하드
    게이트 반영 (passed에 보강재 세장비 통과 포함)."""
    from src.physics.structure.materials import MATERIALS
    from src.physics.structure.strength import longitudinal_strength
    r = longitudinal_strength(
        100.0, 16.2, 8.5, 5.9, -55000.0, 190000.0, -255000.0,
        MATERIALS["mild_steel"])
    assert "stiffener" in r["buckling"]
    st = r["buckling"]["stiffener"]
    assert st["profile"] == "tee"          # 실선 표준 T바
    assert "web_height_mm" in st and "web_slenderness" in st
    assert "flange_width_mm" in st
    assert isinstance(st["passed"], bool)
    # 합격 설계면 보강재도 세장비(웨브·플랜지) 통과
    if r["passed"]:
        assert st["passed"]


def test_tee_section_modulus_hand_calc():
    """T바 단면계수 손계산: 웨브 200×10·플랜지 80×10 → Z ≈ 99.8 cm³
    (부착판 무시 바 단독, 웨브 끝 극한섬유 기준)."""
    from src.physics.structure.stiffener import tee_section_modulus
    z = tee_section_modulus(h_w_mm=200.0, t_w_mm=10.0,
                            b_f_mm=80.0, t_f_mm=10.0)
    assert z == pytest.approx(99.8, abs=1.0)


def test_tee_web_shear_slenderness_matches_iso():
    """T웨브 전단 세장비 한계 1.29·√(E/τ_yw), τ_yw=σ_yw/√3 —
    표 B.2 §2 값 강 T웨브 50·알루 5083 40 재현."""
    from src.physics.structure.stiffener import (
        stiffener_web_shear_slenderness_limit,
    )
    assert stiffener_web_shear_slenderness_limit(
        210000.0, 235.0) == pytest.approx(50.0, abs=1.0)
    assert stiffener_web_shear_slenderness_limit(
        70000.0, 125.0) == pytest.approx(40.0, abs=1.0)


def test_tee_lighter_than_flat_bar():
    """T바가 같은 Z에 플랫바보다 가벼움 (웨브 슬렌더 허용 크므로)
    — 실선이 T/L 쓰는 이유. 둘 다 세장비 통과."""
    from src.physics.structure.stiffener import (
        design_flat_bar_stiffener,
        design_tee_stiffener,
    )
    z = 150.0
    flat = design_flat_bar_stiffener(z, 235.0, 210000.0)
    tee = design_tee_stiffener(z, 235.0, 210000.0)
    assert tee["passed"] and flat["passed"]
    assert tee["area_cm2"] < flat["area_cm2"]
    assert tee["web_slenderness"] <= tee["web_slenderness_max"] + 1e-6
    assert tee["flange_slenderness"] <= tee["flange_slenderness_max"] + 1e-6
