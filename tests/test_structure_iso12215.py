"""알루 설계응력 정본화 + 보강재 세장비 — KS V ISO 12215-5 앵커.

원전: KS V ISO 12215-5:2019 (ISO 12215-5 부합화) 표 B.2 (금속
보강재 기계적 특성·설계응력) + 부록 세장비 표 (오너 KS 무료 열람
캡처 판독 2026-08-12). 값은 사실이라 인용 (PDF 미보존·숫자만).
"""
import pytest


def test_al5083_design_stress_authoritative():
    """알루 5083 정본: σ_yw(용접 항복) 125·σ_d(설계응력) 88·
    τ_d 51·σ_uw 275·E 70000 — 과대 125 허용응력 오류 정정."""
    from src.physics.structure.materials import MATERIALS
    al = MATERIALS["al5083"]
    assert al.yield_nmm2 == pytest.approx(125.0)      # σ_yw 용접 항복
    assert al.design_stress_nmm2 == pytest.approx(88.0)  # σ_d 정본
    # σ_d = 0.7·σ_yw (비열처리 알루, 표 B.2)
    assert al.design_stress_nmm2 == pytest.approx(0.7 * 125.0, abs=1.0)
    assert al.tau_d_nmm2 == pytest.approx(51.0, abs=1.0)  # 0.58·σ_d
    assert al.e_nmm2 == pytest.approx(70000.0)
    assert al.grade in ("A", "B")          # C급 → 정본 승급


def test_steel_keeps_iacs_allowable():
    """강은 ISO σ_d 도입 안 함 — 대형 상선 IACS UR S11 프레임
    유지 (design_stress None → 175·f1 경로)."""
    from src.physics.structure.materials import MATERIALS
    assert MATERIALS["mild_steel"].design_stress_nmm2 is None
    assert MATERIALS["ah36"].design_stress_nmm2 is None


def test_gate_uses_iso_design_stress_for_aluminum():
    """구조 게이트 — 알루는 허용응력 σ_d=88 사용 (과대 93 정정).
    강은 175·f1 유지."""
    from src.physics.structure.materials import MATERIALS
    from src.physics.structure.strength import longitudinal_strength
    al = longitudinal_strength(
        30.0, 8.0, 3.0, 1.5, 100.0, 300.0, -400.0,
        MATERIALS["al5083"])
    assert al["sigma_allow_nmm2"] == pytest.approx(88.0, abs=0.5)
    st = longitudinal_strength(
        100.0, 16.2, 8.5, 5.9, -55000.0, 190000.0, -255000.0,
        MATERIALS["mild_steel"])
    assert st["sigma_allow_nmm2"] == pytest.approx(175.0, abs=0.5)


def test_stiffener_slenderness_limit_hand_calc():
    """보강재 세장비 한계 (표 B.2 §1·§2) — 금속 플랫바 웨브
    h/t_w ≤ 0.50·√(E/σ_yw). 강 E24(σ_yw 235)=15·알루 5083
    (σ_yw 125)=12 사전계산값 재현."""
    from src.physics.structure.buckling import stiffener_web_slenderness_max
    # 강 E24: 0.50·√(210000/235) ≈ 14.9 ≈ 15
    assert stiffener_web_slenderness_max(
        e_nmm2=210000.0, sigma_yw_nmm2=235.0) == pytest.approx(
        0.50 * (210000.0 / 235.0) ** 0.5, rel=1e-9)
    assert stiffener_web_slenderness_max(
        210000.0, 235.0) == pytest.approx(15.0, abs=0.5)
    # 알루 5083: 0.50·√(70000/125) ≈ 11.8 ≈ 12
    assert stiffener_web_slenderness_max(
        70000.0, 125.0) == pytest.approx(12.0, abs=0.5)
