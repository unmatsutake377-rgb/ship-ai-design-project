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
    assert al.design_stress_nmm2 == pytest.approx(87.5)  # 보강재 0.7σ_yw
    # σ_d = 0.7·σ_yw (비열처리 알루, 표 B.2)
    assert al.design_stress_nmm2 == pytest.approx(0.7 * 125.0, rel=1e-9)
    assert al.tau_d_nmm2 == pytest.approx(0.58 * 87.5, rel=1e-9)
    assert al.e_nmm2 == pytest.approx(70000.0)
    assert al.grade in ("A", "B")          # C급 → 정본 승급


def test_steel_keeps_iacs_allowable():
    """강은 ISO 값을 물성으로 보유하되 **종강도 허용은 IACS
    175·f1 유지** — ISO 12215-5는 24 m 미만 소형 표준이라 대형
    상선에 적용하지 않는 정직 경계."""
    from src.physics.structure.materials import MATERIALS
    from src.physics.structure.strength import longitudinal_strength
    r = longitudinal_strength(
        100.0, 16.2, 8.5, 5.9, -55000.0, 190000.0, -255000.0,
        MATERIALS["mild_steel"])
    assert r["sigma_allow_nmm2"] == pytest.approx(175.0, abs=0.5)


def test_gate_uses_iso_design_stress_for_aluminum():
    """구조 게이트 — 알루는 허용응력 σ_d=88 사용 (과대 93 정정).
    강은 175·f1 유지."""
    from src.physics.structure.materials import MATERIALS
    from src.physics.structure.strength import longitudinal_strength
    al = longitudinal_strength(
        30.0, 8.0, 3.0, 1.5, 100.0, 300.0, -400.0,
        MATERIALS["al5083"])
    # 표 17 판재값 112.5 (보강재 88과 구분 — 종강도는 판재)
    assert al["sigma_allow_nmm2"] == pytest.approx(112.5, abs=0.5)
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


def test_plate_vs_stiffener_design_stress_split():
    """표 17 — 판재와 보강재 설계응력이 다르다.
    알루: 판재 min(0.6·σ_uw, 0.9·σ_yw)=min(165,112.5)=112.5 /
    보강재 0.7·σ_yw=88. 강: 판재 min(0.6·400,0.9·235)=211.5 /
    보강재 0.8·σ_y=188."""
    from src.physics.structure.materials import MATERIALS
    al = MATERIALS["al5083"]
    assert al.design_stress_plate_nmm2 == pytest.approx(112.5, abs=0.5)
    assert al.design_stress_nmm2 == pytest.approx(87.5, abs=0.1)
    st = MATERIALS["mild_steel"]
    # 강 판재 = min(0.6·σ_u, 0.9·σ_y) — σ_u 400 (E24 표 B.2)
    assert st.design_stress_plate_nmm2 == pytest.approx(211.5, abs=1.0)
    assert st.design_stress_nmm2 == pytest.approx(188.0, abs=1.0)


def test_frp_authoritative_csm():
    """FRP E-glass CSM 정본 (표 C.9 ψ=0.30 손적층 + 표 17):
    E 8267·σ_uf 155·σ_ut 112·σ_uc 141·τ_u 59 →
    판재 σ_d = 0.5·σ_uf·k_AM(0.9) ≈ 70 · 보강재 0.5·σ_ut·0.9 ≈ 50."""
    from src.physics.structure.materials import MATERIALS
    f = MATERIALS["frp_eglass"]
    assert f.e_nmm2 == pytest.approx(8267.0, rel=0.01)
    # 정확값 (반올림 금지 — 비보수 방지, 백지 리뷰 [중])
    assert f.design_stress_plate_nmm2 == pytest.approx(0.5*155*0.9, rel=1e-9)
    assert f.design_stress_nmm2 == pytest.approx(0.5*112*0.9, rel=1e-9)
    assert f.tau_d_nmm2 == pytest.approx(0.5*59*0.9, rel=1e-9)
    # 좌굴 σ_F는 압축 극한 σ_uc (굽힘 아님)
    assert f.sigma_compression_nmm2 == pytest.approx(141.0)
    assert f.corrosion_tk_mm == 0.0        # FRP 부식 안 함
    assert f.grade in ("A", "B")
    assert "C.9" in f.note or "12215" in f.note


def test_gate_uses_plate_stress_for_girder():
    """종강도 허용 = 판재 설계응력 (갑판·선저 극한섬유).
    알루 88 → 112.5 완화 (과보수 정정)."""
    from src.physics.structure.materials import MATERIALS
    from src.physics.structure.strength import longitudinal_strength
    r = longitudinal_strength(30.0, 8.0, 3.0, 1.5, 100.0, 300.0,
                              -400.0, MATERIALS["al5083"])
    assert r["sigma_allow_nmm2"] == pytest.approx(112.5, abs=0.5)


def test_girder_frame_declared_not_name_blacklist():
    """프레임은 재료가 선언 — 신규 강종이 조용히 ISO 타는 것 방지
    (백지 리뷰 [중]). ISO 선언인데 판재 σ_d 없으면 예외."""
    import dataclasses

    import pytest as _pt

    from src.physics.structure.materials import MATERIALS
    from src.physics.structure.strength import longitudinal_strength
    assert MATERIALS["mild_steel"].girder_frame == "IACS"
    assert MATERIALS["al5083"].girder_frame == "ISO"
    broken = dataclasses.replace(MATERIALS["al5083"],
                                 design_stress_plate_nmm2=None)
    with _pt.raises(ValueError):
        longitudinal_strength(30.0, 8.0, 3.0, 1.5, 100.0, 300.0,
                              -400.0, broken)
