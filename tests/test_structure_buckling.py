"""판 좌굴 강도 — IACS UR S11.5 손계산 앵커 (Rev.9 페이지 판독).

원전: references/IACS_UR_S11.pdf p8·p11·p13 (수식 조판 뒤섞임 →
페이지 이미지 렌더 판독 관례):
- S11.5.2.1 탄성 좌굴 σ_E = 0.9·m·E·(t_b/1000s)², m=8.4/(ψ+1.1)
- S11.5.3.1 임계 좌굴 (Johnson-Ostenfeld): σ_c = σ_E (σ_E≤σ_F/2) /
  σ_F(1−σ_F/4σ_E) (초과)
- S11.5.5.1 판정 σ_c ≥ β·σ_a (β=1 판)
"""
import pytest

from src.physics.structure.buckling import (
    critical_buckling_stress,
    elastic_plate_buckling_stress,
    plate_buckling_check,
)


def test_elastic_buckling_hand_calc():
    """σ_E 손계산: t 10mm·s 0.6m·ψ 1(균일)·연강 E 2.06e5 →
    m=8.4/2.1=4.0, σ_E=0.9·4·2.06e5·(10/600)² = 206.0 N/mm²."""
    se = elastic_plate_buckling_stress(t_net_mm=10.0, s_m=0.6, psi=1.0)
    assert se == pytest.approx(0.9 * 4.0 * 2.06e5 * (10.0 / 600.0) ** 2,
                               rel=1e-9)
    assert se == pytest.approx(206.0, abs=0.5)


def test_johnson_ostenfeld_branches():
    """임계 좌굴 두 분기 — 두꺼운 판(σ_E>σ_F/2)은 소성 보정,
    얇은 판(σ_E≤σ_F/2)은 탄성 그대로."""
    sf = 235.0
    # σ_E 206 > 117.5 → 소성: 235(1−235/824)=168.0
    assert critical_buckling_stress(206.0, sf) == pytest.approx(
        sf * (1.0 - sf / (4.0 * 206.0)), rel=1e-9)
    assert critical_buckling_stress(206.0, sf) == pytest.approx(
        168.0, abs=0.5)
    # σ_E 100 ≤ 117.5 → 탄성 그대로
    assert critical_buckling_stress(100.0, sf) == 100.0


def test_thin_plate_stays_elastic():
    """얇은 판(t 5mm) σ_E 51.5 ≤ σ_F/2 → σ_c = σ_E (탄성 지배)."""
    se = elastic_plate_buckling_stress(t_net_mm=5.0, s_m=0.6, psi=1.0)
    assert se == pytest.approx(51.5, abs=0.5)
    assert critical_buckling_stress(se, 235.0) == se


def test_plate_buckling_check_pass_fail():
    """판정 σ_c ≥ σ_a: 두꺼운 판은 여유(합격), 얇은 판+큰 응력은
    좌굴 불합격 (증육 필요 신호)."""
    # 두꺼운 판 t 14mm·s 0.6·σ_a 150 → σ_c 여유
    ok = plate_buckling_check(t_net_mm=14.0, s_m=0.6,
                              sigma_a_nmm2=150.0, sigma_f_nmm2=235.0)
    assert ok["passed"] and ok["sigma_c_nmm2"] >= 150.0
    # 얇은 판 t 6mm·s 0.8·σ_a 160 → 좌굴 불합격
    bad = plate_buckling_check(t_net_mm=6.0, s_m=0.8,
                               sigma_a_nmm2=160.0, sigma_f_nmm2=235.0)
    assert not bad["passed"] and bad["sigma_c_nmm2"] < 160.0


def test_gate_reports_buckling():
    """게이트 통합 — 종강도 리포트에 좌굴 성적표 병기 (게이트
    승격 아님 — 증육 루프 연동 백로그, CII 선례 정직)."""
    from src.physics.structure.materials import MATERIALS
    from src.physics.structure.strength import longitudinal_strength
    r = longitudinal_strength(
        loa=100.0, beam=16.2, depth=8.5, draft=5.9,
        m_still_knm=-55000.0, m_wave_hog_knm=190000.0,
        m_wave_sag_knm=-255000.0, material=MATERIALS["mild_steel"])
    assert "buckling" in r
    b = r["buckling"]
    assert "sigma_c_nmm2" in b and "sigma_a_nmm2" in b
    assert isinstance(b["passed"], bool)
    assert "성적표" in b["note"] or "게이트" in b["note"]
