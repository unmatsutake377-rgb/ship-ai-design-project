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


def test_gate_hard_buckling_converges():
    """게이트 승격 — 좌굴이 증육 수렴 루프에 하드 연동. 100m
    화물선이 항복+좌굴 둘 다 통과할 때까지 갑판·선저 증육 →
    passed=True·양 판 좌굴 합격. 항복만이던 시절(갑판 8mm)보다
    지배 압축판이 두꺼워짐 (좌굴이 항복보다 먼저 지배 실증)."""
    from src.physics.structure.materials import MATERIALS
    from src.physics.structure.strength import longitudinal_strength
    r = longitudinal_strength(
        loa=100.0, beam=16.2, depth=8.5, draft=5.9,
        m_still_knm=-55000.0, m_wave_hog_knm=190000.0,
        m_wave_sag_knm=-255000.0, material=MATERIALS["mild_steel"])
    assert r["passed"] is True
    b = r["buckling"]
    # 하드 게이트 = 갑판·선저 압축판 좌굴 각각 통과
    assert b["deck"]["passed"] and b["bottom"]["passed"]
    # 좌굴이 항복보다 먼저 지배 → 압축판이 항복 전용(8mm)보다 두꺼움
    assert r["t_bottom_mm"] > 8.0 or r["t_deck_mm"] > 8.0


def test_gate_buckling_correct_compression_side():
    """물리 방향 — 표준 보 부호: 새깅=갑판 압축·호깅=선저 압축
    (백지 리뷰 [상] 부호 반전 검거 반영). |새깅 −310000| >
    |호깅 135000| → 갑판 압축이 더 커 갑판 좌굴이 지배."""
    from src.physics.structure.materials import MATERIALS
    from src.physics.structure.strength import longitudinal_strength
    r = longitudinal_strength(
        loa=100.0, beam=16.2, depth=8.5, draft=5.9,
        m_still_knm=-55000.0, m_wave_hog_knm=190000.0,
        m_wave_sag_knm=-255000.0, material=MATERIALS["mild_steel"])
    b = r["buckling"]
    # 새깅(갑판 압축)이 호깅(선저 압축)보다 크다 → 갑판 σ_a 지배
    assert b["deck"]["sigma_a_nmm2"] > b["bottom"]["sigma_a_nmm2"]
