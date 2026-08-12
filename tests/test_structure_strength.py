"""종강도 판정·증육 수렴 (구조 3단계, 스펙 §2 strength)."""
import pytest

from src.physics.structure.materials import MATERIALS


def _loads_100m():
    """100m 화물선 대표 하중 [kN·m] — 1단계 실측 자릿수."""
    from src.physics.structure.wave_loads import iacs_wave_bending_knm
    hog, sag = iacs_wave_bending_knm(100.0, 15.0, 0.75)
    return 0.5 * hog, hog, sag        # 정수 개략 = 호깅의 절반


def test_100m_rule_thickness_passes_quickly():
    """규칙 두께가 거의 그대로 합격 (2단계 비 1.06 실증 재확인) —
    증육 2회 이하."""
    from src.physics.structure.strength import longitudinal_strength
    m_still, hog, sag = _loads_100m()
    r = longitudinal_strength(100.0, 15.0, 8.0, 5.5, m_still, hog, sag,
                              MATERIALS["mild_steel"])
    assert r["passed"] is True
    assert r["iterations"] <= 2
    assert r["z_deck_m3"] >= r["z_required_m3"]
    assert r["z_keel_m3"] >= r["z_required_m3"]


def test_oversized_moment_forces_thickening():
    """인위 3배 모멘트 → 증육 반복 후 합격, 두께 > 규칙 시작값."""
    from src.physics.structure.strength import longitudinal_strength
    m_still, hog, sag = _loads_100m()
    base = longitudinal_strength(100.0, 15.0, 8.0, 5.5, m_still, hog,
                                 sag, MATERIALS["mild_steel"])
    big = longitudinal_strength(100.0, 15.0, 8.0, 5.5, 3.0 * m_still,
                                3.0 * hog, 3.0 * sag,
                                MATERIALS["mild_steel"])
    assert big["passed"] is True
    assert big["iterations"] > 0
    assert big["t_bottom_mm"] > base["t_bottom_mm"] - 1e-9
    assert big["t_deck_mm"] > base["t_deck_mm"]


def test_impossible_moment_honest_failure():
    """수렴 한도 초과 → passed=False 정직 반환 (예외 아님)."""
    from src.physics.structure.strength import longitudinal_strength
    m_still, hog, sag = _loads_100m()
    r = longitudinal_strength(100.0, 15.0, 8.0, 5.5, 100.0 * m_still,
                              100.0 * hog, 100.0 * sag,
                              MATERIALS["mild_steel"], max_iter=5)
    assert r["passed"] is False
    assert "수렴" in r["note"] or "한도" in r["note"]


def test_small_alu_path():
    """3m 알루 USV — 허용응력 = ISO 설계응력 σ_d 88 (KS V ISO
    12215-5 정본, 이전 175·f1=93 과대 정정)."""
    from src.physics.structure.strength import longitudinal_strength
    r = longitudinal_strength(3.0, 1.2, 0.5, 0.3, 0.5, 1.0, -1.0,
                              MATERIALS["al5083"])
    assert r["passed"] is True
    assert r["sigma_allow_nmm2"] == pytest.approx(88.0, abs=0.5)
