"""CII 운항 탄소등급 — 손계산·등급 경계 앵커 (스펙 2026-08-09-cii)."""
import pytest


def test_attained_aer_hand_calc():
    """AER 손계산 (G1): 연료 1,944 t/년 × CF 3.206 → CO₂ 6,232 t,
    ÷ (8,000 DWT × 거리) [gCO₂/(DWT·nm)]."""
    from src.physics.economics.cii import attained_aer
    dist = 136_500.0                     # nm/년
    r = attained_aer(1944.0, 8000.0, dist)
    co2_g = 1944.0 * 3.206 * 1e6
    assert r["aer_g_per_dwt_nm"] == pytest.approx(
        co2_g / (8000.0 * dist), rel=1e-9)


def test_required_cii_reference_and_z():
    """기준선 손계산 (G2: <20k a=588·c=0.3885) + Z 연도 강화
    (G3: 2023 5% → 2026 11%) — required 단조 감소."""
    from src.physics.economics.cii import required_cii
    ref = 588.0 * 8000.0 ** (-0.3885)
    r23 = required_cii(8000.0, 2023)
    r26 = required_cii(8000.0, 2026)
    assert r23["reference_g_per_dwt_nm"] == pytest.approx(ref, rel=1e-9)
    assert r23["required_g_per_dwt_nm"] == pytest.approx(ref * 0.95,
                                                        rel=1e-9)
    assert r26["required_g_per_dwt_nm"] == pytest.approx(ref * 0.89,
                                                        rel=1e-9)
    assert r26["required_g_per_dwt_nm"] < r23["required_g_per_dwt_nm"]


def test_required_cii_2027_honest_undefined():
    """2027+ Z 미정 (원전 각주) — 정직 표기."""
    from src.physics.economics.cii import required_cii
    r = required_cii(8000.0, 2028)
    assert r["z_defined"] is False
    assert "미정" in r["note"]


def test_rating_boundaries():
    """등급 경계 손계산 (G4: General cargo 0.83/0.94/1.06/1.19) —
    비 0.80→A, 0.90→B, 1.00→C, 1.10→D, 1.25→E."""
    from src.physics.economics.cii import cii_rating
    req = 10.0
    assert cii_rating(8.0, req)["rating"] == "A"
    assert cii_rating(9.0, req)["rating"] == "B"
    assert cii_rating(10.0, req)["rating"] == "C"
    assert cii_rating(11.0, req)["rating"] == "D"
    assert cii_rating(12.5, req)["rating"] == "E"


def test_outlook_degrades_over_years():
    """같은 배·같은 운항 → 해가 갈수록 등급 악화 방향 (Z 강화)."""
    from src.physics.economics.cii import cii_outlook
    out = cii_outlook(attained_g_per_dwt_nm=9.0, dwt_t=8000.0)
    years = [o["year"] for o in out]
    assert years == [2023, 2024, 2025, 2026]
    ratings = [o["rating"] for o in out]
    # 등급 문자 단조 (A<B<...) — 악화 또는 유지
    assert all(a <= b for a, b in zip(ratings, ratings[1:]))
