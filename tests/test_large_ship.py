"""대형 강선 법칙 검증 (전 크기 개방 2단계, 스펙 2026-08-06 §3)."""
import pytest

from src.physics.large_ship import (
    IMO_GM_MIN_M,
    equipment_numeral,
    icll_freeboard_m,
    large_gm_band,
    watson_lightship,
)


def test_equipment_numeral_hand_calc():
    """E 손계산: 230·(32.2+10.8) + 0.85·230·(19−10.8) = 11,493."""
    e = equipment_numeral(230.0, 32.2, 19.0, 10.8)
    assert e == pytest.approx(230 * 43.0 + 0.85 * 230 * 8.2, rel=1e-9)
    assert e == pytest.approx(11493.1, abs=0.5)


def test_watson_lightship_kcs_class_magnitude():
    """KCS급(230 m 컨테이너선) 경하 자릿수: 만 톤대 (문헌 계보
    2만 t 근방 — 개략식 대역 [1e4, 3e4] t)."""
    ls = watson_lightship(230.0, 32.2, 19.0, 10.8, cb=0.65,
                          mcr_kw=25000.0, ship_type="container")
    assert 1.0e4 < ls.total_t < 3.0e4
    assert ls.structure_t > ls.outfit_t > ls.machinery_t > 0


def test_watson_cb_prime_correction_direction():
    """Cb 풍만할수록 구조 중량 증가 (보정항 부호 검사)."""
    lean = watson_lightship(100.0, 16.4, 8.2, 5.9, cb=0.60, mcr_kw=3000.0)
    full = watson_lightship(100.0, 16.4, 8.2, 5.9, cb=0.80, mcr_kw=3000.0)
    assert full.structure_t > lean.structure_t


def test_icll_freeboard_interpolation():
    """ICLL Type B: 표점 재현 + 중간 보간 + 단조 증가."""
    assert icll_freeboard_m(100.0) == pytest.approx(1.271, abs=1e-6)
    assert icll_freeboard_m(230.0) == pytest.approx(
        (3264 + (3883 - 3264) * 30 / 50) / 1000.0, abs=1e-6)
    assert icll_freeboard_m(20.0) == pytest.approx(0.200)
    fs = [icll_freeboard_m(x) for x in (24, 60, 120, 240, 350)]
    assert all(a < b for a, b in zip(fs, fs[1:]))


def test_large_gm_band_imo_floor():
    """대형 GM 밴드: 하한 = 0.15 m 절대 (B 30 m → GM/B 0.005),
    소형 밴드 하한(0.04)보다 느슨 — IMO 규정 채택 근거."""
    lo, hi = large_gm_band(30.0)
    assert lo == pytest.approx(IMO_GM_MIN_M / 30.0)
    assert lo < 0.04
    assert hi == 0.40


def test_cargo_dimension_estimate_hand_calc():
    """cargo 치수 손계산: 짐 5,000 t, fraction 0.70 → Δ 7,143 t →
    ∇ 6,969 m³ → L = (∇·lb²·bt/cb)^(1/3) ≈ 98.9 m (실선 정합:
    DWT 5000t급 일반화물선 ~100 m)."""
    from src.ai.dimension_estimator import estimate_dimensions
    from src.core.types import GoalSpec

    goal = GoalSpec(target_speed_ms=7.0, payload_kg=5_000_000.0,
                    purpose="cargo", endurance_h=240.0)
    dims = estimate_dimensions(goal)
    assert dims.loa == pytest.approx(98.9, rel=0.02)
    assert dims.loa / dims.beam == pytest.approx(6.1, rel=1e-6)
    assert dims.cb == pytest.approx(0.75)
