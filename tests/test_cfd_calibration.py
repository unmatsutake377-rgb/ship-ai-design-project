"""Michell 보정 계수 — 앵커드 최소제곱 (손계산 정답지)."""
import pandas as pd
import pytest

from src.cfd.calibration import fit_wave_ratio, ratios_from_labels, wave_ratio


def test_fit_recovers_known_slope():
    """ratio = 1 - 1.6·(B/L) 인 합성 점 → b=-1.6 정확 복원."""
    pts = [(x, 1 - 1.6 * x) for x in (0.1, 0.25, 0.5)]
    assert fit_wave_ratio(pts) == pytest.approx(-1.6)


def test_fit_least_squares_hand_calc():
    """잡음 점: b = Σ(r-1)x / Σx² 손계산과 일치."""
    pts = [(0.5, 0.2), (0.1, 0.9)]
    b_hand = ((0.2 - 1) * 0.5 + (0.9 - 1) * 0.1) / (0.5**2 + 0.1**2)
    assert fit_wave_ratio(pts) == pytest.approx(b_hand)


def test_anchor_at_zero():
    """B/L=0 (무한히 얇음) → ratio=1: Michell이 정확한 극한."""
    assert wave_ratio(0.0, b=-1.6) == 1.0


def test_clipping():
    assert wave_ratio(1.0, b=-5.0) == 0.05    # 하한
    assert wave_ratio(1.0, b=+5.0) == 1.5     # 상한


def test_ratios_from_labels_pairs_modes():
    df = pd.DataFrame([
        {"case_name": "wigley_lb4_simple_1.85ms", "cfd_pressure_n": 4.0,
         "emp_rw_n": 10.0, "loa_m": 3.0, "beam_m": 0.75},
        {"case_name": "wigley_lb4_inter_1.85ms", "cfd_pressure_n": 9.0,
         "emp_rw_n": 10.0, "loa_m": 3.0, "beam_m": 0.75},
    ])
    pts = ratios_from_labels(df)
    assert len(pts) == 1
    assert pts[0] == pytest.approx((0.25, 0.5))   # (B/L, (9-4)/10)
