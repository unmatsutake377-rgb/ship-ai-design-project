"""EEDI 정본 — 손계산·기준선 앵커 (경제성 1단계, 스펙 §3)."""
import pytest


def test_reference_line_hand_calc():
    """기준선 a·DWT^{-c} 손계산 — 원전 Table 2 (General cargo
    a=107.48, c=0.216). DWT 10,000 → 107.48·10000^-0.216."""
    from src.physics.economics.eedi import required_eedi
    r = required_eedi(10_000.0, phase=3)
    ref = 107.48 * 10_000.0 ** (-0.216)
    assert r["reference_g_per_tnm"] == pytest.approx(ref, rel=1e-9)
    assert 14.0 < ref < 15.5          # 자릿수 sanity (문헌 대역)


def test_reduction_factor_bands():
    """감축률 — 원전 Table 1: 15,000+ = 30%, 3,000~15,000 선형
    0→30, <3,000 적용 밖."""
    from src.physics.economics.eedi import required_eedi
    assert required_eedi(20_000.0)["reduction_pct"] == pytest.approx(30.0)
    mid = required_eedi(9_000.0)["reduction_pct"]
    assert 0.0 < mid < 30.0
    assert required_eedi(9_000.0)["applicable"] is True
    small = required_eedi(2_000.0)
    assert small["applicable"] is False


def test_attained_hand_calc():
    """attained 손계산 — MCR 4640 kW·SFOC 178.8·DWT 8000·
    설계 7 m/s @ 2900 kW: P75 = 3480 kW, Vref = 7·(3480/2900)^⅓,
    EEDI = 3.206·178.8·P75 / (8000·Vref_kn)  [g/(t·nm)]."""
    from src.physics.economics.eedi import attained_eedi
    r = attained_eedi(mcr_kw=4640.0, sfoc_g_per_kwh=178.8,
                      dwt_t=8000.0, v_service_ms=7.0,
                      p_service_kw=2900.0)
    p75 = 0.75 * 4640.0
    v_ref_ms = 7.0 * (p75 / 2900.0) ** (1.0 / 3.0)
    v_ref_kn = v_ref_ms * 3600.0 / 1852.0
    expected = (3.206 * 178.8 * p75) / (8000.0 * v_ref_kn)
    assert r["eedi_g_per_tnm"] == pytest.approx(expected, rel=1e-9)
    assert r["v_ref_kn"] == pytest.approx(v_ref_kn, rel=1e-9)
    assert 5.0 < r["eedi_g_per_tnm"] < 40.0   # 일반화물선 자릿수


def test_verdict():
    """판정 — attained ≤ required 합격·여유 % 부호."""
    from src.physics.economics.eedi import eedi_verdict
    v = eedi_verdict(10.0, 12.0)
    assert v["passed"] is True
    assert v["margin_pct"] == pytest.approx(100.0 * (12.0 - 10.0) / 12.0)
    assert eedi_verdict(13.0, 12.0)["passed"] is False
