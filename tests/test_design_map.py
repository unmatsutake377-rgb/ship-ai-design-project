"""설계 지도 — 평가·추천 (스펙 2026-08-09-design-map)."""
import pytest


def test_recommend_picks_cheapest_passing():
    """추천 = 전 게이트 합격 중 수송단가 최소 (동률 시 EEDI 여유)."""
    from src.ai.design_map import recommend
    results = [
        {"passed": True, "transport_usd_per_tnm": 2.0e-3,
         "eedi_margin_pct": 5.0, "speed_ms": 6.0, "payload_kg": 8e6},
        {"passed": True, "transport_usd_per_tnm": 1.5e-3,
         "eedi_margin_pct": 2.0, "speed_ms": 5.5, "payload_kg": 8e6},
        {"passed": False, "transport_usd_per_tnm": 1.0e-3,
         "eedi_margin_pct": -10.0, "speed_ms": 7.0,
         "payload_kg": 8e6},
        {"passed": None, "transport_usd_per_tnm": None,
         "eedi_margin_pct": None, "speed_ms": 7.0,
         "payload_kg": 20e6, "error": "엔진 한계"},
    ]
    best = recommend(results)
    assert best["transport_usd_per_tnm"] == pytest.approx(1.5e-3)
    assert best["passed"] is True


def test_recommend_none_when_all_fail():
    """전부 불합격 → 정직 None."""
    from src.ai.design_map import recommend
    assert recommend([{"passed": False}, {"passed": None}]) is None


def test_evaluate_design_fast_smoke(tmp_path):
    """fast 평가 1점 스모크 — 핵심 지표 존재·자릿수 (8000t·7 m/s).

    EPL 판정 승급 후 정정: 13.6kn은 **박빙** (여유 한 자릿수 %) —
    설치 MCR 기준은 불합격 병기 (attained_installed)."""
    from src.ai.design_map import evaluate_design
    r = evaluate_design(8_000_000.0, 7.0, fast=True)
    assert r["loa_m"] > 100.0
    assert -5.0 < r["eedi_margin_pct"] < 10.0    # 박빙 대역
    assert r["transport_usd_per_tnm"] > 0
    assert r["cii_rating_2026"] in "ABCDE"
    assert r["passed"] in (True, False)


def test_evaluate_design_slower_passes(tmp_path):
    """감속 6.0 m/s → EEDI 합격 방향 (1단계 민감도 재현) +
    수송단가 하락 (V² 성질 지도 차원 확인)."""
    from src.ai.design_map import evaluate_design
    fast7 = evaluate_design(8_000_000.0, 7.0, fast=True)
    fast6 = evaluate_design(8_000_000.0, 6.0, fast=True)
    assert fast6["eedi_margin_pct"] > fast7["eedi_margin_pct"]
    assert fast6["eedi_margin_pct"] > 0.0
    assert (fast6["transport_usd_per_tnm"]
            < fast7["transport_usd_per_tnm"])
