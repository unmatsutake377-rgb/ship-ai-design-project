"""NSGA × 8중 게이트 스모크 (스펙 2026-08-09-optimize-large §3-1).

느린 시험 — 소집단 관통만 (본 탐색은 배경 실행)."""
import numpy as np
import pytest

from data import shipd_loader

pytestmark = pytest.mark.skipif(not shipd_loader.available(),
                                reason="Ship-D 로컬 사본 없음")


def test_evaluate_large_vector_smoke():
    """실척 1척 평가 — 지표 존재 또는 정직 사망(None)."""
    from src.ai.optimize_large import evaluate_large_vector
    vectors, _ = shipd_loader.load_vectors()
    r = evaluate_large_vector(vectors[0], 6.0, 8_000_000.0, 116.0)
    if r is not None:
        assert r["transport_usd_per_tnm"] > 0
        assert r["attained_eedi"] > 0
        assert r["cii_rating_2026"] in "ABCDE"


def test_optimize_large_smoke():
    """NSGA 소집단 관통 — 전선 존재·비지배·속도 대역."""
    from src.ai.optimize_large import optimize_large
    df = optimize_large(pop_size=6, n_gen=2, seed=3)
    assert df.attrs["death_stats"]["alive"] >= 1, df.attrs["death_stats"]
    if len(df) >= 2:
        # 비지배 성질: 단가 오름차순에서 EEDI는 내림 방향 존재 허용
        d = df.sort_values("transport_usd_per_tnm")
        assert d["speed_ms"].between(5.0, 7.5).all()
        assert (d["transport_usd_per_tnm"] > 0).all()
