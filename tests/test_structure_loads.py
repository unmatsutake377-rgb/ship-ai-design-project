"""구조 강도 1단계 — 하중 곡선 시험 (스펙 2026-08-09 §5-1)."""
import numpy as np
import pytest
import trimesh

G = 9.81
RHO = 1025.0


def test_weight_blocks_closure():
    """성분 블록 합 = 총중량 (폐합 항등식)."""
    from src.physics.structure.loads import (
        standard_weight_blocks,
        weight_linear_density,
    )
    comp = {"structure": 800.0, "outfit": 200.0, "machinery": 300.0,
            "fuel": 100.0, "payload": 600.0}
    blocks = standard_weight_blocks(comp, xmin=-40.0, loa=80.0)
    xs = np.linspace(-40.0, 40.0, 201)
    w = weight_linear_density(xs, blocks)
    total = np.trapezoid(w, xs)
    assert total == pytest.approx(sum(comp.values()) * G, rel=1e-9)
    assert np.all(w >= 0)


def test_weight_blocks_placement():
    """기관·연료 = 선미 구간, 화물 = 중앙 구간 (통상 배치)."""
    from src.physics.structure.loads import standard_weight_blocks
    blocks = standard_weight_blocks(
        {"machinery": 100.0, "payload": 100.0}, xmin=0.0, loa=100.0)
    named = {}
    for (m, x0, x1), name in zip(blocks, ["machinery", "payload"]):
        named[name] = (x0, x1)
    m0, m1 = named["machinery"]
    p0, p1 = named["payload"]
    assert m1 <= 30.0          # 기관실 = 선미 30% 안
    assert 20.0 <= p0 and p1 <= 90.0   # 화물창 = 중앙부
