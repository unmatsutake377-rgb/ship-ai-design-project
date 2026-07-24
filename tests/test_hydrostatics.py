import numpy as np
import pytest
import trimesh

from src.physics.hydrostatics import (
    SinksError,
    StabilityCriteria,
    equilibrium_draft,
    evaluate,
    immersed_volume,
    kb_bm,
    waterplane_properties,
)

L, B, D = 4.0, 1.2, 0.6
RHO = 1025.0


def barge() -> trimesh.Trimesh:
    """직육면체 바지선: 해석해 존재 (spec §4). z∈[0,D], x·y 중심 원점."""
    box = trimesh.creation.box(extents=[L, B, D])
    box.apply_translation([0, 0, D / 2])
    return box


def test_immersed_volume_half_draft():
    t = 0.3
    assert immersed_volume(barge(), t) == pytest.approx(L * B * t, rel=1e-6)


def test_waterplane_properties_analytic():
    aw, ixx = waterplane_properties(barge(), 0.3)
    assert aw == pytest.approx(L * B, rel=1e-6)
    assert ixx == pytest.approx(L * B ** 3 / 12.0, rel=1e-6)


def test_kb_bm_analytic():
    t = 0.3
    kb, bm = kb_bm(barge(), t)
    assert kb == pytest.approx(t / 2, rel=1e-6)
    assert bm == pytest.approx(B ** 2 / (12.0 * t), rel=1e-3)


def test_equilibrium_draft_analytic():
    t_target = 0.25
    mass = RHO * L * B * t_target
    t = equilibrium_draft(barge(), mass)
    assert t == pytest.approx(t_target, abs=1e-4)


def test_sinks_error():
    too_heavy = RHO * L * B * D * 1.5
    with pytest.raises(SinksError):
        equilibrium_draft(barge(), too_heavy)


def test_evaluate_pass_and_report_fields():
    t = 0.25
    mass = RHO * L * B * t
    kg = 0.30  # KB=0.125, BM=0.48 → GM=0.305, GM/B=0.254 → 밴드 내
    report = evaluate(barge(), mass, kg, beam=B, depth=D)
    assert report.passed
    assert report.checks["displacement"]
    assert report.checks["gm_band"]
    assert report.checks["freeboard"]
    assert report.gm == pytest.approx(0.125 + 0.48 - 0.30, abs=1e-3)


def test_evaluate_fails_when_kg_too_high():
    t = 0.25
    mass = RHO * L * B * t
    report = evaluate(barge(), mass, kg=0.65, beam=B, depth=D)
    assert not report.passed
    assert not report.checks["gm_band"]
