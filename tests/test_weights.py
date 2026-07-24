import pytest

from src.physics.weights import (
    AREAL_DENSITY_KG_M2,
    OUTFIT_FACTOR,
    PROPULSION_FRACTION,
    estimate_weights,
)


def test_total_is_self_consistent():
    """W = 구조 + 추진 + 적재가 정확히 닫혀야 함 (spec §4 자기일관성)."""
    est = estimate_weights(hull_area_m2=12.0, depth=0.5, payload_kg=100.0)
    assert est.total_mass == pytest.approx(
        est.structure_mass + est.propulsion_mass + est.payload_mass, rel=1e-9
    )
    # 추진·배터리 비율 확인
    assert est.propulsion_mass == pytest.approx(
        PROPULSION_FRACTION * est.total_mass, rel=1e-9
    )


def test_structure_formula():
    est = estimate_weights(hull_area_m2=10.0, depth=0.5, payload_kg=50.0)
    assert est.structure_mass == pytest.approx(
        10.0 * AREAL_DENSITY_KG_M2 * OUTFIT_FACTOR, rel=1e-9
    )


def test_kg_within_hull():
    est = estimate_weights(hull_area_m2=12.0, depth=0.5, payload_kg=100.0)
    assert 0.0 < est.kg < 0.5  # KG는 킬~형심 사이


def test_assumptions_recorded():
    est = estimate_weights(hull_area_m2=12.0, depth=0.5, payload_kg=100.0)
    for key in ("areal_density", "outfit_factor", "propulsion_fraction",
                "vcg_structure", "vcg_payload", "vcg_propulsion"):
        assert key in est.assumptions


def test_payload_zero_still_positive_weight():
    est = estimate_weights(hull_area_m2=12.0, depth=0.5, payload_kg=0.0)
    assert est.total_mass > 0
