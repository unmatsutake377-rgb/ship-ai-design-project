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


def test_explicit_propulsion_mass_replaces_fraction():
    """설계 나선용: 실측 추진계 중량 지정 시 고정비율 미사용."""
    est = estimate_weights(hull_area_m2=10.0, depth=0.5, payload_kg=50.0,
                           propulsion_mass_kg=4.5)
    assert est.propulsion_mass == 4.5
    assert est.total_mass == pytest.approx(
        est.structure_mass + 50.0 + 4.5, rel=1e-9
    )


def test_lcg_izz_computed():
    """분포모델 (오너 Q4): 추진계만 선미 → LCG 약간 선미, Izz 양수."""
    est = estimate_weights(hull_area_m2=12.0, depth=0.5, payload_kg=100.0,
                           propulsion_mass_kg=4.5, loa=4.0)
    assert -0.2 < est.lcg < 0.0
    assert est.izz > 0
    assert est.izz >= 4.5 * (0.45 * 4.0) ** 2 * 0.9


def test_trim_warning_flag():
    """무거운 추진계가 선미에 몰리면 트림 경고."""
    est = estimate_weights(12.0, 0.5, 100.0, propulsion_mass_kg=40.0, loa=4.0)
    assert est.trim_warning


def test_legacy_call_without_loa_degenerates_safely():
    est = estimate_weights(12.0, 0.5, 100.0)
    assert est.lcg == 0.0
    assert not est.trim_warning
