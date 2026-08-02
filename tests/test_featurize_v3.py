"""특징 v3 — 평형 정합 + 다충실도 (스펙 2026-08-03 §4)."""
import numpy as np
import pytest
import trimesh

from src.featurize_shipd import FEATURE_NAMES, N_FEATURES, hull_features


def test_feature_count_and_names():
    assert N_FEATURES == 32
    assert len(FEATURE_NAMES) == 32
    assert FEATURE_NAMES[-2:] == ["r_wave_lo", "r_michell_lo"]


def test_box_hull_equilibrium_and_lowres():
    """상자 껍데기 손계산: 평형 흘수 = m/(ρLB) (무게 모델 질량 기준),
    저해상 특징은 유한·양수 (마찰 포함 총저항 > 조파)."""
    from src.physics.weights import estimate_weights

    hull = trimesh.creation.box(extents=[3.0, 1.0, 0.5])
    hull.apply_translation([1.5, 0.0, 0.25])
    f = hull_features(hull)
    names = list(FEATURE_NAMES)
    t_eq = f[names.index("t_eq")]
    mass = estimate_weights(float(hull.area), 0.5, 100.0).total_mass
    assert t_eq == pytest.approx(mass / (1025.0 * 3.0 * 1.0), rel=0.05)
    rw = f[names.index("r_wave_lo")]
    rt = f[names.index("r_michell_lo")]
    assert np.isfinite(rw) and np.isfinite(rt)
    assert rt > rw >= 0.0


def test_lowres_matches_fullres_order_of_magnitude():
    """저해상 vs 정해상: 같은 배에서 비율 0.5~2 대역 (자릿수 일치)."""
    from src.physics.resistance import total_resistance_mesh

    hull = trimesh.creation.box(extents=[3.0, 1.0, 0.5])
    hull.apply_translation([1.5, 0.0, 0.25])
    f = hull_features(hull)
    names = list(FEATURE_NAMES)
    t_eq = float(f[names.index("t_eq")])
    full = total_resistance_mesh(hull, 3.0, t_eq, 1.2)
    ratio = f[names.index("r_michell_lo")] / full.total
    assert 0.5 < ratio < 2.0
