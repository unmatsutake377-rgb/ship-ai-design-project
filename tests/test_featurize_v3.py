"""특징 v3 — 평형 정합 + 다충실도 (스펙 2026-08-03 §4)."""
import numpy as np
import pytest
import trimesh

from src.featurize_shipd import FEATURE_NAMES, N_FEATURES, hull_features


def test_feature_count_and_names():
    assert N_FEATURES == 34
    assert len(FEATURE_NAMES) == 34
    assert FEATURE_NAMES[30:32] == ["r_wave_lo", "r_michell_lo"]


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


def test_stability_full_features_hand_calc():
    """v3.1 안정 다충실도 손계산 (상자 껍데기).

    gmb_full = (KB + BM − KG)/beam — KB=t/2(상자), BM=(LB³/12)/∇,
    KG = 무게 모델 실계산 (0.65D 고정 개략 폐기)."""
    import trimesh

    from src.featurize_shipd import FEATURE_NAMES, hull_features
    from src.physics.weights import estimate_weights

    L, B, D = 3.0, 1.0, 0.5
    hull = trimesh.creation.box(extents=[L, B, D])
    hull.apply_translation([L / 2, 0.0, D / 2])
    f = dict(zip(FEATURE_NAMES, hull_features(hull)))

    w = estimate_weights(float(hull.area), D, 100.0)
    vol = w.total_mass / 1025.0
    t = w.total_mass / (1025.0 * L * B)
    gmb_expected = (t / 2 + (L * B**3 / 12) / vol - w.kg) / B
    assert f["gmb_full"] == pytest.approx(gmb_expected, rel=0.05)
    assert f["stab_margin_lo"] == pytest.approx(
        min(gmb_expected - 0.04, 0.40 - gmb_expected), rel=0.05)


def test_feature_count_v31():
    from src.featurize_shipd import FEATURE_NAMES, N_FEATURES

    assert N_FEATURES == 34
    assert FEATURE_NAMES[-2:] == ["gmb_full", "stab_margin_lo"]
