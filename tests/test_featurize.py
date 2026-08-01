"""특징 공학 검증 — 상자 바지선은 거의 모든 특징의 해석 정답을 줌."""
import numpy as np
import pytest
import trimesh

from src.featurize_shipd import FEATURE_NAMES, N_FEATURES, hull_features

L, B, D = 4.0, 1.0, 0.5


@pytest.fixture(scope="module")
def barge_feats():
    barge = trimesh.creation.box(bounds=[[-L / 2, -B / 2, 0], [L / 2, B / 2, D]])
    f = hull_features(barge)
    return dict(zip(FEATURE_NAMES, f))


def test_feature_count():
    assert len(FEATURE_NAMES) == N_FEATURES == 22


def test_barge_global_features(barge_feats):
    f = barge_feats
    assert f["beam"] == pytest.approx(B)
    assert f["depth"] == pytest.approx(D)
    assert f["vol_total"] == pytest.approx(L * B * D)
    assert f["fill"] == pytest.approx(1.0)          # 상자는 꽉 참
    assert f["l_over_b"] == pytest.approx(4.0)


def test_barge_draft_features(barge_feats):
    f = barge_feats
    for frac in (30, 50, 70):
        t = frac / 100 * D
        assert f[f"vol_t{frac}"] == pytest.approx(L * B * t, rel=0.02)
        assert f[f"awp_t{frac}"] == pytest.approx(L * B, rel=0.02)
        # 수선면 관성모멘트 (횡): L·B³/12
        assert f[f"ixx_t{frac}"] == pytest.approx(L * B**3 / 12, rel=0.05)


def test_offcenter_barge_form_coefficients():
    """x가 0~L 범위인 상자 (Ship-D 좌표 관례) — x=0 고정 절단이면
    뱃머리 끝을 자르게 돼 NaN 나던 실측 버그의 회귀 시험."""
    barge = trimesh.creation.box(bounds=[[0, -B / 2, 0], [L, B / 2, D]])
    f = dict(zip(FEATURE_NAMES, hull_features(barge)))
    assert f["cm"] == pytest.approx(1.0, rel=0.02)
    assert f["cp"] == pytest.approx(1.0, rel=0.02)


def test_barge_form_coefficients(barge_feats):
    f = barge_feats
    assert f["cm"] == pytest.approx(1.0, rel=0.02)   # 상자 단면 = 꽉 참
    assert f["cp"] == pytest.approx(1.0, rel=0.02)
    assert f["lcb_frac"] == pytest.approx(0.5, abs=0.02)  # 중앙


def test_wigley_features_sane():
    """Wigley: 전부 유한 + 계수들이 상자보다 작아야 (날씬하니까)."""
    from src.ai.hull_generator import generate_hull_mesh
    from src.core.types import MainDimensions

    dims = MainDimensions(loa=3.0, beam=0.75, depth=0.5,
                          draft_design=0.3, cb=0.444)
    f = dict(zip(FEATURE_NAMES, hull_features(generate_hull_mesh(dims))))
    assert all(np.isfinite(v) for v in f.values())
    assert f["fill"] < 0.8
    assert f["cp"] < 0.9
