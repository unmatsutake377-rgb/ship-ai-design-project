"""Ship-D 로더 테스트 — 로컬 사본 없으면 전체 skip (fresh clone 안전)."""
import numpy as np
import pytest

from data import shipd_loader

pytestmark = pytest.mark.skipif(
    not shipd_loader.available(),
    reason="Ship-D 로컬 사본 없음 (data/shipd/)",
)


def test_vectors_shape_and_labels():
    vectors, labels = shipd_loader.load_vectors()
    assert vectors.shape == (30000, 45)
    assert len(labels) == 45
    assert labels[0] == "LOA"
    assert np.isfinite(vectors).all()


def test_reconstruct_one_hull():
    """벡터 1개 → 메쉬 재구성. 부피 양수 + 유의미한 정점 수."""
    vectors, _ = shipd_loader.load_vectors()
    mesh = shipd_loader.reconstruct_mesh(vectors[0])
    assert len(mesh.vertices) > 500
    assert mesh.volume > 0
