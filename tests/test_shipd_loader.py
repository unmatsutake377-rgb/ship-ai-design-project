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


def test_all_hulls_normalized_loa10():
    """Ship-D 정규화 확인: 전 척 LOA=10 — 스케일 전이 전제."""
    vectors, labels = shipd_loader.load_vectors()
    assert labels[0] == "LOA"
    assert np.allclose(vectors[:, 0], 10.0)


def test_scaled_hull_passes_our_hydrostatics():
    """호환 실증: Ship-D 형상 × 우리 정역학 — 갑판 열린 메쉬여도
    수선하 절단 기반 계산이 정상 동작 (2026-07-27 파일럿 20척 검증됨)."""
    from src.physics.hydrostatics import equilibrium_draft, kb_bm

    vectors, _ = shipd_loader.load_vectors()
    mesh = shipd_loader.scaled_mesh(vectors[18750], target_loa=3.0)
    draft = equilibrium_draft(mesh, 150.0)
    assert 0 < draft < 0.9 * mesh.bounds[1][2]
    kb, bm = kb_bm(mesh, draft)
    assert kb > 0 and bm > 0
