"""대리모델 테스트 — 합성 데이터로 학습 능력 검증 (Ship-D 불필요, 빠름)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ai.surrogate import train_surrogate


@pytest.fixture(scope="module")
def synthetic():
    """학습 가능성 검증용 합성 문제: 타당성·목적 모두 입력의 단순 함수."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(400, 45))
    y_feas = (x[:, 0] + x[:, 1] > 0).astype(float)
    y_obj = np.column_stack([
        2.0 * x[:, 2] + 1.0,
        -1.5 * x[:, 3] + x[:, 4],
        0.5 * x[:, 5],
    ])
    y_obj[y_feas == 0] = np.nan  # infeasible은 라벨 없음 (실전과 동일)
    return x, y_feas, y_obj


@pytest.fixture(scope="module")
def trained(synthetic):
    x, y_feas, y_obj = synthetic
    return train_surrogate(x, y_feas, y_obj, epochs=300, seed=1)


def test_learns_feasibility(trained):
    _, metrics = trained
    assert metrics["feas_accuracy"] > 0.8


def test_learns_objectives(trained):
    _, metrics = trained
    assert all(r > 0.5 for r in metrics["obj_r2"]), metrics["obj_r2"]


def test_predict_shapes(trained, synthetic):
    model, _ = trained
    x = synthetic[0]
    feas, obj = model.predict(x[:7])
    assert feas.shape == (7,)
    assert obj.shape == (7, 3)
    assert np.isfinite(obj).all()
