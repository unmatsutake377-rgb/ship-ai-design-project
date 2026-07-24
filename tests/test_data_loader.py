import numpy as np
import pandas as pd
import pytest

from data.data_loader import (
    PARAM_COLUMNS,
    SHIPD_PARAM_COUNT,
    SchemaError,
    generate_dummy_dataset,
    load_dataset,
    save_dataset,
)


def test_dummy_dataset_shape():
    df = generate_dummy_dataset(10, seed=42)
    assert len(df) == 10
    assert list(df.columns) == ["hull_id"] + PARAM_COLUMNS
    assert len(PARAM_COLUMNS) == SHIPD_PARAM_COUNT == 45


def test_dummy_dataset_reproducible():
    a = generate_dummy_dataset(5, seed=1)
    b = generate_dummy_dataset(5, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_roundtrip(tmp_path):
    df = generate_dummy_dataset(5, seed=0)
    path = tmp_path / "hulls.csv"
    save_dataset(df, path)
    loaded = load_dataset(path)
    pd.testing.assert_frame_equal(loaded, df)


def test_reject_missing_column(tmp_path):
    df = generate_dummy_dataset(5, seed=0).drop(columns=["p45"])
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="p45"):
        load_dataset(path)


def test_reject_nan(tmp_path):
    df = generate_dummy_dataset(5, seed=0)
    df.loc[2, "p10"] = np.nan
    path = tmp_path / "nan.csv"
    df.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="NaN"):
        load_dataset(path)
