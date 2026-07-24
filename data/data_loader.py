"""Ship-D 스키마(45파라미터) 데이터 로더.

현재는 더미 데이터 생성 + 스키마 검증만 담당.
실제 Ship-D 합성 파라메트릭 데이터셋 연동은 M5a (spec §5).
Ship-D: Bagazinski & Ahmed (MIT, 2023) — 합성 파라메트릭 선형 ~30k척.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SHIPD_PARAM_COUNT = 45
PARAM_COLUMNS = [f"p{i:02d}" for i in range(1, SHIPD_PARAM_COUNT + 1)]
ALL_COLUMNS = ["hull_id"] + PARAM_COLUMNS


class SchemaError(ValueError):
    """Ship-D 스키마 위반."""


def generate_dummy_dataset(n_hulls: int, seed: int = 0) -> pd.DataFrame:
    """[0,1) 균등분포 더미. 파이프라인 배관 테스트 전용 — 물리적 의미 없음."""
    rng = np.random.default_rng(seed)
    data = {"hull_id": [f"dummy_{i:04d}" for i in range(n_hulls)]}
    for col in PARAM_COLUMNS:
        data[col] = rng.random(n_hulls)
    return pd.DataFrame(data)


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    _validate(df)
    df.to_csv(path, index=False)


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"필수 컬럼 누락: {missing}")
    extra = [c for c in df.columns if c not in ALL_COLUMNS]
    if extra:
        raise SchemaError(f"알 수 없는 컬럼: {extra}")
    params = df[PARAM_COLUMNS]
    if params.isna().any().any():
        bad = params.columns[params.isna().any()].tolist()
        raise SchemaError(f"NaN 포함 컬럼: {bad}")
    non_numeric = [c for c in PARAM_COLUMNS
                   if not pd.api.types.is_numeric_dtype(params[c])]
    if non_numeric:
        raise SchemaError(f"숫자가 아닌 컬럼: {non_numeric}")
