"""실선 데이터 → 카테고리별 치수 통계 밴드 (오너 설계: 타입별 분리, 2026-07-26).

원칙:
- 통계는 hull_type별로 분리 — 쌍동선 비율을 단동선 공식에 넣지 않는다.
- 등급 가중: A=2표, B=1표 (가중 중앙값). C·격리는 validate_dataset이 이미 배제.
- 표본이 min_samples 미만인 카테고리는 밴드를 만들지 않는다 —
  호출 측이 개략값으로 fallback하되 반드시 "데이터 부족"을 표시할 것.
- Cb는 제조사 비공개 항목이라 데이터에서 못 뽑음 — 카테고리 상수 유지 (명시적 가정).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.quality import validate_dataset

DEFAULT_CSV = Path(__file__).parent / "small_craft_particulars.csv"

GRADE_VOTES = {"A": 2, "B": 1}  # 가중 중앙값용 반복 횟수


@dataclass(frozen=True)
class DataBand:
    """카테고리 하나의 치수 통계. source가 'data'면 실선 근거."""

    lb: float                     # L/B 가중 중앙값
    bt: float                     # B/T 가중 중앙값
    cb: float                     # 방형계수 (상수 가정 — 데이터 아님)
    payload_fraction: float       # 적재량/만재중량 가중 중앙값
    loa_range: tuple[float, float]  # 실선 길이 범위 [m]
    n_samples: int
    source: str                   # "data" | "fallback"


def _weighted_median(values: pd.Series, votes: pd.Series) -> float:
    repeated = np.repeat(values.to_numpy(dtype=float),
                         votes.to_numpy(dtype=int))
    return float(np.median(repeated))


def category_bands(csv_path: str | Path = DEFAULT_CSV,
                   hull_type: str = "monohull",
                   min_samples: int = 3,
                   cb_by_category: dict[str, float] | None = None,
                   ) -> dict[str, DataBand]:
    """CSV → 검증 → hull_type 필터 → 카테고리별 DataBand.

    표본 부족 카테고리는 결과에 없음 (호출 측 fallback 책임).
    """
    cb_map = cb_by_category or {}
    df = validate_dataset(pd.read_csv(csv_path))
    df = df[df["hull_type"] == hull_type]

    bands: dict[str, DataBand] = {}
    for category, group in df.groupby("category"):
        votes = group["grade"].map(GRADE_VOTES)

        lb = _weighted_median(group["loa_m"] / group["beam_m"], votes)

        with_draft = group.dropna(subset=["draft_m"])
        with_payload = group.dropna(subset=["payload_kg"])
        if len(group) < min_samples or with_draft.empty or with_payload.empty:
            continue  # 비율 하나라도 근거 없으면 밴드 안 만듦

        bt = _weighted_median(
            with_draft["beam_m"] / with_draft["draft_m"],
            votes.loc[with_draft.index],
        )
        payload_fraction = _weighted_median(
            with_payload["payload_kg"] / with_payload["weight_full_kg"],
            votes.loc[with_payload.index],
        )
        bands[str(category)] = DataBand(
            lb=lb, bt=bt,
            cb=cb_map.get(str(category), 0.50),
            payload_fraction=payload_fraction,
            loa_range=(float(group["loa_m"].min()),
                       float(group["loa_m"].max())),
            n_samples=len(group),
            source="data",
        )
    return bands
