"""인간 선호도 점수(1~5) 로깅 (spec §2.5).

점수는 보조 fine-tune/재순위 신호 — 주 학습 신호는 물리 라벨(M5b).
이력 보존을 위해 append-only. 같은 선박 재평가도 새 행으로 기록.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

COLUMNS = ["hull_id", "score", "timestamp"]


def record_score(hull_id: str, score: int, csv_path: str | Path) -> None:
    if not isinstance(score, int) or isinstance(score, bool):
        raise ValueError(f"점수는 정수여야 합니다: {score!r}")
    if not 1 <= score <= 5:
        raise ValueError(f"점수는 1~5 범위여야 합니다: {score}")
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(COLUMNS)
        writer.writerow(
            [hull_id, score, datetime.now(timezone.utc).isoformat()]
        )


def load_scores(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)
