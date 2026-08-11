"""갤러리 전선 로더 (streamlit 비의존).

"가장 최신 완전 검증 전선" 규약 (스펙 §4·§7): 현재 v4 —
evo CSV(설계값) + full recheck CSV(판정) idx 병합, full_passed만.
자산 없으면 None (정직 안내 — outputs/는 로컬 전용이라 새 클론엔
없음). ⚠ vector_json은 화면 비노출 (Ship-D 라이선스).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EVO_CSV = "pareto_large_v4_evo.csv"
RECHECK_CSV = "pareto_v4_full_recheck.csv"


def load_verified_front(outputs_dir: str | Path) -> pd.DataFrame | None:
    """완전 검증 전선 DataFrame — 없으면 None."""
    d = Path(outputs_dir)
    evo_p, rc_p = d / EVO_CSV, d / RECHECK_CSV
    if not (evo_p.exists() and rc_p.exists()):
        return None
    evo = pd.read_csv(evo_p).reset_index().rename(
        columns={"index": "idx"})
    rc = pd.read_csv(rc_p)
    keep = [c for c in rc.columns
            if c == "idx" or c not in evo.columns]
    df = evo.merge(rc[keep], on="idx")
    df = df[df["full_passed"]].reset_index(drop=True)
    return df.drop(columns=["vector_json"], errors="ignore")
