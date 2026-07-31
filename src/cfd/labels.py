"""CFD 라벨 축적 (스펙 §2 labels) — 대리모델 재학습용 데이터 그릇.

한 행 = 케이스 1회 실행. 경험식 라벨(emp_*)을 나란히 저장해
"CFD vs 경험식" 오차를 즉시 볼 수 있게 한다. 같은 case_name 재실행은
갱신 (중복 행이 생기면 재학습 때 그 케이스가 이중 가중되므로 금지).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cfd.result_parser import CfdResult


def append_label(csv_path: Path, case_name: str, speed: float, draft: float,
                 result: CfdResult, empirical: dict,
                 extra: dict | None = None) -> pd.DataFrame:
    """extra: 추가 열 (예: loa_m/beam_m — 보정 계수의 설명변수, 능동학습)."""
    row = {
        "case_name": case_name, "speed_ms": speed, "draft_m": draft,
        "cfd_total_n": result.drag_total_n,
        "cfd_pressure_n": result.drag_pressure_n,
        "cfd_viscous_n": result.drag_viscous_n,
        "converged": result.converged, "n_samples": result.n_samples,
        "emp_rf_n": empirical["rf"], "emp_rw_n": empirical["rw"],
        "emp_total_n": empirical["total"],
    }
    if extra:
        row.update(extra)
    csv_path = Path(csv_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["case_name"] != case_name]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
