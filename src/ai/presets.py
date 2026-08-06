"""용도 프리셋 (#25 오너 제안 — 3입력 UX의 1단계, 2026-07-28).

목표: 사용자는 ①용도 ②길이 ③탑재하중만 입력 — 속도는 용도가 결정.
근거: 입문 사용자는 "몇 m/s"의 감각이 없음. 용도별 실선 순항속도
중앙값(실측 CSV)이 기본값, 데이터 없는 용도는 개략값 + 출처 표시.
--speed를 직접 주면 항상 우선 (고급 사용자 경로).

2단계(추후): 용도별 파레토 목적 가중 프리셋 (작업선=적재 중시 등)
— NSGA-II 목적 구조와 연동 (spec §7).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PARTICULARS_CSV = Path(__file__).parent.parent.parent / "data" \
    / "small_craft_particulars.csv"

# 데이터 없는 용도의 개략 기본 순항속도 [m/s]
_FALLBACK_SPEED = {"survey": 1.5, "patrol": 2.5, "workboat": 1.5,
                   "cargo": 7.0}   # ~13.6 kn 화물선 순항 통상 (B급)


@dataclass(frozen=True)
class PurposePreset:
    default_speed_ms: float
    speed_source: str   # "data" (실선 중앙값) | "fallback" (개략값)
    n_samples: int


def purpose_presets(csv_path: str | Path = PARTICULARS_CSV
                    ) -> dict[str, PurposePreset]:
    """용도별 기본 속도 프리셋 — 실선 순항속도 중앙값 우선."""
    presets: dict[str, PurposePreset] = {
        k: PurposePreset(v, "fallback", 0) for k, v in _FALLBACK_SPEED.items()
    }
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return presets
    for category, group in df.groupby("category"):
        cruise = group["speed_cruise_ms"].dropna()
        if len(cruise) >= 2:  # 최소 2척 근거 요구
            presets[str(category)] = PurposePreset(
                default_speed_ms=float(cruise.median()),
                speed_source="data",
                n_samples=int(len(cruise)),
            )
    return presets
