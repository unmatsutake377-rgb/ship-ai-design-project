"""모터 카탈로그 선택 (오너 제안 Q6, 백로그 #16).

실무 방식: 소요 추력 × 여유율 → 시판 모터 카탈로그에서 이산 선택.
- 차동 추력 USV 구성 = 모터 2발 (spec §2.4, 러더 없음)
- 여유율 2.0: 파랑·바람 중 조종 여유 + 열화 마진 (개략, 명명 상수)
- 적합 후보 중 최경량 선택 (소형정은 중량이 지배 제약)
- 카탈로그 최대로도 부족하면 명시적 거절 (조용한 부족 금지)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_CATALOG = Path(__file__).parent.parent.parent / "data" / "motor_catalog.csv"

THRUST_MARGIN = 2.0   # 총 장착 추력 / 소요 추력 최소비
MOTOR_COUNT = 2       # 차동 추력 구성


class NoSuitableMotorError(ValueError):
    """카탈로그의 어떤 모터로도 소요 추력을 못 채움."""


@dataclass(frozen=True)
class MotorSelection:
    motor: dict            # 카탈로그 행 (name, maker, thrust_max_n, ...)
    count: int
    total_thrust_n: float
    total_weight_kg: float
    total_power_w: float
    utilization: float     # 소요 추력 / 장착 총추력 (낮을수록 여유)


def load_catalog(path: str | Path = DEFAULT_CATALOG) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["name", "maker", "thrust_max_n", "weight_kg", "power_max_w",
                "source_url", "source_grade"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"모터 카탈로그 필수 컬럼 누락: {missing}")
    return df


def select_motors(required_thrust_n: float,
                  catalog_path: str | Path = DEFAULT_CATALOG,
                  margin: float = THRUST_MARGIN,
                  count: int = MOTOR_COUNT) -> MotorSelection:
    """소요 추력을 여유율 포함해 감당하는 최경량 모터 구성을 고른다."""
    df = load_catalog(catalog_path)
    needed_total = margin * required_thrust_n

    adequate = df[count * df["thrust_max_n"] >= needed_total]
    if adequate.empty:
        best = count * df["thrust_max_n"].max()
        raise NoSuitableMotorError(
            f"소요 추력 {required_thrust_n:.1f} N × 여유율 {margin} = "
            f"{needed_total:.1f} N을 채울 모터가 카탈로그에 없습니다 "
            f"(현재 {count}발 구성 최대 {best:.1f} N). "
            "카탈로그 확장 또는 저항 저감 필요."
        )

    pick = adequate.loc[adequate["weight_kg"].idxmin()]
    total_thrust = count * float(pick["thrust_max_n"])
    return MotorSelection(
        motor=pick.to_dict(),
        count=count,
        total_thrust_n=total_thrust,
        total_weight_kg=count * float(pick["weight_kg"]),
        total_power_w=count * float(pick["power_max_w"]),
        utilization=required_thrust_n / total_thrust,
    )
