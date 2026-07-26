import pandas as pd
import pytest

from src.physics.propulsion import (
    THRUST_MARGIN,
    MotorSelection,
    NoSuitableMotorError,
    load_catalog,
    select_motors,
)


def test_catalog_loads_and_valid():
    df = load_catalog()
    assert len(df) >= 3
    assert (df["thrust_max_n"] > 0).all()
    assert (df["weight_kg"] > 0).all()
    assert df["source_url"].str.startswith("http").all()
    assert set(df["source_grade"]) <= {"A", "B"}


def test_select_small_boat_picks_t200_pair():
    """소요 29 N (조사 USV 실측 케이스): T200 2발이면 여유율 포함 충분."""
    sel = select_motors(required_thrust_n=29.2)
    assert sel.count == 2                    # 차동 추력 구성 (spec §2.4)
    assert sel.motor["name"] == "T200"       # 충분한 것 중 최경량
    assert sel.total_thrust_n == pytest.approx(2 * 51.5, rel=1e-6)
    assert sel.total_thrust_n >= THRUST_MARGIN * 29.2


def test_select_bigger_boat_escalates():
    """소요 100 N: T200 쌍(103 N)은 여유율 2.0 미달 → 상위 모터로."""
    sel = select_motors(required_thrust_n=100.0)
    assert sel.motor["name"] != "T200"
    assert sel.total_thrust_n >= THRUST_MARGIN * 100.0


def test_no_suitable_motor_raises_with_max():
    """카탈로그 최대치로도 부족하면 명시적 거절 + 최대 가용치 안내."""
    with pytest.raises(NoSuitableMotorError, match="최대"):
        select_motors(required_thrust_n=5000.0)


def test_utilization_reported():
    sel = select_motors(required_thrust_n=29.2)
    assert 0 < sel.utilization < 1
    assert sel.utilization == pytest.approx(29.2 / sel.total_thrust_n, rel=1e-9)


def test_selection_is_lightest_adequate():
    """적합 후보 중 총중량 최소를 골라야 함."""
    df = load_catalog()
    sel = select_motors(required_thrust_n=29.2)
    adequate = df[2 * df["thrust_max_n"] >= THRUST_MARGIN * 29.2]
    assert sel.motor["weight_kg"] == adequate["weight_kg"].min()
