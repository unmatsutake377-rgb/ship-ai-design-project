"""대형 추진 1차 검증 (3단계 골격 — B-시리즈 정밀은 후속)."""
import pytest

from src.physics.propulsion_large import (
    NoSuitableEngineError,
    brake_power_kw,
    select_engine,
)


def test_brake_power_hand_calc():
    """PB 손계산: R 1,014 kN·7.6 m/s·ηD 0.60 → 12,844 kW."""
    pb = brake_power_kw(1_014_000.0, 7.6)
    assert pb == pytest.approx(1_014_000 * 7.6 / 0.6 / 1000, rel=1e-9)
    assert pb == pytest.approx(12_844, rel=0.001)


def test_select_engine_smallest_sufficient():
    """여유 15% 포함 최소 MCR 선택 + 부하율 보고."""
    pick = select_engine(3000.0)
    assert pick.mcr_kw == 4000.0
    assert pick.load_fraction == pytest.approx(3000 * 1.15 / 4000)
    assert pick.source_grade == "C"   # 승급 전 정직 표기


def test_select_engine_honest_refusal():
    with pytest.raises(NoSuitableEngineError, match="채울 엔진"):
        select_engine(40_000.0)
