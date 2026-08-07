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
    assert pick.mcr_kw == 4640.0      # Wärtsilä 8L32 (실물 승급 08-07)
    assert pick.load_fraction == pytest.approx(3000 * 1.15 / 4640)
    assert pick.source_grade == "A"   # 제조사 프로덕트 가이드 계보


def test_select_engine_honest_refusal():
    with pytest.raises(NoSuitableEngineError, match="채울 엔진"):
        select_engine(40_000.0)


def test_design_propulsion_end_to_end():
    """통합: 100 m 화물선급 (R 200 kN·7 m/s·T 6 m) — 프로펠러 실효율
    ηD로 제동동력 → 엔진 선정, 캐비테이션 충족."""
    from src.physics.propulsion_large import design_propulsion

    prop, engine = design_propulsion(200_000.0, 7.0, draft=6.0)
    assert prop.cavitation_ok
    assert prop.diameter == pytest.approx(4.2)     # 0.70·T
    assert 0.55 < prop.eta_d < 0.85
    assert engine.mcr_kw >= prop.brake_power_kw * 1.15
