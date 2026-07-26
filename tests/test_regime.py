import pytest

from src.core.regime import (
    Regime,
    UnsupportedRegimeError,
    classify,
    froude_length,
    froude_volumetric,
    max_displacement_speed,
    min_loa_for_speed,
    require_supported,
)


def test_froude_length_known_value():
    # v=1.5 m/s, L=4 m → Fn = 1.5 / sqrt(9.81*4) = 0.2394...
    assert froude_length(1.5, 4.0) == pytest.approx(0.2394, abs=1e-3)


def test_displacement_regime():
    # 저속 조사 USV: Fn < 0.4
    assert classify(1.5, 4.0, 0.3) is Regime.DISPLACEMENT


def test_semi_displacement_regime():
    # 4 m 선체 5 m/s → Fn ≈ 0.80 (검토 지적 #1의 케이스)
    assert classify(5.0, 4.0, 0.3) is Regime.SEMI_DISPLACEMENT


def test_planing_regime():
    # 작은 배수량 + 고속 → Fn∇ ≥ 3
    vol = 0.2
    speed = 3.1 * (9.81 * vol ** (1 / 3)) ** 0.5
    assert froude_volumetric(speed, vol) > 3.0
    assert classify(speed, 4.0, vol) is Regime.PLANING


def test_require_supported_raises_with_message():
    with pytest.raises(UnsupportedRegimeError) as exc:
        require_supported(Regime.SEMI_DISPLACEMENT)
    assert "반배수량" in str(exc.value)


def test_max_speed_known_value():
    # L=4 m: v_max = 0.4·√(9.81·4) = 2.505... m/s
    assert max_displacement_speed(4.0) == pytest.approx(2.5054, abs=1e-3)


def test_max_speed_grows_with_length():
    assert max_displacement_speed(9.0) > max_displacement_speed(4.0)


def test_roundtrip_inverse():
    """역함수 관계: min_loa(max_speed(L)) == L."""
    loa = 5.3
    assert min_loa_for_speed(max_displacement_speed(loa)) == pytest.approx(
        loa, rel=1e-9
    )


def test_boundary_consistency_with_classify():
    """v_max 바로 아래는 배수량형, 바로 위는 아님 — 경계 상수 일관성."""
    loa, vol = 4.0, 0.3
    v = max_displacement_speed(loa)
    assert classify(v * 0.999, loa, vol) is Regime.DISPLACEMENT
    assert classify(v * 1.001, loa, vol) is not Regime.DISPLACEMENT
