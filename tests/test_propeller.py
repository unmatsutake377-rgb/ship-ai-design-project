"""Wageningen B-시리즈 검증 (3단계 2차) — 물리 성질 + 문헌 대표점."""
import pytest

from src.physics.propeller import (
    PropellerDesignError,
    design_propeller,
    keller_min_ear,
    kt_kq,
)


def test_kt_kq_curve_matches_literature_b470():
    """문헌 삼각 검증: B4-70 P/D 1.0 개수 곡선 (공개 도표 계보) —
    ① 계류(J=0) Kt 0.45~0.50 ② 영추력점 J 1.00~1.10
    ③ 최대 효율 0.63~0.72 (J 0.8~0.95 부근)."""
    import math

    kt0, kq0 = kt_kq(0.0, 1.0, 0.70, 4)
    assert 0.45 < kt0 < 0.50
    assert 0.060 < kq0 < 0.072
    # 영추력점
    j = 1.0
    while kt_kq(j, 1.0, 0.70, 4)[0] > 0:
        j += 0.01
    assert 1.00 < j < 1.10
    # 최대 효율
    etas = []
    for ji in range(5, 106):
        jj = ji / 100
        kt, kq = kt_kq(jj, 1.0, 0.70, 4)
        if kt > 0 and kq > 0:
            etas.append((jj * kt / (2 * math.pi * kq), jj))
    eta_max, j_at = max(etas)
    assert 0.63 < eta_max < 0.72
    assert 0.80 < j_at < 0.95


def test_kt_monotonic_decreasing_in_j():
    """물리 성질: J 증가 → Kt 단조 감소, J=0에서 최대 (계류 추력)."""
    kts = [kt_kq(j / 10, 1.0, 0.55, 4)[0] for j in range(0, 11)]
    assert all(a > b for a, b in zip(kts, kts[1:]))
    assert kts[0] > 0.3


def test_kt_increases_with_pitch():
    """물리 성질: 같은 J에서 피치 크면 추력 큼."""
    assert kt_kq(0.5, 1.2, 0.55, 4)[0] > kt_kq(0.5, 0.8, 0.55, 4)[0]


def test_design_cargo_100m_sane():
    """100 m 화물선급: R 200 kN·7 m/s·D 4.5 m → 효율 0.5~0.75,
    rpm 수십~200, 캐비테이션 판정 산출."""
    d = design_propeller(200_000.0, 7.0, diameter_max=4.5)
    assert 0.45 < d.eta0 < 0.80
    assert 0.55 < d.eta_d < 0.85          # ηH=(1-0.2)/(1-0.3)=1.143 증폭
    assert 30 < d.rpm < 250
    from src.physics.propeller import thrust_deduction
    assert d.thrust_n >= 200_000.0 / (1 - thrust_deduction(0.70)) * 0.999
    assert d.brake_power_kw == pytest.approx(
        200_000 * 7.0 / d.eta_d / 1000, rel=1e-9)
    assert d.ear_min_keller > 0


def test_design_honest_refusal():
    """비물리 요구 (거대 추력·소직경) — 정직 거절."""
    with pytest.raises(PropellerDesignError, match="달성 불가"):
        design_propeller(5_000_000.0, 7.0, diameter_max=1.0)


def test_keller_ear_scales_with_thrust():
    assert keller_min_ear(2e6, 5.0, 4, 5.0) > keller_min_ear(5e5, 5.0, 4, 5.0)


def test_holtrop_wake_thrust_kcs_magnitudes():
    """Holtrop 정밀 w·t·ηR: KCS 실선 자릿수 (문헌 w 0.2~0.35,
    t 0.14~0.21, ηR 0.98~1.03) + 성질 t < w."""
    from src.physics.propeller import holtrop_wake_thrust

    w, t, er = holtrop_wake_thrust(lpp=230.0, beam=32.2, draft=10.8,
                                   d_prop=7.9, cp=0.66, lcb_pct=-1.48,
                                   wetted_surface=9530.0, k1=1.165,
                                   speed=12.35)
    assert 0.20 < w < 0.35
    assert 0.14 < t < 0.21
    assert 0.98 < er < 1.03
    assert t < w
