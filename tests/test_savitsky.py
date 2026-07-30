"""Savitsky 활주 모듈 테스트 (Phase C-2 Task 1) — 검증 3층."""
import math

import pytest

from src.physics.savitsky import (
    PlaningEquilibriumError,
    center_of_pressure_ratio,
    cl_beta,
    cl_zero,
    solve_cl0,
    solve_equilibrium,
)


# ① 경험식 전사 — 손계산 고정값
def test_cl_zero_pinned():
    """τ=4°, λ=2, Cv=3: C_L0 = 4^1.1·(0.0120·√2 + 0.0055·2^2.5/9)."""
    expected = 4 ** 1.1 * (0.0120 * 2 ** 0.5 + 0.0055 * 2 ** 2.5 / 9.0)
    assert cl_zero(4.0, 2.0, 3.0) == pytest.approx(expected, rel=1e-12)


def test_cl_beta_pinned():
    """C_L0=0.1, β=15°: C_Lβ = 0.1 − 0.0065·15·0.1^0.6."""
    expected = 0.1 - 0.0065 * 15 * 0.1 ** 0.6
    assert cl_beta(0.1, 15.0) == pytest.approx(expected, rel=1e-12)


def test_cp_pinned():
    """Cv=3, λ=2: Cp = 0.75 − 1/(5.21·9/4 + 2.39)."""
    expected = 0.75 - 1.0 / (5.21 * 9 / 4 + 2.39)
    assert center_of_pressure_ratio(3.0, 2.0) == pytest.approx(
        expected, rel=1e-12)


def test_solve_cl0_roundtrip():
    cl0 = 0.08
    clb = cl_beta(cl0, 15.0)
    assert solve_cl0(clb, 15.0) == pytest.approx(cl0, rel=1e-6)


# ② 평형 자기일관
CASE = dict(weight_n=1200.0 * 9.81, speed=12.0, beam=2.0,
            deadrise_deg=15.0, lcg_from_transom=2.4)


def test_equilibrium_self_consistent():
    st = solve_equilibrium(**CASE)
    # 양력 일치: 풀린 (τ,λ)의 C_Lβ가 요구값 재현
    cv = CASE["speed"] / math.sqrt(9.81 * CASE["beam"])
    clb_req = CASE["weight_n"] / (0.5 * 1025 * CASE["speed"] ** 2
                                  * CASE["beam"] ** 2)
    cl0 = cl_zero(st.trim_deg, st.lam, cv)
    assert cl_beta(cl0, 15.0) == pytest.approx(clb_req, rel=1e-3)
    # 모멘트 일치: lcp = LCG
    lcp = center_of_pressure_ratio(cv, st.lam) * st.lam * CASE["beam"]
    assert lcp == pytest.approx(CASE["lcg_from_transom"], rel=1e-3)
    assert st.resistance_n > 0
    assert st.friction_n > 0 and st.induced_n > 0


# ③ 물리 경향
def test_heavier_more_trim():
    light = solve_equilibrium(**{**CASE, "weight_n": CASE["weight_n"]})
    heavy = solve_equilibrium(**{**CASE, "weight_n": CASE["weight_n"] * 1.5})
    assert heavy.trim_deg > light.trim_deg


def test_faster_less_wetted():
    slow = solve_equilibrium(**CASE)
    fast = solve_equilibrium(**{**CASE, "speed": CASE["speed"] * 1.4})
    assert fast.lam < slow.lam


def test_more_deadrise_more_trim():
    flat = solve_equilibrium(**{**CASE, "deadrise_deg": 10.0})
    vee = solve_equilibrium(**{**CASE, "deadrise_deg": 20.0})
    assert vee.trim_deg > flat.trim_deg


def test_impossible_lcg_rejected():
    """LCG가 압력중심 도달범위 밖 → 명시적 실패."""
    with pytest.raises(PlaningEquilibriumError):
        solve_equilibrium(**{**CASE, "lcg_from_transom": 20.0})
