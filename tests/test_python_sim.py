import json

import numpy as np
import pytest

from src.core.types import GoalSpec
from src.pipeline import run_pipeline
from src.sim_adapters.python_sim import (
    SimResult,
    default_square_course,
    simulate_waypoints,
    step,
    vessel_from_report,
)


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    out = tmp_path_factory.mktemp("design")
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=100.0, purpose="survey")
    return run_pipeline(goal, out)


@pytest.fixture(scope="module")
def vessel(report):
    return vessel_from_report(report)


def test_zero_thrust_stays_at_rest(vessel):
    """물리 불변량: 추력 0이면 정지 유지 (spec §4)."""
    state = np.zeros(6)
    for _ in range(200):
        state = step(vessel, state, 0.0, 0.0, 0.05)
    assert np.allclose(state, 0.0, atol=1e-9)


def test_constant_thrust_terminal_speed(vessel):
    """등추력 → 이론 종속도(R(u)=2T의 해) 수렴 (spec §4)."""
    t_each = 10.0
    state = np.zeros(6)
    for _ in range(12000):  # 600 s
        state = step(vessel, state, t_each, t_each, 0.05)
    u_terminal = state[3]
    expected = float(np.interp(2 * t_each, vessel.resistances, vessel.speeds))
    assert u_terminal == pytest.approx(expected, rel=0.05)


def test_straight_running_no_drift(vessel):
    """좌우 등추력 직진: 횡 이탈이 미소해야 함."""
    state = np.zeros(6)
    for _ in range(4000):  # 200 s
        state = step(vessel, state, 10.0, 10.0, 0.05)
    assert abs(state[1]) < 0.01 * abs(state[0])  # |y| < 1%·x


def test_differential_thrust_turns_ccw(vessel):
    """우현 추력 우세 → 반시계 선회 (r > 0, ψ 증가)."""
    state = np.zeros(6)
    for _ in range(2000):
        state = step(vessel, state, 5.0, 12.0, 0.05)
    assert state[5] > 0  # r
    assert state[2] > 0  # psi


def test_demo_vessel_completes_square_course(vessel, report):
    """데모 설계가 사각 코스(변 10L) 4개 웨이포인트 완주."""
    waypoints = default_square_course(vessel.loa)
    result = simulate_waypoints(
        vessel, waypoints, u_desired=report["goal"]["target_speed_ms"]
    )
    assert result.success, (
        f"도달 {result.waypoints_reached}/{len(waypoints)}"
    )
    assert result.duration_s < 600.0


def test_result_arrays_consistent(vessel, report):
    waypoints = default_square_course(vessel.loa)
    result = simulate_waypoints(vessel, waypoints, u_desired=1.5)
    n = len(result.time)
    assert n == len(result.x) == len(result.y) == len(result.u)
    assert isinstance(result, SimResult)


def test_path_quality_no_weaving(vessel, report):
    """경로 품질: 실제 경로 길이 / 이상 경로(둘레) < 1.35.

    회귀 방지: 초기 게인(임의 상수)에서 S자 요동으로 비율 ~2가 나왔음 —
    '도달'만 검사하면 뱀 궤적도 통과한다. 극배치 게인 도입 근거.
    """
    waypoints = default_square_course(vessel.loa)
    result = simulate_waypoints(
        vessel, waypoints, u_desired=report["goal"]["target_speed_ms"]
    )
    assert result.success
    xs, ys = np.array(result.x), np.array(result.y)
    path_length = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    ideal = 4 * 10.0 * vessel.loa  # 사각 둘레
    # 고유값 설계 게인(#21) 후 실측 0.96 (코너 안쪽 절단로 1.0 미만 가능).
    # 1.10 = 실측 + 여유. 이력: 임의 게인 ~2 → 1.35 → 1.20 → 1.10.
    assert path_length / ideal < 1.10, f"ratio={path_length / ideal:.2f}"
    # 폐루프 선형 안정 확인 (한계 사이클 회귀 방지)
    assert result.control_design["stable"] is True


def test_straight_line_weave_killed(vessel):
    """직선 추종 잔물결 회귀 방지: 2 m 이탈에서 시작해도 말단에서 침묵.

    이력 (07-27): 헤딩 P만으로는 불안정 선체가 1.29 m 한계 사이클에
    고착했음 (dt 무관 — 실제 동역학). 고유값 설계 게인이 해법.
    """
    from src.sim_adapters.python_sim import design_gains, ssa
    import math

    kp, kd, lookahead, info = design_gains(vessel, 1.5)
    assert info["stable"]
    state = np.array([0.0, 2.0, 0.0, 1.2, 0.0, 0.0])
    kp_u = 8.0 * vessel.m_x / 10.0
    tail_e = []
    for _ in range(8000):
        x, y, psi, u, v_, r = state
        if x > 95:
            break
        psi_d = math.atan2(-y, lookahead)
        m = kp * ssa(psi_d - psi) - kd * r
        d = float(np.clip(m / vessel.thruster_sep,
                          -vessel.thrust_max, vessel.thrust_max))
        head = vessel.thrust_max - abs(d)
        c = float(np.clip(kp_u * (1.5 - u), -head, head))
        from src.sim_adapters.python_sim import step
        state = step(vessel, state, c - d, c + d, 0.05)
        if state[0] > 70:
            tail_e.append(abs(state[1]))
    assert max(tail_e) < 0.3, f"말단 잔물결 {max(tail_e):.2f} m"
