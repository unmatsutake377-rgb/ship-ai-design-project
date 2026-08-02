"""러더 물리 모형 — 손계산 정답지 + 가설 검증 (자기지속 루프 해소)."""
import numpy as np
import pytest

from src.sim_adapters.rudder import (
    RUDDER_STALL_RAD,
    RudderModel,
    rudder_moment,
)


def test_rudder_force_hand_calc():
    """δ=10°, u=1 m/s, A=0.05 m², x_r=−1.4: 손계산.

    L = ½·1025·1²·0.05·(2π·0.9·0.1745) = 25.28 N
    N = −L·x_r = +25.28·1.4 = 35.4 N·m (양의 δ → 양의 모멘트)"""
    r = RudderModel(area=0.05, x_pos=-1.4, k=0.9)
    n = rudder_moment(r, u=1.0, delta=np.radians(10.0))
    lift = 0.5 * 1025.0 * 0.05 * 2 * np.pi * 0.9 * np.radians(10.0)
    assert n == pytest.approx(lift * 1.4, rel=1e-6)


def test_rudder_powerless_without_flow():
    """물살 없으면 무력 — V² 항의 핵심 특성."""
    r = RudderModel(area=0.05, x_pos=-1.4)
    assert rudder_moment(r, u=0.0, delta=np.radians(30.0)) == 0.0


def test_rudder_stall_saturates():
    """실속 경계 밖에선 모멘트가 더 안 늘어남 (포화)."""
    r = RudderModel(area=0.05, x_pos=-1.4)
    n_stall = rudder_moment(r, u=1.0, delta=RUDDER_STALL_RAD)
    n_beyond = rudder_moment(r, u=1.0, delta=RUDDER_STALL_RAD * 1.4)
    assert n_beyond == pytest.approx(n_stall)


def test_low_speed_command_followed_with_rudder():
    """가설 판정관 (대조 실험): 전진 전용 세계에서 저속 명령 0.77.

    발견 (2026-08-03): 자기지속 루프는 '추력기 후진 금지'가 전제 —
    파이썬 시뮬 기본(후진 대칭 허용)에선 차동 합력이 0이라 재현 안 됨.
    forward_only=True(실물·gz 프로펠러 근사)를 걸면 활주정 실측:
      차동:  u평균 1.73 (명령의 2.2배 폭주, 경로비 1.74) ← 루프 재현
      러더:  u평균 0.73 (추종, 경로비 1.00)              ← 루프 해소
    러더 배분은 구조상 이미 전진 전용 (f∈[0,2T], |d|≤min(f,2T−f))."""
    import json
    from pathlib import Path

    from src.sim_adapters.python_sim import (
        simulate_waypoints,
        simulate_waypoints_rudder,
        vessel_from_report,
    )

    report = json.loads(Path("outputs/planing_demo/report.json").read_text())
    v = vessel_from_report(report)
    L = report["dimensions"]["loa"]
    wps = [(10 * L, 0.0), (10 * L, 10 * L), (0.0, 10 * L), (0.0, 0.0)]

    diff = simulate_waypoints(v, wps, u_desired=0.77, t_max=1500.0,
                              forward_only=True)
    rud = simulate_waypoints_rudder(v, wps, u_desired=0.77, t_max=1500.0)
    assert diff.success and rud.success

    u_diff = float(np.mean(diff.u[len(diff.u) // 4:]))
    u_rud = float(np.mean(rud.u[len(rud.u) // 4:]))
    assert u_diff > 0.77 * 1.5           # 대조군: 루프 재현 (과속 고착)
    assert u_rud == pytest.approx(0.77, rel=0.20)   # 러더: 명령 추종


def test_rudder_course_quality():
    """러더 코스 품질: 경로비 < 1.15 (차동 기준 1.10과 동급 요구)."""
    import json
    from pathlib import Path

    from src.sim_adapters.python_sim import (
        simulate_waypoints_rudder,
        vessel_from_report,
    )

    report = json.loads(Path("outputs/demo_cfd/report.json").read_text())
    v = vessel_from_report(report)
    L = report["dimensions"]["loa"]
    wps = [(10 * L, 0.0), (10 * L, 10 * L), (0.0, 10 * L), (0.0, 0.0)]
    res = simulate_waypoints_rudder(
        v, wps, u_desired=report["goal"]["target_speed_ms"])
    assert res.success
    xs, ys = np.array(res.x), np.array(res.y)
    path = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    assert path / (4 * 10 * L) < 1.15
