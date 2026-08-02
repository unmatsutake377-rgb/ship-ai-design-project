"""러더 물리 모형 — 손계산 정답지 + 가설 검증 (자기지속 루프 해소)."""
import numpy as np
import pytest

from src.sim_adapters.rudder import (
    RUDDER_STALL_RAD,
    RudderModel,
    lift_slope_mandel,
    rudder_area_dnv,
    rudder_moment,
)


def test_lift_slope_mandel_hand_calc():
    """Mandel(1967) 손계산: ΛE=3 → 1.8π·3/(√13+1.8) = 3.1375 /rad.

    출처: Liu & Hekkenberg 2017 eq.(7) (data/rudder_servo_specs.csv)."""
    expected = 1.8 * np.pi * 3.0 / (np.sqrt(13.0) + 1.8)
    assert lift_slope_mandel(3.0) == pytest.approx(expected, rel=1e-9)
    assert lift_slope_mandel(3.0) == pytest.approx(3.1375, rel=1e-3)
    # 대안식(Söding 1982)과 자릿수 교차 검증: 2π·3·4/25 = 3.016
    assert lift_slope_mandel(3.0) == pytest.approx(3.016, rel=0.05)


def test_rudder_area_dnv_hand_calc():
    """DNV(1975): L=2, B=0.8, T=0.15 → (0.3/100)·(1+25·0.16)=0.015 m²."""
    assert rudder_area_dnv(2.0, 0.8, 0.15) == pytest.approx(0.015, rel=1e-9)


def test_rudder_force_hand_calc():
    """δ=10°, u=1 m/s, A=0.05 m², x_r=−1.4, ΛG=1.5 (ΛE=3): 손계산.

    L = ½·1025·1²·0.05·(3.1375·0.1745) = 14.03 N
    N = −L·x_r = +14.03·1.4 (양의 δ → 양의 모멘트)"""
    r = RudderModel(area=0.05, x_pos=-1.4, ar_geometric=1.5)
    n = rudder_moment(r, u=1.0, delta=np.radians(10.0))
    lift = (0.5 * 1025.0 * 0.05
            * lift_slope_mandel(3.0) * np.radians(10.0))
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


def test_single_thruster_rudder_completes_course():
    """구성 B (단일+러더): 목표 속도 코스 완주 + 품질."""
    import json
    from pathlib import Path

    from src.sim_adapters.python_sim import (
        default_square_course,
        simulate_course,
        vessel_from_report,
    )

    report = json.loads(Path("outputs/planing_demo/report.json").read_text())
    v = vessel_from_report(report)
    wps = default_square_course(v.loa)
    res = simulate_course(v, wps, report["goal"]["target_speed_ms"],
                          steering="rudder1", t_max=1500.0)
    assert res.success
    assert res.control_design["steering"] == "single+rudder"
    xs, ys = np.array(res.x), np.array(res.y)
    path = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    assert path / (4 * 10 * v.loa) < 1.4   # 저속 코너 쇠약 감안 완화 기준


def test_simulate_course_rejects_unknown_steering():
    from src.sim_adapters.python_sim import VesselModel, simulate_course

    v = VesselModel(loa=2.0, m_x=100, m_y=150, i_z=50, yv=40, nv=10, nr=30,
                    thrust_max=20, thruster_sep=0.8,
                    speeds=(0.0, 1.0), resistances=(0.0, 10.0))
    with pytest.raises(ValueError):
        simulate_course(v, [(1.0, 0.0)], 1.0, steering="warp_drive")


def test_steering_report_smoke(tmp_path):
    """비교 리포트: JSON 스키마 + 3구성 전부 채점."""
    import json
    from pathlib import Path

    from src.sim_adapters.steering_report import compare_steering

    report = json.loads(Path("outputs/demo_cfd/report.json").read_text())
    table = compare_steering(report)
    for mode in ("diff", "rudder2", "rudder1"):
        for phase in ("cruise", "low_speed"):
            s = table["modes"][mode][phase]
            assert set(s) == {"success", "duration_s", "path_ratio",
                              "cross_track_sigma_m", "u_mean_ms"}


def test_rudder_sizing_takes_max_of_dnv_and_required():
    """사이징: DNV 최소와 모멘트 역산 중 큰 쪽.

    조사선(느림·불안정)은 역산이 이김, 활주정(빠름)은 DNV가 이김 —
    V² 항 때문 (data/rudder_servo_specs.csv 실측 2026-08-03)."""
    from src.sim_adapters.rudder import RudderModel, rudder_area_dnv

    dnv = rudder_area_dnv(2.0, 1.0, 0.3)
    # 저속 설계점 + 큰 요구 모멘트 → 역산 면적이 DNV를 초과
    big = RudderModel.for_vessel(2.0, 0.3, beam=1.0,
                                 required_moment=40.0, u_design=0.9)
    assert big.area > dnv
    # 고속 설계점 → DNV 바닥 유지
    small = RudderModel.for_vessel(2.0, 0.3, beam=1.0,
                                   required_moment=40.0, u_design=10.0)
    assert small.area == pytest.approx(dnv)
