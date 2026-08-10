"""조종 프록시 — 정상 선회 닫힌 해법 앵커 (NSGA fast 게이트용).

원리: 전체 게이트와 **같은 MMG derivatives**를 (u̇,v̇,ṙ)=0
정상상태로 풀어 (시간 적분 생략) 정상 선회 반경 → 선회지름
프록시. 새 물리·새 계수 0 — 오차는 "과도기 대 정상상태" 하나뿐
이라 보정 계수 1개로 캘리브레이션.
"""
import math

import numpy as np
import pytest

from src.physics.maneuvering.kvlcc2 import KVLCC2_COEFFS, KVLCC2_L7

U0 = 1.179
SCALE = 45.7
RATE_DPS = 1.76 * math.sqrt(SCALE)


def _r0() -> float:
    s_wet = 27194.0 / SCALE ** 2
    re = U0 * 7.00 / 1.14e-6
    cf = 0.075 / (math.log10(re) - 2.0) ** 2
    return 0.5 * 1000.0 * s_wet * U0 ** 2 * cf * 1.25


def _kvlcc2_ship():
    from src.physics.maneuvering.mmg import MMGShip, solve_self_propulsion
    r0 = _r0()
    n_p = solve_self_propulsion(KVLCC2_L7, KVLCC2_COEFFS, U0, r0)
    return MMGShip(par=KVLCC2_L7, co=KVLCC2_COEFFS, r0_n=r0,
                   n_p=n_p, u0=U0)


def test_steady_state_matches_sim_tail():
    """정상해 = 시뮬 꼬리 — 같은 모델이니 r·u가 수 % 안에서 일치
    해야 함 (프록시의 존재 증명: 적분 없이 같은 답)."""
    from src.physics.maneuvering.proxy import steady_turning_state
    from src.physics.maneuvering.trials import simulate
    ship = _kvlcc2_ship()
    st = steady_turning_state(ship, U0, delta_deg=35.0)
    assert st["converged"]
    t_end = 60.0 * ship.par.lpp / U0
    res = simulate(ship, lambda t, s: math.radians(35.0), t_end, U0,
                   RATE_DPS, dt=ship.par.lpp / U0 / 120.0)
    tail = res["state"][int(0.9 * len(res["state"])):]
    r_sim = float(np.mean(tail[:, 2]))
    u_sim = float(np.mean(tail[:, 0]))
    assert st["r"] == pytest.approx(r_sim, rel=0.05)
    assert st["u"] == pytest.approx(u_sim, rel=0.05)


def test_proxy_tracks_sim_estimation_ships():
    """프록시 선회지름 ≈ 시뮬 선회지름 — 추정 사슬 선박군 (합성
    100 m, Cb 스윕) ±10%. 보정 1.15는 이 선박군 실측 (KVLCC2는
    박제 계수·초비대 Cb 0.81 — 비 1.42 별개 모집단이라 앵커 아님,
    정직 각주). 물리 방향: Cb↓(날씬) → 선회지름↑."""
    from src.physics.maneuvering.builder import mmg_ship_from_dims
    from src.physics.maneuvering.proxy import turning_proxy
    from src.physics.maneuvering.trials import turning_circle

    dts = []
    for cb in (0.47, 0.55, 0.60):
        ship, _ = mmg_ship_from_dims(
            loa=100.0, beam=16.2, draft=5.9, cb=cb,
            displacement_m3=cb * 100.0 * 16.2 * 5.9,
            d_prop=4.0, resistance_n=2.0e5, speed_ms=5.5,
            ear=0.45, z=4, pitch_ratio=0.8)
        p = turning_proxy(ship, 5.5)
        sim = turning_circle(ship, 5.5, delta_deg=35.0,
                             rudder_rate_dps=2.34,
                             dt=100.0 / 5.5 / 120.0)
        assert p["tactical_diameter_proxy_over_l"] == pytest.approx(
            sim["tactical_diameter_over_l"], rel=0.10)
        dts.append(p["tactical_diameter_proxy_over_l"])
    assert dts[0] > dts[1] > dts[2]      # 날씬할수록 크게 돎


@pytest.mark.skipif(
    not (__import__("pathlib").Path("outputs/pareto_large_v2.csv")
         .exists()
         and __import__("pathlib").Path(
             "outputs/pareto_large_v3_evo.csv").exists()),
    reason="전선 CSV 없음 (outputs 로컬 전용) — 정직 스킵")
def test_proxy_discriminates_real_fronts():
    """판별력 실전 — v2 합격선(시뮬 D_T ≤5)과 v3 불합격선(6.5)
    실벡터 각 3척을 한계 5.0으로 완전 분리 (전수 실측 2026-08-11:
    v2 97척 오탈락 0·v3 21척 오생존 0의 회귀 앵커)."""
    import json

    import pandas as pd

    from data import shipd_loader
    from src.ai.dimension_estimator import estimate_dimensions
    from src.core.types import GoalSpec
    from src.physics.holtrop import holtrop_input_from_mesh
    from src.physics.maneuvering.builder import mmg_ship_from_dims
    from src.physics.maneuvering.proxy import turning_proxy
    from src.pipeline_large import design_spiral_large, dims_from_shipd_mesh

    tl = estimate_dimensions(GoalSpec(
        target_speed_ms=6.0, payload_kg=8e6, purpose="cargo")).loa

    def dt_for(vector_json: str, speed: float) -> float:
        v45 = np.asarray(json.loads(vector_json), float)
        mesh = shipd_loader.scaled_mesh(v45, tl)
        h0 = holtrop_input_from_mesh(mesh, tl,
                                     float(mesh.bounds[1][2]) / 1.6)
        dims = dims_from_shipd_mesh(mesh, tl, h0.cb)
        large = design_spiral_large(mesh, dims, GoalSpec(
            target_speed_ms=speed, payload_kg=8e6, purpose="cargo"))
        ship, _ = mmg_ship_from_dims(
            dims.loa, dims.beam, large["draft"], dims.cb,
            large["total_t"] * 1000.0 / 1025.0,
            large["propeller"]["diameter"],
            large["resistance"]["total"], speed,
            large["propeller"]["ear"], large["propeller"]["z"],
            large["propeller"]["pitch_ratio"])
        return turning_proxy(ship, speed)[
            "tactical_diameter_proxy_over_l"]

    v2 = pd.read_csv("outputs/pareto_large_v2.csv")
    v3 = pd.read_csv("outputs/pareto_large_v3_evo.csv")
    for i in (0, len(v2) // 2, len(v2) - 1):
        r = v2.iloc[i]
        assert dt_for(r["vector_json"], r["speed_ms"]) <= 5.0
    for i in (0, len(v3) // 2, len(v3) - 1):
        r = v3.iloc[i]
        assert dt_for(r["vector_json"], r["speed_ms"]) > 5.0


def test_builder_returns_band_notes():
    """빌더 스모크 — MMGShip + 대역 노트 (기존 게이트와 같은 추정
    사슬 재사용 확인)."""
    from src.physics.maneuvering.builder import mmg_ship_from_dims
    ship, notes = mmg_ship_from_dims(
        loa=115.0, beam=18.8, draft=6.9, cb=0.62,
        displacement_m3=0.62 * 115.0 * 18.8 * 6.9,
        d_prop=4.2, resistance_n=2.5e5, speed_ms=6.0,
        ear=0.45, z=4, pitch_ratio=0.8)
    assert ship.par.lpp == pytest.approx(115.0)
    assert notes["band"] == "in"
    assert ship.n_p > 0


@pytest.mark.skipif(
    not __import__("pathlib").Path(
        "outputs/pareto_large_v3_evo.csv").exists(),
    reason="전선 CSV 없음 (outputs 로컬 전용) — 정직 스킵")
def test_fast_gate_kills_slender_turner():
    """배선 — evaluate_large_vector가 v3 유형(선회 초과)을 fast
    단계에서 사망 처리 (진화의 조종 사각지대 봉쇄)."""
    import json

    import pandas as pd

    from src.ai.dimension_estimator import estimate_dimensions
    from src.ai.optimize_large import evaluate_large_vector
    from src.core.types import GoalSpec

    tl = estimate_dimensions(GoalSpec(
        target_speed_ms=6.0, payload_kg=8e6, purpose="cargo")).loa
    r = pd.read_csv("outputs/pareto_large_v3_evo.csv").iloc[0]
    v45 = np.asarray(json.loads(r["vector_json"]), float)
    assert evaluate_large_vector(v45, float(r["speed_ms"]),
                                 8e6, tl) is None
