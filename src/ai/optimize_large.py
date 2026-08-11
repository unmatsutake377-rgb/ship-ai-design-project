"""NSGA × 8중 게이트 — 대형 다목적 최적화 (스펙 2026-08-09).

유전자 46 = Ship-D 45 (dataset_bounds·constraints_ok 재사용) +
속도 [5.0, 7.5] m/s. 평가는 전부 기존 사슬 재사용 (새 물리 없음):
design_spiral_large → _structure_gate → _economics_gate_large.
내항·조종은 생략 (fast) — 대표 설계만 전체 8중 재검 (지도 관례).

목적 2: 수송단가 min · **속도 max** (수송 시간 가치 — 소예산
실측에서 단가·EEDI가 연료 비례로 정렬돼 전선이 점으로 붕괴,
"친환경=저비용" 정직 발견 후 목적 재구성). EEDI는 제약으로 유지.
제약: 나선(GM·건현) ∧ 구조 ∧ EEDI 합격 — 사망 페널티 + 사유 집계.

⚠ Ship-D 라이선스: 전선 CSV(vector_json 포함)는 outputs/ 로컬만
(커밋·공개 금지).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.ai.shipgen_optimize import constraints_ok, dataset_bounds
from src.core.types import GoalSpec

SPEED_LO, SPEED_HI = 5.0, 7.5
DEATH = (1e9, 1e9)


def evaluate_large_vector(vec45: np.ndarray, speed_ms: float,
                          goal_payload_kg: float,
                          target_loa: float) -> dict | None:
    """한 후보 평가 — 실패(게이트 사망)는 None + 호출측 집계."""
    from data import shipd_loader
    from src.pipeline import _economics_gate_large, _structure_gate
    from src.pipeline_large import design_spiral_large, dims_from_shipd_mesh
    from src.physics.holtrop import holtrop_input_from_mesh

    goal = GoalSpec(target_speed_ms=speed_ms,
                    payload_kg=goal_payload_kg, purpose="cargo")
    mesh = shipd_loader.scaled_mesh(vec45, target_loa)
    h0 = holtrop_input_from_mesh(mesh, target_loa,
                                 float(mesh.bounds[1][2]) / 1.6)
    dims = dims_from_shipd_mesh(mesh, target_loa, h0.cb)
    large = design_spiral_large(mesh, dims, goal)
    if not large["passed"]:
        return None

    from src.physics.cargo_capacity import space_gate_large
    sp = space_gate_large(mesh, dims.depth, dims.loa,
                          fuel_t=large["fuel_t"],
                          payload_t=large["payload_t"])
    if not sp["passed"]:
        return None

    # roll 프록시 (2026-08-10): 전 회차 전체 재검에서 내항이 47/50
    # 학살 — 주범은 전 케이스 roll (수식 2개짜리 초저렴 항목).
    # fast에 삽입해 fast·full 정합 개선. pitch·heave는 스트립 필요
    # — full 재검 몫 유지 (정직).
    from src.physics.seakeeping.criteria import (
        DESIGN_SEA_STATE,
        SEAKEEPING_LIMITS,
        roll_natural_period,
    )
    from src.physics.seakeeping.waves import significant_roll_deg
    t_roll = roll_natural_period(dims.beam, large["draft"],
                                 dims.loa, large["gm"])
    hs, tz = DESIGN_SEA_STATE["cargo"]
    roll_deg = significant_roll_deg(hs, tz, t_roll)
    if roll_deg / 2.0 > SEAKEEPING_LIMITS["cargo"]["roll_rms_deg"]:
        return None                     # RMS 정의 (원전 승급)

    # 조종 프록시 (2026-08-11): v3 전체 재검에서 조종이 183/183
    # 학살 (선회지름 6.5L > IMO 5.0L) — fast의 마지막 사각지대.
    # 정상 선회 닫힌 해법 (같은 MMG, 적분 생략 — fsolve 수십 회).
    # 대역 밖(Cb<0.40)은 full 게이트와 동일하게 정직 통과.
    from src.physics.maneuvering.builder import mmg_ship_from_dims
    from src.physics.maneuvering.estimation import EstimationRangeError
    from src.physics.maneuvering.proxy import turning_proxy
    turn_dt = None
    if dims.loa >= 20.0:
        try:
            mmg_ship, _ = mmg_ship_from_dims(
                dims.loa, dims.beam, large["draft"], dims.cb,
                large["total_t"] * 1000.0 / 1025.0,
                large["propeller"]["diameter"],
                large["resistance"]["total"], speed_ms,
                large["propeller"]["ear"], large["propeller"]["z"],
                large["propeller"]["pitch_ratio"])
            tp = turning_proxy(mmg_ship, speed_ms)
            if not tp["passed"]:
                return None
            turn_dt = tp["tactical_diameter_proxy_over_l"]
        except EstimationRangeError:
            pass                        # 대역 밖 — full과 동일 스킵
        except ValueError:
            return None                 # 자항 브래킷 밖 = 실현불가
            # (저항 > 최대 추력 — 게이트 사망, 오류 아님. 정직)

    st = _structure_gate(
        mesh, dims.loa, dims.beam, dims.depth, large["draft"],
        dims.cb, "cargo",
        {"structure": large["lightship_t"]["structure"] * 1e3,
         "outfit": large["lightship_t"]["outfit"] * 1e3,
         "machinery": large["lightship_t"]["machinery"] * 1e3,
         "fuel": large["fuel_t"] * 1e3,
         "payload": large["payload_t"] * 1e3})
    if not st.get("passed"):
        return None

    ec = _economics_gate_large(
        large["engine"]["mcr_kw"], large["engine"]["sfoc_g_per_kwh"],
        large["payload_t"], large["fuel_t"], speed_ms,
        large["engine"]["brake_power_kw"])
    if not ec.get("passed"):
        return None

    return {
        "speed_ms": speed_ms,
        "loa": dims.loa, "beam": dims.beam, "cb": dims.cb,
        "lightship_t": large["lightship_t"]["total"],
        "resistance_n": large["resistance"]["total"],
        "engine": large["engine"]["name"],
        "attained_eedi": ec["attained_g_per_tnm"],
        "eedi_margin_pct": ec["margin_pct"],
        "cii_rating_2026": ec["cii"]["rating_2026"],
        "fuel_cost_usd_per_year": ec["fuel_cost_usd_per_year"],
        "transport_usd_per_tnm": ec["transport_usd_per_tnm"],
        "t_bottom_mm": st["t_bottom_mm"],
        "hold_m3": sp["hold_m3"],
        "hold_margin": sp["margin_ratio"],
        "sig_roll_deg": roll_deg,
        "turn_proxy_dt_over_l": turn_dt,
    }


def optimize_large(payload_kg: float = 8_000_000.0,
                   target_loa: float | None = None,
                   pop_size: int = 24, n_gen: int = 12,
                   seed: int = 1, verbose: bool = False
                   ) -> pd.DataFrame:
    """NSGA-II → 최종 세대 비지배 전선 DataFrame (사망 집계 attrs)."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize

    from data import shipd_loader
    from src.ai.dimension_estimator import estimate_dimensions

    if target_loa is None:
        goal0 = GoalSpec(target_speed_ms=6.0, payload_kg=payload_kg,
                         purpose="cargo")
        target_loa = estimate_dimensions(goal0).loa

    vectors, _ = shipd_loader.load_vectors()
    xl45, xu45 = dataset_bounds()
    xl = np.concatenate([xl45, [SPEED_LO]])
    xu = np.concatenate([xu45, [SPEED_HI]])
    rng = np.random.default_rng(seed)
    picks = vectors[rng.choice(len(vectors), pop_size, replace=False)]
    speeds = rng.uniform(SPEED_LO, SPEED_HI, (pop_size, 1))
    seeds = np.hstack([picks, speeds])          # 실척 시드 관례

    stats = {"constraint": 0, "gate": 0, "error": 0, "alive": 0}

    def _objectives(x: np.ndarray):
        v45 = np.asarray(x[:45], float)
        sp = float(x[45])
        if not constraints_ok(v45):
            stats["constraint"] += 1
            return None
        try:
            r = evaluate_large_vector(v45, sp, payload_kg, target_loa)
        except Exception:
            stats["error"] += 1
            return None
        if r is None:
            stats["gate"] += 1
            return None
        stats["alive"] += 1
        return (r["transport_usd_per_tnm"], -sp)

    class LargeProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=46, n_obj=2, xl=xl, xu=xu)

        def _evaluate(self, x, out, *args, **kwargs):
            f = _objectives(np.asarray(x, float))
            out["F"] = list(f) if f is not None else list(DEATH)

    res = minimize(LargeProblem(),
                   NSGA2(pop_size=pop_size, sampling=seeds),
                   ("n_gen", n_gen), seed=seed, verbose=verbose)

    rows = []
    for x in np.atleast_2d(res.X):
        v45 = np.asarray(x[:45], float)
        sp = float(x[45])
        if not constraints_ok(v45):
            continue
        try:
            r = evaluate_large_vector(v45, sp, payload_kg, target_loa)
        except Exception:
            continue
        if r is not None:
            r["vector_json"] = json.dumps(v45.tolist())
            rows.append(r)
    df = pd.DataFrame(rows).reset_index(drop=True)
    df.attrs["death_stats"] = dict(stats)
    df.attrs["target_loa"] = float(target_loa)
    if verbose:
        print(f"사망 집계: 제약 {stats['constraint']} / 게이트 "
              f"{stats['gate']} / 오류 {stats['error']} / 생존 "
              f"{stats['alive']}")
    return df


def full_recheck(vector_json: str, speed_ms: float,
                 payload_kg: float, target_loa: float) -> dict:
    """전선 대표 설계 전체 8중 재검 — fast에서 생략한 내항·조종
    게이트 추가 (지도 2단 구도의 NSGA판)."""
    from data import shipd_loader
    from src.physics.holtrop import holtrop_input_from_mesh
    from src.physics.seakeeping.criteria import seakeeping_gate
    from src.pipeline import _maneuvering_gate
    from src.pipeline_large import design_spiral_large, dims_from_shipd_mesh

    v45 = np.asarray(json.loads(vector_json), float)
    fast = evaluate_large_vector(v45, speed_ms, payload_kg, target_loa)
    if fast is None:
        return {"passed": False, "note": "fast 게이트 재현 실패"}
    mesh = shipd_loader.scaled_mesh(v45, target_loa)
    h0 = holtrop_input_from_mesh(mesh, target_loa,
                                 float(mesh.bounds[1][2]) / 1.6)
    dims = dims_from_shipd_mesh(mesh, target_loa, h0.cb)
    goal = GoalSpec(target_speed_ms=speed_ms, payload_kg=payload_kg,
                    purpose="cargo")
    large = design_spiral_large(mesh, dims, goal)
    sk = seakeeping_gate(mesh, large["draft"],
                         large["total_t"] * 1000.0,
                         large["total_t"] * 1000.0
                         * (0.25 * dims.loa) ** 2,
                         beam=dims.beam, lwl=dims.loa,
                         gm=large["gm"], purpose="cargo")
    mv = _maneuvering_gate(
        dims.loa, dims.beam, large["draft"], dims.cb,
        large["total_t"] * 1000.0 / 1025.0,
        large["propeller"]["diameter"], large["resistance"]["total"],
        speed_ms, large["propeller"]["ear"], large["propeller"]["z"],
        large["propeller"]["pitch_ratio"])
    passed = bool(fast is not None and sk["passed"] and mv["passed"])
    return {**fast, "seakeeping": sk, "maneuvering": mv,
            "passed": passed}
