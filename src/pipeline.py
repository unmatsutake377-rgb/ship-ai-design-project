"""End-to-End 파이프라인 CLI (spec §3, M3 첫 관통).

흐름: GoalSpec → 치수 추정 → 체계 판정 → Wigley 메쉬 → 중량/KG
      → 평형 흘수 · GM 밴드 · 건현 필터 → 리포트(STL + JSON + 콘솔).

Exit codes: 0 통과 / 2 필터 불합격 / 3 미지원(체계·용도·Cb 범위).
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import json
import sys
from pathlib import Path

from src.ai.dimension_estimator import (
    PayloadInfeasibleError,
    UnknownPurposeError,
    band_report,
    estimate_dimensions,
)
from src.ai.hull_generator import (
    CbOutOfRangeError,
    cm_for_purpose,
    generate_hull_mesh,
    lcb_for_purpose,
    generate_transom_hull_mesh,
    solve_exponents,
    submerged_transom_area,
)
from src.core.regime import (
    Regime,
    UnsupportedRegimeError,
    classify,
    froude_length,
    froude_volumetric,
    max_displacement_speed,
    max_semi_speed,
    min_loa_for_speed,
    require_supported,
)
from src.core.types import GoalSpec
from src.physics.coefficients import estimate_coefficients
from src.physics.hydrostatics import RHO_SEAWATER, SinksError, evaluate
from src.physics.propulsion import (
    NoSuitableMotorError,
    battery_mass,
    select_motors,
)
from src.physics.resistance import total_resistance, total_resistance_semi
from src.physics.seakeeping.criteria import roll_natural_period
from src.physics.savitsky import PlaningEquilibriumError

MAX_SPIRAL_ITER = 12   # 설계 나선 최대 반복
SPIRAL_TOL = 1e-3      # 전체 중량 상대 변화 수렴 기준


class SpiralNotConvergedError(RuntimeError):
    """설계 나선이 수렴하지 않음 — 요구조건 조합이 발산."""
from src.physics.weights import estimate_weights


def design_spiral(mesh, dims, goal: GoalSpec, resistance_fn=None,
                  criteria=None, cm: float | None = None,
                  lcb_frac: float = 0.0):
    """설계 나선: 중량 → 흘수 → 저항 → 모터·배터리 → 중량 … 수렴까지.

    추진계 중량을 고정비율 개략에서 시작해 실측(모터+배터리)으로 수렴.
    반환: (weights, hydro, resist, motors, batt_kg, iteration).
    run_pipeline·최적화기·Ship-D 선별기가 공유하는 평가 코어.

    resistance_fn(mesh, draft, speed, weights) 주입 시 그 경로 사용
    (예: 메쉬형 Michell — Ship-D 임의 형상, Savitsky — 활주).
    기본은 Wigley 해석 경로. criteria로 체계별 안정성 밴드 주입 가능.
    """
    # 저항 경로 정합 (2026-08-05 asym-hull에서 발굴한 혼합 세계 구멍):
    # 해석형 Michell은 대칭 Wigley 전제 + 지수에 cm 반영 필요.
    # 비대칭(lcb≠0)은 해석형 무효 → 메쉬형 Michell(이중검증)로 전환.
    if resistance_fn is None and abs(lcb_frac) > 1e-9:
        from src.physics.resistance import total_resistance_mesh
        resistance_fn = (lambda m_, d_, s_, w_=None:
                         total_resistance_mesh(m_, dims.loa, d_, s_))
    n_exp, m_exp = (solve_exponents(dims.cb, cm) if cm is not None
                    else solve_exponents(dims.cb)) \
        if resistance_fn is None else (None, None)
    propulsion_mass: float | None = None
    prev_total: float | None = None
    for iteration in range(1, MAX_SPIRAL_ITER + 1):
        weights = estimate_weights(float(mesh.area), dims.depth,
                                   goal.payload_kg, propulsion_mass,
                                   loa=dims.loa)
        hydro = evaluate(mesh, weights.total_mass, weights.kg,
                         beam=dims.beam, depth=dims.depth,
                         criteria=criteria)
        if resistance_fn is None:
            resist = total_resistance(mesh, dims, n_exp, m_exp,
                                      draft=hydro.draft,
                                      speed=goal.target_speed_ms)
        else:
            # weights 전달 (C-2): Savitsky는 질량·LCG 의존 — 나선과 결합
            resist = resistance_fn(mesh, hydro.draft, goal.target_speed_ms,
                                   weights)
        motors = select_motors(resist.total)
        batt_kg = battery_mass(resist.effective_power, goal.endurance_h)
        propulsion_mass = motors.total_weight_kg + batt_kg

        if prev_total is not None and \
                abs(weights.total_mass - prev_total) / prev_total < SPIRAL_TOL:
            return weights, hydro, resist, motors, batt_kg, iteration
        prev_total = weights.total_mass
    raise SpiralNotConvergedError(
        f"{MAX_SPIRAL_ITER}회 반복에도 중량이 수렴하지 않음 — "
        "요구조건(적재량·속도·항속시간) 조합을 조정해 주세요."
    )


def _structure_gate(mesh, loa: float, beam: float, depth: float,
                    draft: float, cb: float, purpose: str,
                    component_masses_kg: dict, n_grid: int = 41) -> dict:
    """6번째 게이트 — 종강도 (구조 스펙 2026-08-09 §4).

    파랑 하중: 대형(L≥90) = IACS UR S11 정본, 소형 = 준정적 표준파
    a=L/40 (IACS 범위 밖 — C급 정직 표기). 절단 실패 등 데이터
    부재는 정직 스킵 (데이터 없음 ≠ 불합격)."""
    from src.physics.structure.loads import (
        standard_weight_blocks,
        still_water_curves,
    )
    from src.physics.structure.materials import select_material
    from src.physics.structure.strength import longitudinal_strength
    from src.physics.structure.wave_loads import (
        IACSRangeError,
        iacs_wave_bending_knm,
        quasi_static_wave_moment,
    )
    try:
        xmin = float(mesh.bounds[0][0])
        wl_z = float(mesh.bounds[0][2]) + draft
        blocks = standard_weight_blocks(component_masses_kg, xmin, loa)
        still = still_water_curves(mesh, wl_z, blocks, n=n_grid)
        m_still = float(still.moment_nm[n_grid // 2]) / 1e3   # kN·m
        try:
            hog, sag = iacs_wave_bending_knm(loa, beam, cb)
            wave_src = "IACS UR S11 (극치 설계값)"
        except IACSRangeError:
            amp = loa / 40.0
            h = quasi_static_wave_moment(mesh, wl_z, blocks,
                                         wave_amp=amp, wavelength=loa,
                                         crest_mid=True, n=n_grid)
            s = quasi_static_wave_moment(mesh, wl_z, blocks,
                                         wave_amp=amp, wavelength=loa,
                                         crest_mid=False, n=n_grid)
            hog = h["m_wave_mid_nm"] / 1e3
            sag = s["m_wave_mid_nm"] / 1e3
            wave_src = "준정적 표준파 a=L/40 (IACS 범위 밖 — C급)"
        rep = longitudinal_strength(loa, beam, depth, draft,
                                    m_still, hog, sag,
                                    select_material(loa, purpose))
        rep["m_still_knm"] = m_still
        rep["m_wave_hog_knm"] = hog
        rep["m_wave_sag_knm"] = sag
        rep["wave_source"] = wave_src
        return rep
    except Exception as e:                       # 절단 실패 등
        return {"passed": True, "skipped": True,
                "note": f"구조 게이트 스킵 — {type(e).__name__}: {e} "
                        "(데이터 없음 ≠ 불합격, 정직 기록)"}


def _maneuvering_gate(loa: float, beam: float, draft: float,
                      cb: float, displacement_m3: float,
                      d_prop: float, resistance_n: float,
                      speed_ms: float, ear: float, z: int,
                      pitch_ratio: float) -> dict:
    """7번째 게이트 — 조종성 (조종 스펙 2026-08-09 §4).

    MMG 계수 추정 (Kijima+Yoshimura 회귀) → 표준 시험 (35° 선회·
    지그재그) → IMO MSC.137(76) 판정 (L ≥ 100 m — 범위 밖은
    성적표만). 정직 스킵 2종: Cb<0.40 (외삽 한계) · L<20 m (상선
    회귀 무의미 — 소형은 기존 민첩성 지표 담당)."""
    if loa < 20.0:
        return {"passed": True, "skipped": True, "applicable": False,
                "note": "조종 게이트 스킵 — L<20 m 소형은 상선 MMG "
                        "회귀 밖 (기존 민첩성 지표 담당, 정직)"}
    from src.physics.maneuvering.criteria import (
        maneuvering_gate,
        maneuvering_report,
    )
    from src.physics.maneuvering.estimation import (
        EstimationRangeError,
        estimate_mmg_coeffs,
        fujii_lift_gradient,
    )
    from src.physics.maneuvering.kvlcc2 import ShipParticulars
    from src.physics.maneuvering.mmg import MMGShip, solve_self_propulsion
    from src.physics.propeller import (
        kt_kq,
        thrust_deduction,
        wake_fraction,
    )
    from src.sim_adapters.rudder import rudder_area_dnv
    try:
        ar = rudder_area_dnv(loa, beam, draft)
        hr = math.sqrt(1.5 * ar)          # 종횡비 1.5 통상 (C급)
        w0 = wake_fraction(cb)
        t0 = thrust_deduction(cb)
        # Kt(J) 2차 다항 — 우리 Wageningen B 실계수 3점 정합
        kt0 = kt_kq(0.0, pitch_ratio, ear, z)[0]
        kt5 = kt_kq(0.5, pitch_ratio, ear, z)[0]
        kt1 = kt_kq(1.0, pitch_ratio, ear, z)[0]
        k0 = kt0
        k2 = 2.0 * (kt1 + kt0 - 2.0 * kt5)
        k1 = kt1 - kt0 - k2
        co, notes = estimate_mmg_coeffs(
            loa=loa, beam=beam, draft=draft, cb=cb,
            displacement_m3=displacement_m3, xg=0.0,
            dp=d_prop, hr=hr, ar=ar, w_p0=w0, t_p=t0,
            k0=k0, k1=k1, k2=k2)
        par = ShipParticulars(
            lpp=loa, beam=beam, draft=draft,
            displacement_m3=displacement_m3, xg=0.0, cb=cb,
            dp=d_prop, hr=hr, ar=ar, rho=1025.0)
        n_p = solve_self_propulsion(par, co, speed_ms, resistance_n)
        ship = MMGShip(par=par, co=co, r0_n=resistance_n, n_p=n_p,
                       u0=speed_ms)
        rep = maneuvering_report(ship, speed_ms)
        gate = maneuvering_gate(rep, loa)
        return {**rep, **gate, "coeff_grade": notes["grade"],
                "coeff_note": notes["note"],
                "passed": (gate["passed"] if gate["applicable"]
                           else True),
                "skipped": False}
    except EstimationRangeError as e:
        return {"passed": True, "skipped": True, "applicable": False,
                "note": f"조종 게이트 스킵 — {e}"}
    except Exception as e:
        return {"passed": True, "skipped": True, "applicable": False,
                "note": f"조종 게이트 스킵 — {type(e).__name__}: {e} "
                        "(데이터 없음 ≠ 불합격, 정직 기록)"}


def _economics_gate_large(mcr_kw: float, sfoc_g_per_kwh: float,
                          payload_t: float, fuel_t: float,
                          v_service_ms: float,
                          p_service_kw: float) -> dict:
    """8번째 게이트 — EEDI + 운항 경제 (경제 스펙 2026-08-09 §2).

    DWT ≈ 짐 + 연료 (창고품 생략 개략). DWT < 3,000 = 규제 적용
    밖 (성적표만 — 정직)."""
    from src.physics.economics.eedi import (
        attained_eedi,
        eedi_verdict,
        required_eedi,
    )
    from src.physics.economics.opex import fuel_opex
    dwt = payload_t + fuel_t
    # 판정 = EPL(기관 출력 제한) 기준 — 카탈로그 이산성으로 과대
    # 설치된 MCR을 봉인해 EEDI 대응하는 실업계 표준 (C급 관례).
    # MCR_lim = P_service/0.75 → P75 = P_service, Vref = 설계 속도.
    mcr_epl = p_service_kw / 0.75
    att = attained_eedi(min(mcr_kw, mcr_epl), sfoc_g_per_kwh, dwt,
                        v_service_ms, p_service_kw)
    att_installed = attained_eedi(mcr_kw, sfoc_g_per_kwh, dwt,
                                  v_service_ms, p_service_kw)
    req = required_eedi(dwt)
    ver = eedi_verdict(att["eedi_g_per_tnm"],
                       req["required_g_per_tnm"])
    opx = fuel_opex(p_service_kw, sfoc_g_per_kwh, v_service_ms, dwt)
    from src.physics.economics.cii import (
        attained_aer,
        cii_outlook,
        cii_rating,
        required_cii,
    )
    aer = attained_aer(opx["fuel_t_per_year"], dwt,
                       opx["distance_nm_per_year"])
    req_cii = required_cii(dwt, 2026)
    rating = cii_rating(aer["aer_g_per_dwt_nm"],
                        req_cii["required_g_per_dwt_nm"])
    cii = {
        "aer_g_per_dwt_nm": aer["aer_g_per_dwt_nm"],
        "required_2026_g_per_dwt_nm": req_cii["required_g_per_dwt_nm"],
        "rating_2026": rating["rating"],
        "outlook": cii_outlook(aer["aer_g_per_dwt_nm"], dwt),
        "note": "운항 지표 — 게이트 아님 (성적표)·설계 속도 만재 "
                "가동 가정; " + rating["note"],
    }
    return {
        "kind": "eedi",
        "cii": cii,
        "dwt_t": dwt,
        "attained_g_per_tnm": att["eedi_g_per_tnm"],
        "attained_installed_g_per_tnm":
            att_installed["eedi_g_per_tnm"],
        "epl_applied": bool(mcr_epl < mcr_kw),
        "v_ref_kn": att["v_ref_kn"],
        "required_g_per_tnm": req["required_g_per_tnm"],
        "reduction_pct": req["reduction_pct"],
        "applicable": req["applicable"],
        "margin_pct": ver["margin_pct"],
        "fuel_t_per_year": opx["fuel_t_per_year"],
        "fuel_cost_usd_per_year": opx["fuel_cost_usd_per_year"],
        "transport_usd_per_tnm": opx["transport_usd_per_tnm"],
        "passed": (ver["passed"] if req["applicable"] else True),
        "note": req["note"] + "; " + opx["note"]
                + "; 판정 = EPL(출력 제한) 기준 — 카탈로그 이산성 "
                  "대응 실업계 관례 (설치 MCR 기준 병기, C급)",
    }


def _economics_report_small(effective_power_w: float,
                            v_service_ms: float,
                            payload_kg: float) -> dict:
    """소형 전기 등가 성적표 — Wh/(kg·km)·전기료 (규제 아님)."""
    from src.physics.economics.opex import electric_transport
    from src.physics.propulsion import PROP_EFFICIENCY
    p_elec = effective_power_w / PROP_EFFICIENCY
    rep = electric_transport(p_elec, v_service_ms, payload_kg)
    return {"kind": "electric", "p_electric_w": p_elec,
            "passed": True, "applicable": False, **rep}


def run_pipeline(goal: GoalSpec, out_dir: str | Path,
                 loa: float | None = None,
                 payload_volume: float | None = None,
                 cm_override: float | None = None,
                 hull_source: str = "auto",
                 shipd_pool: int = 60,
                 seakeeping: bool = True,
                 structure: bool = True,
                 maneuvering: bool = True,
                 economics: bool = True,
                 catamaran: bool = False) -> dict:
    """payload_volume [m³]: 짐 부피 직접 입력 (생략 시 용도별 밀도 환산).

    hull_source (스펙 shipgen 4단계 — 수식 생성기 강등):
    - "auto": 배수량 체계 + Ship-D 로컬 있으면 실척 선별(shipd),
      아니면 수식 폴백 — 기본값
    - "shipd": 실척 선별 강제 (불가하면 수식 폴백 + 리포트 표기)
    - "formula": 기존 수식 생성기 (검증 표준기)
    반배수량·활주는 항상 수식 (Savitsky 계열 유지 — 오너 철학).
    shipd_pool: 선별 표본 크기 (클수록 좋은 배·느림, 60 ≈ 1분)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dims = estimate_dimensions(goal, loa=loa)
    volume_est = dims.cb * dims.loa * dims.beam * dims.draft_design
    regime = classify(goal.target_speed_ms, dims.loa, volume_est)

    # 대형 상선 경로 (전 크기 개방 4단계): Watson 유효 대역(60 m+)은
    # 크기 유효 법칙 나선 (Holtrop·Watson·Wageningen B·IMO·ICLL).
    # 크기 스위치가 아니라 법칙 유효성 분기 (스펙 §1).
    from src.pipeline_large import LARGE_LOA_MIN
    if regime is Regime.DISPLACEMENT and dims.loa >= LARGE_LOA_MIN:
        from src.pipeline_large import design_spiral_large
        mesh = generate_hull_mesh(dims, cm=cm_for_purpose(goal.purpose,
                                                          cm_override),
                                  lcb_frac=lcb_for_purpose(goal.purpose))
        large = design_spiral_large(mesh, dims, goal)
        src_used, hull_id, hull_note = "formula", -1, None
        if hull_source in ("auto", "shipd"):
            # Ship-D 대형 접속 (08-07): 실척 후보를 대형 나선으로 실측
            # 평가 → 수식과 A/B (cargo 가중: 저항·경하 반반 — 안정은
            # 게이트 몫). 이긴 쪽 채택 + 사유 병기 (품질 방어 관례).
            from data import shipd_loader
            if shipd_loader.available():
                from src.pipeline_large import (
                    dims_from_shipd_mesh,
                    select_hull_large,
                )
                pick = select_hull_large(goal, dims.loa,
                                         pool_size=shipd_pool // 2)
                if pick is not None:
                    vec, hid, r_sd = pick
                    rel = (0.5 * r_sd["resistance"]["total"]
                           / max(large["resistance"]["total"], 1e-9)
                           + 0.5 * r_sd["lightship_t"]["total"]
                           / max(large["lightship_t"]["total"], 1e-9))
                    if rel < 1.0 or hull_source == "shipd":
                        mesh = shipd_loader.scaled_mesh(vec, dims.loa)
                        from src.physics.holtrop import (
                            holtrop_input_from_mesh,
                        )
                        h0 = holtrop_input_from_mesh(
                            mesh, dims.loa,
                            float(mesh.bounds[1][2]) / 1.6)
                        dims = dims_from_shipd_mesh(mesh, dims.loa, h0.cb)
                        large = r_sd
                        src_used, hull_id = "shipd", hid
                        hull_note = f"실척이 수식 대비 종합 {rel:.2f}배"
                    else:
                        hull_note = (f"실척 후보(hull {hid}) 종합 "
                                     f"{rel:.2f}배 열세 — 수식 유지")
        mesh.export(out / "hull.stl")
        report = {"goal": dataclasses.asdict(goal),
                  "dimensions": dataclasses.asdict(dims),
                  "regime": regime.name, "hull_source": src_used,
                  "hull_id": hull_id, "hull_note": hull_note,
                  "hull_family": "large_cargo",
                  "froude_length": froude_length(goal.target_speed_ms,
                                                 dims.loa),
                  "large": large,
                  "seakeeping": (None if not seakeeping else
                                 __import__(
                                     "src.physics.seakeeping.criteria",
                                     fromlist=["seakeeping_gate"]
                                 ).seakeeping_gate(
                                     mesh, large["draft"],
                                     large["total_t"] * 1000.0,
                                     large["total_t"] * 1000.0
                                     * (0.25 * dims.loa) ** 2,
                                     beam=dims.beam, lwl=dims.loa,
                                     gm=large["gm"],
                                     purpose=goal.purpose)),
                  "space_large": __import__(
                      "src.physics.cargo_capacity",
                      fromlist=["space_gate_large"]).space_gate_large(
                          mesh, dims.depth, dims.loa,
                          fuel_t=large["fuel_t"],
                          payload_t=large["payload_t"]),
                  "structure": (None if not structure else
                                _structure_gate(
                                    mesh, dims.loa, dims.beam,
                                    dims.depth, large["draft"],
                                    dims.cb, goal.purpose,
                                    {"structure":
                                     large["lightship_t"]["structure"]
                                     * 1e3,
                                     "outfit":
                                     large["lightship_t"]["outfit"]
                                     * 1e3,
                                     "machinery":
                                     large["lightship_t"]["machinery"]
                                     * 1e3,
                                     "fuel": large["fuel_t"] * 1e3,
                                     "payload":
                                     large["payload_t"] * 1e3})),
                  "maneuvering": (None if not maneuvering else
                                  _maneuvering_gate(
                                      dims.loa, dims.beam,
                                      large["draft"], dims.cb,
                                      large["total_t"] * 1000.0
                                      / 1025.0,
                                      large["propeller"]["diameter"],
                                      large["resistance"]["total"],
                                      goal.target_speed_ms,
                                      large["propeller"]["ear"],
                                      large["propeller"]["z"],
                                      large["propeller"]["pitch_ratio"])),
                  "economics": (None if not economics else
                                _economics_gate_large(
                                    large["engine"]["mcr_kw"],
                                    large["engine"]["sfoc_g_per_kwh"],
                                    large["payload_t"],
                                    large["fuel_t"],
                                    goal.target_speed_ms,
                                    large["engine"]["brake_power_kw"])),
                  "passed": large["passed"],
                  "roll_period_s": roll_natural_period(
                      dims.beam, large["draft"], dims.loa, large["gm"]),
                  "mesh_file": "hull.stl"}
        report["passed"] = bool(report["passed"]
                                and report["space_large"]["passed"])
        if report["structure"] is not None:
            report["passed"] = bool(report["passed"]
                                    and report["structure"]["passed"])
        if report["maneuvering"] is not None:
            report["passed"] = bool(report["passed"]
                                    and report["maneuvering"]["passed"])
        if report["economics"] is not None:
            report["passed"] = bool(report["passed"]
                                    and report["economics"]["passed"])
        with open(out / "report.json", "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        _print_summary_large(report)
        return report
    vmax = max_displacement_speed(dims.loa)
    require_supported(regime)  # C-2로 전 체계 개방 — 향후 신규 체계 게이트

    # 체계별 경로: 배수량 = Wigley + 해석 Michell / 반배수량 = 트랜섬 +
    # 메쉬 Michell·기저항 (C-1) / 활주 = 데드라이즈 프리즘 + Savitsky (C-2)
    criteria = None  # 기본 밴드 — 활주 분기에서만 교체
    hull_cm = None
    hull_lcb = None
    picked = None    # Ship-D 실척 채택 (배수량 분기에서만)
    hull_note = None  # 선형 출처 판정 사유 (rel = 수식 대비 비율)
    spiral_kw: dict = {}
    if regime is Regime.SEMI_DISPLACEMENT:
        # 트랜섬 선저 배선 (백로그 이행): patrol 실측 Cm 0.80 —
        # 단 기하 도달 범위로 정직 클램프 (날씬 Cb + 풍만 요청 충돌 시
        # 무조건 배선하면 CbOutOfRange 전멸 — hull_note에 표기)
        from src.ai.hull_generator import (
            clamp_transom_cm,
            transom_cm_for_purpose,
        )
        _cm_req = transom_cm_for_purpose(goal.purpose, cm_override)
        hull_cm, _clamped = clamp_transom_cm(dims.cb, _cm_req)
        if _clamped:
            hull_note = (f"요청 Cm {_cm_req:.2f}는 이 Cb({dims.cb:.2f})의 "
                         f"트랜섬 기하 밖 — {hull_cm:.2f}로 정직 클램프")
        mesh = generate_transom_hull_mesh(dims, cm=hull_cm)
        n_exp = m_exp = None

        def resistance_fn(m_, d_, s_, w_=None):
            return total_resistance_semi(
                m_, dims.loa, d_, s_, submerged_transom_area(dims, d_))
    elif regime is Regime.PLANING:
        import math as _math

        from src.ai.hull_generator import (
            PLANING_DEADRISE_DEG,
            generate_planing_hull_mesh,
        )
        from src.physics.hydrostatics import StabilityCriteria
        from src.physics.resistance import ResistanceReport
        from src.physics.savitsky import solve_equilibrium

        mesh = generate_planing_hull_mesh(dims)
        n_exp = m_exp = None
        # 활주정은 얕은 흘수·넓은 수선면이라 정지 GM/B가 0.5~1.5로
        # 원래 큼 (BM = Ixx/∇, 흘수↓ → ∇↓ → BM↑). 배수량형의 상한
        # 0.40(횡요 안락 기준)을 그대로 쓰면 전 활주 설계가 '너무
        # 뻣뻣'으로 탈락 — 상한만 완화. 하한(복원력 최소)은 유지.
        criteria = StabilityCriteria(gm_over_beam=(0.04, 1.50))

        def resistance_fn(m_, d_, s_, w_):
            st = solve_equilibrium(
                weight_n=w_.total_mass * 9.81, speed=s_, beam=dims.beam,
                deadrise_deg=PLANING_DEADRISE_DEG,
                lcg_from_transom=dims.loa / 2.0 + w_.lcg)
            return ResistanceReport(
                speed=s_, froude=froude_length(s_, dims.loa),
                reynolds=s_ * dims.loa / 1.19e-6,
                wetted_area=st.wetted_length * dims.beam
                / _math.cos(_math.radians(PLANING_DEADRISE_DEG)),
                cf=0.0, form_factor=0.0,
                rf=st.friction_n, rw=st.induced_n,
                total=st.resistance_n,
                effective_power=st.resistance_n * s_)
    else:
        if hull_source in ("auto", "shipd"):
            from data import shipd_loader
            if shipd_loader.available():
                from src.ai.shipgen_select import select_hull
                picked = select_hull(goal, dims.loa, pool_size=shipd_pool)
        if picked is not None and hull_source == "auto":
            # 품질 방어 (08-05 실측): 무작위 표본 선별은 z-상대평가라
            # "안정 좋고 저항 폭탄"이 이길 수 있음 (실측 6배 저항 사례).
            # 수식 기준선을 먼저 재고 — 용도 가중 종합에서 실척이
            # 이길 때만 교체. 지면 수식 유지 + 사유 표기 (정직 강등).
            # "shipd" 명시는 사용자 강제 — 방어 없이 그대로 채택.
            from src.ai.shipgen_select import condition_weights
            _cm0 = cm_for_purpose(goal.purpose, cm_override)
            _lcb0 = lcb_for_purpose(goal.purpose)
            _mesh0 = generate_hull_mesh(dims, cm=_cm0, lcb_frac=_lcb0)
            try:
                _w0, _h0, _r0, _, _, _ = design_spiral(
                    _mesh0, dims, goal, criteria=criteria,
                    cm=_cm0, lcb_frac=_lcb0)
                _wt = condition_weights(goal.purpose, goal.target_speed_ms,
                                        dims.loa)
                # 안정은 양쪽 다 게이트(GM 밴드) 통과 — 성능 비교는
                # 저항·중량만 (게이트 몫을 점수로 이중 계상 않는 교훈)
                wr, wm = _wt["resistance"], _wt["mass"]
                rel = (wr * picked.row["resistance_n"] / max(_r0.total, 1e-9)
                       + wm * picked.row["total_mass_kg"]
                       / max(_w0.total_mass, 1e-9)) / (wr + wm)
                # rel < 1 = 실척 종합 우세 (비율 가중합, 수식 = 1 기준)
                if rel >= 1.0:
                    hull_note = (f"실척 후보(hull {picked.hull_id}) 종합 "
                                 f"{rel:.2f}배 열세 — 수식 유지")
                    picked = None
                else:
                    hull_note = f"실척이 수식 대비 종합 {rel:.2f}배 (우세)"
            except Exception:
                pass   # 수식 기준선 실패 = 비교 불가 — 실척 그대로
        if picked is not None:
            # Ship-D 실척 경로 (스펙 4단계): 조건부 선별 승자를 채택.
            # 치수·평가 전부 실척 메쉬 기반 (screen_shipd 관례),
            # 저항 = 메쉬 Michell (비대칭 실선형이라 해석 지수 없음).
            from src.physics.resistance import total_resistance_mesh
            from src.screen_shipd import DEPTH_OVER_DRAFT_FALLBACK

            from data.shipd_loader import scaled_mesh
            mesh = scaled_mesh(picked.vector, dims.loa)
            beam = float(mesh.extents[1])
            depth = float(mesh.bounds[1][2])
            dims = dataclasses.replace(
                dims, beam=beam, depth=depth,
                draft_design=depth / DEPTH_OVER_DRAFT_FALLBACK)
            n_exp = m_exp = None
            hull_cm = None
            hull_lcb = None

            def resistance_fn(m_, d_, s_, w_=None):
                return total_resistance_mesh(m_, dims.loa, d_, s_)
        elif catamaran:
            # 쌍동 갈래 (2단계, 스펙 2026-08-10): 데미헐 병합 메쉬,
            # 저항 = 데미헐 메쉬형 Michell × 2 (간섭 무시 C급),
            # GM/B 상한 해제 (쌍동 과복원 = 정상 — 평행축)
            from src.ai.catamaran import (
                demihull_dims,
                generate_catamaran_mesh,
            )
            from src.physics.hydrostatics import StabilityCriteria
            from src.physics.resistance import total_resistance_mesh
            mesh = generate_catamaran_mesh(dims, separation_ratio=0.7)
            _demi = demihull_dims(dims)

            def resistance_fn(m_, d_, s_, w_=None):
                import dataclasses as _dc
                from src.ai.hull_generator import generate_hull_mesh                     as _g
                r1 = total_resistance_mesh(_g(_demi), dims.loa, d_, s_)
                return _dc.replace(
                    r1, rf=2.0 * r1.rf, rw=2.0 * r1.rw,
                    total=2.0 * r1.total,
                    effective_power=2.0 * r1.effective_power)

            criteria = StabilityCriteria(gm_over_beam=(0.04, 10.0))
            spiral_kw = {}
            n_exp, m_exp = None, None      # 계수 추정은 메쉬 경로
        else:
            hull_cm = cm_for_purpose(goal.purpose, cm_override)
            hull_lcb = lcb_for_purpose(goal.purpose)  # 비대칭 기본 (08-05)
            mesh = generate_hull_mesh(dims, cm=hull_cm, lcb_frac=hull_lcb)
            spiral_kw = {"cm": hull_cm, "lcb_frac": hull_lcb}
            n_exp, m_exp = solve_exponents(dims.cb, hull_cm)
            resistance_fn = None

    weights, hydro, resist, motors, batt_kg, iteration = \
        design_spiral(mesh, dims, goal, resistance_fn=resistance_fn,
                      criteria=criteria, **spiral_kw)

    payload_lcg_frac = None
    try:
        from src.physics.hydrostatics import (
            longitudinal_properties,
            trim_angle_deg,
        )
        from src.physics.weights import estimate_weights as _ew

        _lcb_x, _bml = longitudinal_properties(mesh, hydro.draft)
        static_trim_deg = trim_angle_deg(
            lcg_x=weights.lcg, lcb_x=_lcb_x, kb=hydro.kb, bml=_bml,
            kg=weights.kg)
        # 짐 배치 균형 처방 (실선 방식): LCG=LCB가 되는 짐 위치 역산.
        # 질량·흘수·KG 불변 (x만 이동)이라 1회 재계산으로 정확.
        if static_trim_deg is not None and weights.payload_mass > 0:
            # 구조 LCG=0(중앙) 가정 유지 — 항: 추진(−0.45L)만 상쇄
            x_p = (_lcb_x * weights.total_mass
                   - weights.propulsion_mass * (-0.45 * dims.loa)) \
                / weights.payload_mass
            payload_lcg_frac = float(x_p / dims.loa)
            # 선체 안 제약 (±0.35L) — 밖이면 잔여 트림 정직 표기
            payload_lcg_frac = max(-0.35, min(0.35, payload_lcg_frac))
            weights = _ew(float(mesh.area), dims.depth, goal.payload_kg,
                          weights.propulsion_mass, loa=dims.loa,
                          payload_lcg_frac=payload_lcg_frac)
            static_trim_deg = trim_angle_deg(
                lcg_x=weights.lcg, lcb_x=_lcb_x, kb=hydro.kb, bml=_bml,
                kg=weights.kg)
    except Exception:
        static_trim_deg = None

    coeff_resistance = (None if resistance_fn is None else
                        (lambda m_, d_, s_: resistance_fn(m_, d_, s_,
                                                          weights)))
    coeffs = estimate_coefficients(
        dims=dims, draft=hydro.draft, mass=weights.total_mass,
        lcg=weights.lcg, speed=goal.target_speed_ms,
        mesh=mesh, n_exp=n_exp, m_exp=m_exp,
        resistance_fn=coeff_resistance,
    )

    # MaxBox 공간 검사 (#27): 무게(아르키메데스)와 별개로 "부피가
    # 물리적으로 들어가나" — 정역학과 책임 분리해 이 층에서 합성
    from src.physics.maxbox import largest_box, payload_volume_for

    pv, pv_basis = payload_volume_for(goal.payload_kg, goal.purpose,
                                      payload_volume)
    if picked is not None:
        # Ship-D 메쉬는 비수밀(갑판 열림) — largest_box의 contains
        # 검사가 0을 뱉는 그 버그 (다구획 스펙 §7). 선별 게이트가
        # 이미 검증한 다구획 절단법 결과를 정본으로 사용.
        # 단 payload_volume 직접 입력은 선별 게이트가 모름 — 여기서
        # 다구획 총부피와 재대조 (50 m³ 괴물 거부 시험이 파수꾼).
        space_ok = bool(picked.row["space_ok"]) \
            and pv <= picked.row["hold_volume_m3"]
        maxbox_report = {
            "volume": picked.row["hold_volume_m3"],
            "n_bays": picked.row["n_bays"], "payload_volume": pv,
            "volume_basis": pv_basis,
            "margin_ratio": (picked.row["hold_volume_m3"] - pv) / pv
            if pv > 0 else float("inf"),
            "note": "다구획 절단법 (Ship-D 비수밀 메쉬 — 단일 상자 "
                    "contains 부적용, 다구획 스펙 §7)",
        }
    else:
        box = largest_box(mesh, depth=dims.depth)
        space_ok = pv <= box.volume
        maxbox_report = {
            "length": box.length, "width": box.width, "height": box.height,
            "volume": box.volume, "payload_volume": pv,
            "volume_basis": pv_basis,
            "margin_ratio": (box.volume - pv) / pv if pv > 0
            else float("inf"),
            "note": "내부 구조물(모터·배터리 자리) 미차감 — 비보수적 "
                    "(스펙 §5)",
        }

    # 조종성 지표 (민첩성 — 오너 발견의 정량화, 2026-08-02):
    # "충분하면 됨" 결정 → 문턱 표기만 (경고 수준, 필터 아님)
    from src.physics.agility import agility_metrics
    from src.sim_adapters.python_sim import THRUSTER_SEP_OVER_B

    ag = agility_metrics(
        izz_total=weights.izz + coeffs.nr_dot,
        m_x=weights.total_mass + coeffs.xu_dot,
        yv=abs(coeffs.yv), nv=abs(coeffs.nv), nr=abs(coeffs.nr),
        thrust_max=float(motors.motor["thrust_max_n"]),
        thruster_sep=THRUSTER_SEP_OVER_B * dims.beam,
        speed=goal.target_speed_ms, loa=dims.loa)
    agility_report = {**dataclasses.asdict(ag),
                      "note": "IMO 5L 기준은 대형선 외삽 + Clarke 계수 "
                              "외삽 (이중 외삽) — 경고 표기용, 필터 아님"}

    seakeeping_rep = None
    if seakeeping and catamaran:
        seakeeping_rep = {"passed": True, "skipped": True,
                          "note": "쌍동 내항 스킵 — 스트립 2D 계수는 "
                                  "단동 단면(Lewis/Frank) 가정 밖 "
                                  "(쌍동 내항 물리는 후속 단계, 정직)"}
    elif seakeeping and regime is Regime.DISPLACEMENT:
        # 5번째 게이트 (내항성 4단계): 설계 해상에서 유의 응답 합불
        from src.physics.seakeeping.criteria import seakeeping_gate
        try:
            seakeeping_rep = seakeeping_gate(
                mesh, hydro.draft, weights.total_mass, weights.izz,
                beam=dims.beam, lwl=dims.loa, gm=hydro.gm,
                purpose=goal.purpose)
        except Exception as e:
            seakeeping_rep = {"passed": True,
                              "note": f"내항 계산 실패 — 게이트 보류: {e}"}

    structure_rep = None
    if structure and regime is Regime.DISPLACEMENT:
        # 6번째 게이트 (구조 3단계): 종강도 — 소형은 준정적 표준파
        structure_rep = _structure_gate(
            mesh, dims.loa, dims.beam, dims.depth, hydro.draft,
            dims.cb, goal.purpose,
            {"structure": weights.structure_mass,
             "payload": weights.payload_mass,
             "machinery": weights.propulsion_mass})

    economics_rep = None
    if economics:
        # 8번째 축 — 소형은 전기 등가 성적표 (규제 아님, 정직)
        economics_rep = _economics_report_small(
            resist.effective_power, goal.target_speed_ms,
            goal.payload_kg)

    mesh_file = "hull.stl"
    mesh.export(out / mesh_file)

    report = {
        "goal": dataclasses.asdict(goal),
        "dimensions": dataclasses.asdict(dims),
        "dimension_basis": band_report(goal.purpose),
        "regime": regime.name,
        # 수식 생성기 강등 (스펙 shipgen 4단계): shipd = 실척 선별 채택.
        # ⚠ shipd 메쉬 STL은 라이선스 확정 전 커밋·공개 금지 (로컬 열람만)
        "hull_source": "shipd" if picked is not None else "formula",
        "hull_id": picked.hull_id if picked is not None else -1,
        "hull_note": hull_note,
        "hull_cm": hull_cm,
        # 정적 트림 (2026-08-05 asym-hull): LCB≠0 세계의 실물리 균형.
        # 소각 선형 θ=(LCG−LCB)/GML (다구획 3단계 도구 재사용).
        # 구식 trim_warning(LCB=0 가정)은 weights에 잔존 — 이 값이 정본.
        "static_trim_deg": static_trim_deg,
        "payload_lcg_frac": payload_lcg_frac,
        "hull_lcb_frac": hull_lcb,   # 수식 배수량만 값 — shipd·타 체계 None
        "froude_length": froude_length(goal.target_speed_ms, dims.loa),
        "froude_volumetric": froude_volumetric(goal.target_speed_ms, volume_est),
        "max_displacement_speed": vmax,
        "max_semi_speed": max_semi_speed(dims.loa),
        "hull_family": ("catamaran" if catamaran else
                        {"SEMI_DISPLACEMENT": "transom",
                         "PLANING": "planing_deadrise"}.get(
            regime.name, "shipd" if picked is not None else "wigley")),
        "catamaran": ({"separation_ratio": 0.7,
                       "note": "쌍동 1~2단계 — 저항 간섭 무시 (C급)·"
                               "내항 스트립 단동 단면 가정 밖 스킵"}
                      if catamaran else None),
        "weights": dataclasses.asdict(weights),
        "hydrostatics": dataclasses.asdict(hydro),
        "resistance": dataclasses.asdict(resist),
        "propulsion": {
            **dataclasses.asdict(motors),
            "battery_mass_kg": batt_kg,
            "endurance_h": goal.endurance_h,
            "spiral_iterations": iteration,
        },
        "coefficients": dataclasses.asdict(coeffs),
        "maxbox": maxbox_report,
        "agility": agility_report,
        "checks_space": bool(space_ok),
        "seakeeping": seakeeping_rep,
        "structure": structure_rep,
        "economics": economics_rep,
        "passed": bool(hydro.passed and space_ok
                       and (seakeeping_rep is None
                            or seakeeping_rep["passed"])
                       and (structure_rep is None
                            or structure_rep["passed"])),
        "mesh_file": mesh_file,
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def _print_summary_large(report: dict) -> None:
    d, lg = report["dimensions"], report["large"]
    ls, pr, en = lg["lightship_t"], lg["propeller"], lg["engine"]
    print("=" * 56)
    print("대형 상선 설계 리포트 (전 크기 개방)")
    print("=" * 56)
    print(f"용도/속도/적재  : {report['goal']['purpose']} / "
          f"{report['goal']['target_speed_ms']} m/s / "
          f"{report['goal']['payload_kg'] / 1000:.0f} t")
    print(f"치수 L×B×D (T)  : {d['loa']:.1f} × {d['beam']:.1f} × "
          f"{d['depth']:.1f} ({lg['draft']:.2f}) m, Cb={d['cb']:.2f} "
          f"(Fn={report['froude_length']:.3f})")
    if report.get("hull_source") == "shipd":
        print(f"선형 출처       : Ship-D 실척 (hull {report['hull_id']}) "
              f"⚠ 형상 공개 금지")
    if report.get("hull_note"):
        print(f"출처 판정       : {report['hull_note']}")
    print(f"중량 [t]        : 경하 {ls['total']:.0f} (구조 "
          f"{ls['structure']:.0f}/의장 {ls['outfit']:.0f}/기관 "
          f"{ls['machinery']:.0f}) + 연료 {lg['fuel_t']:.0f} + 짐 "
          f"{lg['payload_t']:.0f} = {lg['total_t']:.0f}")
    print(f"저항 (Holtrop)  : {lg['resistance']['total'] / 1e3:.0f} kN "
          f"(점성 {lg['resistance']['rv'] / 1e3:.0f} + 조파 "
          f"{lg['resistance']['rw'] / 1e3:.0f})")
    print(f"프로펠러 (B{pr['z']}) : D {pr['diameter']:.2f} m · P/D "
          f"{pr['pitch_ratio']:.2f} · {pr['rpm']:.0f} rpm · η0 "
          f"{pr['eta0']:.3f} · ηD {pr['eta_d']:.3f} · 캐비테이션 "
          f"{'OK' if pr['cavitation_ok'] else '⚠'}")
    print(f"엔진            : {en['name']} {en['mcr_kw']:.0f} kW "
          f"(부하율 {en['load_fraction']:.0%}, 등급 {en['source_grade']}"
          f"{'⚠미검증' if en['source_grade'] == 'C' else ''})")
    print(f"GM / 건현       : {lg['gm']:.2f} m (밴드 "
          f"{lg['gm_band'][0]:.3f}~{lg['gm_band'][1]:.2f}·GM/B) / "
          f"{lg['freeboard_m']:.2f} m (ICLL 최소 "
          f"{lg['freeboard_min_icll']:.2f}) → "
          f"{'합격' if report['passed'] else '불합격'}")
    st = report.get("structure")
    if st is not None and not st.get("skipped"):
        print(f"종강도 (6th)    : Z {st['z_deck_m3']:.2f}/"
              f"{st['z_keel_m3']:.2f} ≥ 요구 {st['z_required_m3']:.2f}"
              f" m³ · 판 {st['t_bottom_mm']:.1f}/{st['t_side_mm']:.1f}/"
              f"{st['t_deck_mm']:.1f} mm ({st['material']}) → "
              f"{'합격' if st['passed'] else '불합격'}")
    ec = report.get("economics")
    if ec is not None and ec.get("kind") == "eedi":
        verdict = ("합격" if ec["passed"] else "불합격") \
            if ec["applicable"] else "성적표만"
        print(f"EEDI (8th)      : {ec['attained_g_per_tnm']:.1f} vs "
              f"요구 {ec['required_g_per_tnm']:.1f} g/t·nm (여유 "
              f"{ec['margin_pct']:+.0f}%) · 연료 "
              f"{ec['fuel_t_per_year']:.0f} t/년 "
              f"(${ec['fuel_cost_usd_per_year'] / 1e6:.2f}M) → {verdict}")
    print(f"주의            : {lg['kg_note']}")
    sk = report.get("seakeeping")
    if sk and "sig_roll_deg" in sk:
        print(f"내항 (5게이트)  : Hs {sk['sea_state']['hs_m']} m에서 "
              f"heave {sk['sig_heave_m']:.2f} m·pitch "
              f"{sk['sig_pitch_deg']:.1f}°·roll {sk['sig_roll_deg']:.1f}°"
              f" → {'합격' if sk['passed'] else '불합격'} (C급 기준)")


def _print_summary(report: dict) -> None:
    d = report["dimensions"]
    h = report["hydrostatics"]
    w = report["weights"]
    print("=" * 56)
    print("선박 설계 리포트")
    print("=" * 56)
    print(f"용도/속도/적재  : {report['goal']['purpose']} / "
          f"{report['goal']['target_speed_ms']} m/s / "
          f"{report['goal']['payload_kg']} kg")
    print(f"체계 (Fn)       : {report['regime']} "
          f"(Fn={report['froude_length']:.3f})")
    if report.get("hull_source") == "shipd":
        print(f"선형 출처       : Ship-D 실척 선별 (hull "
              f"{report['hull_id']}) ⚠ 형상 파일 공개 금지 (라이선스)")
    else:
        print("선형 출처       : 수식 생성기 (표준기)")
    if report.get("hull_note"):
        print(f"출처 판정       : {report['hull_note']}")
    print(f"치수 L×B×D (T)  : {d['loa']:.2f} × {d['beam']:.2f} × "
          f"{d['depth']:.2f} ({d['draft_design']:.2f}) m, Cb={d['cb']:.2f}")
    basis = report["dimension_basis"]
    if basis["source"] == "data":
        print(f"치수 근거       : 실선 단동 {basis['n_samples']}척 통계 "
              f"(길이 범위 {basis['loa_range'][0]:.1f}~"
              f"{basis['loa_range'][1]:.1f} m)")
    else:
        print("치수 근거       : ⚠ 개략값 (이 용도는 실선 데이터 미확보)")
    print(f"전체 중량       : {w['total_mass']:.1f} kg "
          f"(구조 {w['structure_mass']:.1f} / 추진 {w['propulsion_mass']:.1f} "
          f"/ 적재 {w['payload_mass']:.1f})")
    print(f"평형 흘수/건현  : {h['draft']:.3f} m / {h['freeboard']:.3f} m")
    print(f"KB / BM / KG    : {h['kb']:.3f} / {h['bm']:.3f} / {h['kg']:.3f} m")
    print(f"GM              : {h['gm']:.3f} m (GM/B="
          f"{h['gm'] / d['beam']:.3f})")
    r = report["resistance"]
    print(f"저항 @목표속도  : 전체 {r['total']:.1f} N "
          f"(마찰 {r['rf']:.1f} + 조파 {r['rw']:.1f})")
    print(f"유효 파워       : {r['effective_power']:.1f} W")
    print(f"한계속도(참고)  : {report['max_displacement_speed']:.2f} m/s "
          f"(이 선체 길이의 배수량형 상한)")
    p = report["propulsion"]
    print(f"권장 모터       : {p['motor']['name']} ({p['motor']['maker']}) "
          f"× {p['count']}발 — 장착 {p['total_thrust_n']:.0f} N, "
          f"사용률 {p['utilization'] * 100:.0f}%, "
          f"모터 중량 {p['total_weight_kg']:.1f} kg")
    print(f"배터리          : {p['battery_mass_kg']:.1f} kg "
          f"(항속 {p['endurance_h']}h 기준, 나선 {p['spiral_iterations']}회 수렴)")
    c = report["coefficients"]
    stable = "안정" if c["straight_line_stable"] else "불안정"
    print(f"동역학 계수     : 횡 부가질량 {c['yv_dot']:.0f} kg · "
          f"직진 {stable} · ⚠ 대형선 회귀 외삽")
    ag = report["agility"]
    if ag["coupled_unstable"]:
        print("조종성(참고)   : 방향 불안정 속도역 — 정상 선회 한계 없음 "
              "(직진 유지가 과제인 배)")
    else:
        print(f"조종성(참고)   : 선회지름 ≈ {ag['diameter_over_l']:.1f}L "
              f"(IMO 대형선 기준 5L "
              f"{'이내 ✓' if ag['within_imo'] else '초과 ⚠'} — 외삽 주의)"
              f" · 반응 {ag['nomoto_t']:.1f}s")
    mb = report["maxbox"]
    if "n_bays" in mb:   # Ship-D 실척 = 다구획 절단법 (상자 치수 없음)
        shape = f"다구획 {mb['n_bays']}칸"
    else:
        shape = (f"MaxBox {mb['length']:.2f}×{mb['width']:.2f}×"
                 f"{mb['height']:.2f} m")
    print(f"탑재 공간 (#27) : {shape} = {mb['volume']:.3f} m³ vs 짐 "
          f"{mb['payload_volume']:.3f} m³ ({mb['volume_basis']}) → "
          f"{'들어감' if report['checks_space'] else '공간 부족 ✗'}"
          f" (여유 {mb['margin_ratio']:+.0%})")
    print(f"필터 판정       : {h['checks']} + "
          f"{{'payload_space': {report['checks_space']}}} → "
          f"{'통과' if report['passed'] else '불합격'}")
    print("=" * 56)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="목적 지향형 선박 설계 PoC")
    parser.add_argument("--speed", type=float, default=None,
                        help="목표 속도 [m/s] (생략 시 용도 프리셋 — "
                             "실선 순항 중앙값)")
    parser.add_argument("--payload", type=float, required=True,
                        help="적재량 [kg]")
    parser.add_argument("--purpose", required=True,
                        help="용도: survey | patrol | workboat")
    parser.add_argument("--out", default="outputs", help="출력 디렉토리")
    parser.add_argument("--loa", type=float, default=None,
                        help="선체 길이 직접 지정 [m] (생략 시 적재량에서 역산)")
    parser.add_argument("--endurance", type=float, default=None,
                        help="항속시간 [h] (생략 시 기본값 — 활주형은 "
                             "전력 소모가 커서 짧게 잡아야 배터리가 가벼움)")
    parser.add_argument("--cm", type=float, default=None,
                        help="중앙단면계수 덮어쓰기 (선저 풍만도 — 기본은 "
                             "용도 프리셋: survey 0.85/workboat 0.92)")
    parser.add_argument("--payload-volume", type=float, default=None,
                        help="짐 부피 [m³] 직접 입력 (생략 시 용도별 "
                             "화물 밀도 개략 가정으로 무게→부피 환산)")
    parser.add_argument("--hull-source", default="auto",
                        choices=["auto", "shipd", "formula"],
                        help="선형 출처 (배수량 체계 한정): auto = Ship-D "
                             "있으면 실척 선별, formula = 수식 표준기")
    parser.add_argument("--shipd-pool", type=int, default=60,
                        help="Ship-D 선별 표본 수 (클수록 좋은 배·느림)")
    parser.add_argument("--catamaran", action="store_true",
                        help="쌍동선(카타마란) 선형 (배수량 소형)")
    parser.add_argument("--no-economics", action="store_true",
                        help="8번째 게이트(EEDI·경제) 생략")
    parser.add_argument("--no-maneuvering", action="store_true",
                        help="7번째 게이트(조종성) 생략")
    parser.add_argument("--no-structure", action="store_true",
                        help="6번째 게이트(종강도) 생략")
    parser.add_argument("--no-seakeeping", action="store_true",
                        help="내항 게이트 생략 (빠른 실행)")
    args = parser.parse_args(argv)

    # 3입력 UX (#25 오너 제안): 속도 생략 시 용도가 결정
    speed = args.speed
    if speed is None:
        from src.ai.presets import purpose_presets

        preset = purpose_presets().get(args.purpose)
        if preset is None:
            print(f"[중단] 알 수 없는 용도: {args.purpose}", file=sys.stderr)
            return 3
        speed = preset.default_speed_ms
        origin = (f"실선 {preset.n_samples}척 순항 중앙값"
                  if preset.speed_source == "data" else "개략 기본값")
        print(f"속도 미지정 → 용도 프리셋 적용: {speed:.2f} m/s ({origin})")

    goal_kwargs = dict(target_speed_ms=speed, payload_kg=args.payload,
                       purpose=args.purpose)
    if args.endurance is not None:
        goal_kwargs["endurance_h"] = args.endurance
    goal = GoalSpec(**goal_kwargs)
    try:
        report = run_pipeline(goal, args.out, loa=args.loa,
                              cm_override=args.cm,
                              payload_volume=args.payload_volume,
                              hull_source=args.hull_source,
                              shipd_pool=args.shipd_pool,
                              seakeeping=not args.no_seakeeping,
                              structure=not args.no_structure,
                              maneuvering=not args.no_maneuvering,
                              economics=not args.no_economics,
                              catamaran=args.catamaran)
    except (UnsupportedRegimeError, UnknownPurposeError, CbOutOfRangeError,
            SinksError, PayloadInfeasibleError, NoSuitableMotorError,
            SpiralNotConvergedError, PlaningEquilibriumError) as e:
        print(f"[중단] {e}", file=sys.stderr)
        return 3
    if "large" not in report:      # 대형 리포트는 run_pipeline이 출력
        _print_summary(report)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
