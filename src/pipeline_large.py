"""대형 상선 설계 나선 (전 크기 개방 4단계, 스펙 2026-08-06 §5).

소형 나선(design_spiral)과 같은 구조, 크기 유효 법칙으로 교체:
- 중량: Watson-Gilfillan 경하 + 연료(SFOC×항속) + 짐
- 저항: Holtrop-Mennen (메쉬 실측 계수)
- 추진: Wageningen B 프로펠러 + 엔진 카탈로그
- 게이트: 평형 흘수·IMO GM 밴드·ICLL 건현

분기 기준은 크기 스위치가 아니라 **법칙 유효 대역** (Watson 60~350 m
— 스펙 §1). KG 개략 0.55D (C급 — 대형 중량 분포모델은 후속).
"""
from __future__ import annotations

from src.core.types import GoalSpec, MainDimensions

MAX_ITER = 30
TOL = 1e-3
# 성분별 VCG/D (상선 통상 대역, B~C급 — Watson 계보 개략):
# 구조 0.60(선각+갑판), 의장 0.85(갑판 위 장비), 기관 0.45(기관실),
# 연료 0.15(이중저·저부 탱크), 짐 0.55(창구 중앙)
VCG_OVER_D = {"structure": 0.60, "outfit": 0.85, "machinery": 0.45,
              "fuel": 0.15, "payload": 0.55}
LARGE_LOA_MIN = 60.0       # Watson 유효 대역 하한


class LargeSpiralError(ValueError):
    """대형 나선 수렴 실패 또는 게이트 위반."""


def design_spiral_large(mesh, dims: MainDimensions, goal: GoalSpec):
    """대형 나선: Watson 중량 ↔ 흘수 ↔ Holtrop ↔ 프로펠러+엔진 수렴.

    반환 dict: 중량 3분해·연료·흘수·GM·건현·저항 성분·프로펠러·엔진·
    게이트 판정 (불합격도 정상 결과 — 사유 병기)."""
    from src.physics.holtrop import (
        holtrop_input_from_mesh,
        total_resistance_holtrop,
    )
    from src.physics.hydrostatics import StabilityCriteria, evaluate
    from src.physics.large_ship import (
        icll_freeboard_m,
        large_gm_band,
        watson_lightship,
    )
    from src.physics.propulsion_large import design_propulsion

    mcr = 0.0
    fuel_t = 0.0
    prev = None
    payload_t = goal.payload_kg / 1000.0
    for iteration in range(1, MAX_ITER + 1):
        ls = watson_lightship(dims.loa, dims.beam, dims.depth,
                              dims.draft_design, dims.cb, mcr_kw=mcr,
                              ship_type="cargo")
        total_t = ls.total_t + fuel_t + payload_t
        crit = StabilityCriteria(gm_over_beam=large_gm_band(dims.beam))
        kg = (dims.depth / max(total_t, 1e-9)) * (
            ls.structure_t * VCG_OVER_D["structure"]
            + ls.outfit_t * VCG_OVER_D["outfit"]
            + ls.machinery_t * VCG_OVER_D["machinery"]
            + fuel_t * VCG_OVER_D["fuel"]
            + payload_t * VCG_OVER_D["payload"])
        hydro = evaluate(mesh, total_t * 1000.0, kg,
                         beam=dims.beam, depth=dims.depth, criteria=crit)
        h_in = holtrop_input_from_mesh(mesh, dims.loa, hydro.draft)
        resist = total_resistance_holtrop(h_in, goal.target_speed_ms)
        prop, engine = design_propulsion(resist["total"],
                                         goal.target_speed_ms, hydro.draft,
                                         cb=dims.cb, holtrop_input=h_in)
        mcr = engine.mcr_kw
        fuel_t = (prop.brake_power_kw * engine.sfoc_g_per_kwh
                  * goal.endurance_h) / 1e6 * 1.10   # 예비 10%
        if prev is not None and abs(total_t - prev) / prev < TOL:
            break
        prev = total_t
    else:
        raise LargeSpiralError(f"{MAX_ITER}회 반복에도 중량 미수렴")

    freeboard = dims.depth - hydro.draft
    fb_min = icll_freeboard_m(dims.loa)
    return {
        "iterations": iteration,
        "lightship_t": {"structure": ls.structure_t, "outfit": ls.outfit_t,
                        "machinery": ls.machinery_t, "total": ls.total_t},
        "fuel_t": fuel_t, "payload_t": payload_t, "total_t": total_t,
        "draft": float(hydro.draft), "gm": float(hydro.gm),
        "gm_band": crit.gm_over_beam, "hydro_passed": bool(hydro.passed),
        "hydro_checks": {k: bool(v) for k, v in hydro.checks.items()},
        "freeboard_m": freeboard, "freeboard_min_icll": fb_min,
        "freeboard_ok": bool(freeboard >= fb_min),
        "resistance": {k: resist[k] for k in
                       ("rf", "rv", "rw", "ra", "total", "ct", "froude")},
        "propeller": {"diameter": prop.diameter,
                      "pitch_ratio": prop.pitch_ratio, "rpm": prop.rpm,
                      "ear": prop.ear, "z": prop.z, "eta0": prop.eta0,
                      "eta_d": prop.eta_d,
                      "cavitation_ok": bool(prop.cavitation_ok)},
        "engine": {"name": engine.name, "mcr_kw": engine.mcr_kw,
                   "mass_t": engine.mass_t,
                   "load_fraction": engine.load_fraction,
                   "source_grade": engine.source_grade,
                   "brake_power_kw": prop.brake_power_kw},
        "passed": bool(hydro.passed and freeboard >= fb_min),
        "kg_m": float(kg),
        "kg_note": "KG = 성분별 VCG/D 합성 (구조 0.60·의장 0.85·"
                   "기관 0.45·연료 0.15·짐 0.55 — B~C급 통상 대역)",
    }
