"""End-to-End 파이프라인 CLI (spec §3, M3 첫 관통).

흐름: GoalSpec → 치수 추정 → 체계 판정 → Wigley 메쉬 → 중량/KG
      → 평형 흘수 · GM 밴드 · 건현 필터 → 리포트(STL + JSON + 콘솔).

Exit codes: 0 통과 / 2 필터 불합격 / 3 미지원(체계·용도·Cb 범위).
"""
from __future__ import annotations

import argparse
import dataclasses
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
    generate_hull_mesh,
    solve_exponents,
)
from src.core.regime import (
    UnsupportedRegimeError,
    classify,
    froude_length,
    froude_volumetric,
    max_displacement_speed,
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
from src.physics.resistance import total_resistance

MAX_SPIRAL_ITER = 12   # 설계 나선 최대 반복
SPIRAL_TOL = 1e-3      # 전체 중량 상대 변화 수렴 기준


class SpiralNotConvergedError(RuntimeError):
    """설계 나선이 수렴하지 않음 — 요구조건 조합이 발산."""
from src.physics.weights import estimate_weights


def design_spiral(mesh, dims, goal: GoalSpec):
    """설계 나선: 중량 → 흘수 → 저항 → 모터·배터리 → 중량 … 수렴까지.

    추진계 중량을 고정비율 개략에서 시작해 실측(모터+배터리)으로 수렴.
    반환: (weights, hydro, resist, motors, batt_kg, iteration).
    run_pipeline과 최적화기(src/optimize.py)가 공유하는 평가 코어.
    """
    n_exp, m_exp = solve_exponents(dims.cb)
    propulsion_mass: float | None = None
    prev_total: float | None = None
    for iteration in range(1, MAX_SPIRAL_ITER + 1):
        weights = estimate_weights(float(mesh.area), dims.depth,
                                   goal.payload_kg, propulsion_mass,
                                   loa=dims.loa)
        hydro = evaluate(mesh, weights.total_mass, weights.kg,
                         beam=dims.beam, depth=dims.depth)
        resist = total_resistance(mesh, dims, n_exp, m_exp,
                                  draft=hydro.draft,
                                  speed=goal.target_speed_ms)
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


def run_pipeline(goal: GoalSpec, out_dir: str | Path,
                 loa: float | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dims = estimate_dimensions(goal, loa=loa)
    volume_est = dims.cb * dims.loa * dims.beam * dims.draft_design
    regime = classify(goal.target_speed_ms, dims.loa, volume_est)
    vmax = max_displacement_speed(dims.loa)
    try:
        require_supported(regime)
    except UnsupportedRegimeError as e:
        # 거절에도 대안 숫자를 준다 (오너 제안 Q2): 이 크기의 한계속도,
        # 이 속도에 필요한 최소 길이
        raise UnsupportedRegimeError(
            e.regime,
            f"{e} | 추정 선체 L={dims.loa:.2f} m의 배수량형 한계속도는 "
            f"{vmax:.2f} m/s입니다. 목표 {goal.target_speed_ms} m/s를 내려면 "
            f"최소 L={min_loa_for_speed(goal.target_speed_ms):.2f} m가 "
            f"필요합니다."
        ) from e

    mesh = generate_hull_mesh(dims)
    n_exp, m_exp = solve_exponents(dims.cb)

    weights, hydro, resist, motors, batt_kg, iteration = \
        design_spiral(mesh, dims, goal)

    coeffs = estimate_coefficients(
        dims=dims, draft=hydro.draft, mass=weights.total_mass,
        lcg=weights.lcg, speed=goal.target_speed_ms,
        mesh=mesh, n_exp=n_exp, m_exp=m_exp,
    )

    mesh_file = "hull.stl"
    mesh.export(out / mesh_file)

    report = {
        "goal": dataclasses.asdict(goal),
        "dimensions": dataclasses.asdict(dims),
        "dimension_basis": band_report(goal.purpose),
        "regime": regime.name,
        "froude_length": froude_length(goal.target_speed_ms, dims.loa),
        "froude_volumetric": froude_volumetric(goal.target_speed_ms, volume_est),
        "max_displacement_speed": vmax,
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
        "passed": hydro.passed,
        "mesh_file": mesh_file,
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


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
    print(f"필터 판정       : {h['checks']} → "
          f"{'통과' if report['passed'] else '불합격'}")
    print("=" * 56)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="목적 지향형 선박 설계 PoC")
    parser.add_argument("--speed", type=float, required=True,
                        help="목표 속도 [m/s]")
    parser.add_argument("--payload", type=float, required=True,
                        help="적재량 [kg]")
    parser.add_argument("--purpose", required=True,
                        help="용도: survey | patrol | workboat")
    parser.add_argument("--out", default="outputs", help="출력 디렉토리")
    parser.add_argument("--loa", type=float, default=None,
                        help="선체 길이 직접 지정 [m] (생략 시 적재량에서 역산)")
    args = parser.parse_args(argv)

    goal = GoalSpec(target_speed_ms=args.speed, payload_kg=args.payload,
                    purpose=args.purpose)
    try:
        report = run_pipeline(goal, args.out, loa=args.loa)
    except (UnsupportedRegimeError, UnknownPurposeError, CbOutOfRangeError,
            SinksError, PayloadInfeasibleError, NoSuitableMotorError,
            SpiralNotConvergedError) as e:
        print(f"[중단] {e}", file=sys.stderr)
        return 3
    _print_summary(report)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
