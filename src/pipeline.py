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
from src.physics.savitsky import PlaningEquilibriumError

MAX_SPIRAL_ITER = 12   # 설계 나선 최대 반복
SPIRAL_TOL = 1e-3      # 전체 중량 상대 변화 수렴 기준


class SpiralNotConvergedError(RuntimeError):
    """설계 나선이 수렴하지 않음 — 요구조건 조합이 발산."""
from src.physics.weights import estimate_weights


def design_spiral(mesh, dims, goal: GoalSpec, resistance_fn=None,
                  criteria=None):
    """설계 나선: 중량 → 흘수 → 저항 → 모터·배터리 → 중량 … 수렴까지.

    추진계 중량을 고정비율 개략에서 시작해 실측(모터+배터리)으로 수렴.
    반환: (weights, hydro, resist, motors, batt_kg, iteration).
    run_pipeline·최적화기·Ship-D 선별기가 공유하는 평가 코어.

    resistance_fn(mesh, draft, speed, weights) 주입 시 그 경로 사용
    (예: 메쉬형 Michell — Ship-D 임의 형상, Savitsky — 활주).
    기본은 Wigley 해석 경로. criteria로 체계별 안정성 밴드 주입 가능.
    """
    n_exp, m_exp = solve_exponents(dims.cb) if resistance_fn is None \
        else (None, None)
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


def run_pipeline(goal: GoalSpec, out_dir: str | Path,
                 loa: float | None = None,
                 payload_volume: float | None = None) -> dict:
    """payload_volume [m³]: 짐 부피 직접 입력 (생략 시 용도별 밀도 환산)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dims = estimate_dimensions(goal, loa=loa)
    volume_est = dims.cb * dims.loa * dims.beam * dims.draft_design
    regime = classify(goal.target_speed_ms, dims.loa, volume_est)
    vmax = max_displacement_speed(dims.loa)
    require_supported(regime)  # C-2로 전 체계 개방 — 향후 신규 체계 게이트

    # 체계별 경로: 배수량 = Wigley + 해석 Michell / 반배수량 = 트랜섬 +
    # 메쉬 Michell·기저항 (C-1) / 활주 = 데드라이즈 프리즘 + Savitsky (C-2)
    criteria = None  # 기본 밴드 — 활주 분기에서만 교체
    if regime is Regime.SEMI_DISPLACEMENT:
        mesh = generate_transom_hull_mesh(dims)
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
        mesh = generate_hull_mesh(dims)
        n_exp, m_exp = solve_exponents(dims.cb)
        resistance_fn = None

    weights, hydro, resist, motors, batt_kg, iteration = \
        design_spiral(mesh, dims, goal, resistance_fn=resistance_fn,
                      criteria=criteria)

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

    box = largest_box(mesh, depth=dims.depth)
    pv, pv_basis = payload_volume_for(goal.payload_kg, goal.purpose,
                                      payload_volume)
    space_ok = pv <= box.volume
    maxbox_report = {
        "length": box.length, "width": box.width, "height": box.height,
        "volume": box.volume, "payload_volume": pv,
        "volume_basis": pv_basis,
        "margin_ratio": (box.volume - pv) / pv if pv > 0 else float("inf"),
        "note": "내부 구조물(모터·배터리 자리) 미차감 — 비보수적 (스펙 §5)",
    }

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
        "max_semi_speed": max_semi_speed(dims.loa),
        "hull_family": {"SEMI_DISPLACEMENT": "transom",
                        "PLANING": "planing_deadrise"}.get(regime.name,
                                                           "wigley"),
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
        "checks_space": bool(space_ok),
        "passed": bool(hydro.passed and space_ok),
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
    mb = report["maxbox"]
    print(f"탑재 공간 (#27) : MaxBox {mb['length']:.2f}×{mb['width']:.2f}×"
          f"{mb['height']:.2f} m = {mb['volume']:.3f} m³ vs 짐 "
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
    parser.add_argument("--payload-volume", type=float, default=None,
                        help="짐 부피 [m³] 직접 입력 (생략 시 용도별 "
                             "화물 밀도 개략 가정으로 무게→부피 환산)")
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
                              payload_volume=args.payload_volume)
    except (UnsupportedRegimeError, UnknownPurposeError, CbOutOfRangeError,
            SinksError, PayloadInfeasibleError, NoSuitableMotorError,
            SpiralNotConvergedError, PlaningEquilibriumError) as e:
        print(f"[중단] {e}", file=sys.stderr)
        return 3
    _print_summary(report)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
