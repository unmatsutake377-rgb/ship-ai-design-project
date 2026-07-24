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

from src.ai.dimension_estimator import UnknownPurposeError, estimate_dimensions
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
    require_supported,
)
from src.core.types import GoalSpec
from src.physics.hydrostatics import RHO_SEAWATER, SinksError, evaluate
from src.physics.resistance import total_resistance
from src.physics.weights import estimate_weights


def run_pipeline(goal: GoalSpec, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dims = estimate_dimensions(goal)
    volume_est = dims.cb * dims.loa * dims.beam * dims.draft_design
    regime = classify(goal.target_speed_ms, dims.loa, volume_est)
    require_supported(regime)

    mesh = generate_hull_mesh(dims)
    weights = estimate_weights(float(mesh.area), dims.depth, goal.payload_kg)
    hydro = evaluate(mesh, weights.total_mass, weights.kg,
                     beam=dims.beam, depth=dims.depth)

    n_exp, m_exp = solve_exponents(dims.cb)
    resist = total_resistance(mesh, dims, n_exp, m_exp,
                              draft=hydro.draft, speed=goal.target_speed_ms)

    mesh_file = "hull.stl"
    mesh.export(out / mesh_file)

    report = {
        "goal": dataclasses.asdict(goal),
        "dimensions": dataclasses.asdict(dims),
        "regime": regime.name,
        "froude_length": froude_length(goal.target_speed_ms, dims.loa),
        "froude_volumetric": froude_volumetric(goal.target_speed_ms, volume_est),
        "weights": dataclasses.asdict(weights),
        "hydrostatics": dataclasses.asdict(hydro),
        "resistance": dataclasses.asdict(resist),
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
    args = parser.parse_args(argv)

    goal = GoalSpec(target_speed_ms=args.speed, payload_kg=args.payload,
                    purpose=args.purpose)
    try:
        report = run_pipeline(goal, args.out)
    except (UnsupportedRegimeError, UnknownPurposeError,
            CbOutOfRangeError, SinksError) as e:
        print(f"[중단] {e}", file=sys.stderr)
        return 3
    _print_summary(report)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
