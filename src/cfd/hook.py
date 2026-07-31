"""CFD 훅 CLI (원안 Step 5) — 파이프라인 산출물 ↔ OpenFOAM 연결 고리.

사용 순서 (수동 2단계 — 실행이 수십 분이라 자동 연결하지 않음):
  1) 케이스 생성:  python -m src.cfd.hook --report outputs/demo --mode simple
  2) Docker 실행:  cfd/docker/run_case.sh outputs/cfd_cases/<이름> simpleFoam
  3) 라벨 수확:    python -m src.cfd.hook --report outputs/demo --mode simple --parse-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.cfd.case_builder import build_case
from src.cfd.labels import append_label
from src.cfd.result_parser import parse_forces

SOLVER = {"simple": "simpleFoam", "inter": "interFoam"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CFD 훅 (원안 Step 5)")
    parser.add_argument("--report", required=True,
                        help="파이프라인 산출물 폴더 (hull.stl + report.json)")
    parser.add_argument("--mode", choices=("simple", "inter"),
                        default="simple")
    parser.add_argument("--out", default=None,
                        help="케이스 폴더 (기본 outputs/cfd_cases/<자동이름>)")
    parser.add_argument("--parse-only", action="store_true",
                        help="실행 완료된 케이스에서 결과 파싱 + 라벨 병합")
    parser.add_argument("--labels", default="data/cfd_labels.csv")
    parser.add_argument("--grid-factor", type=float, default=1.0,
                        help="격자 배율 (수렴 연구용) — 1.5면 칸 수 ~3.4배")
    args = parser.parse_args(argv)

    report_dir = Path(args.report)
    report = json.loads((report_dir / "report.json").read_text())
    speed = report["goal"]["target_speed_ms"]
    draft = report["hydrostatics"]["draft"]
    name = f"{report_dir.name}_{args.mode}_{speed}ms"
    if args.grid_factor != 1.0:
        name += f"_g{args.grid_factor}"  # 격자별 라벨 분리 (수렴 비교용)
    case = Path(args.out) if args.out else Path("outputs/cfd_cases") / name

    if args.parse_only:
        result = parse_forces(case)
        dims = report["dimensions"]
        append_label(Path(args.labels), name, speed, draft, result,
                     report["resistance"],
                     extra={"loa_m": dims["loa"], "beam_m": dims["beam"]})
        emp = report["resistance"]
        print(f"CFD 전저항  : {result.drag_total_n:.1f} N "
              f"(압력 {result.drag_pressure_n:.1f} + "
              f"점성 {result.drag_viscous_n:.1f})")
        print(f"경험식      : {emp['total']:.1f} N "
              f"(조파 {emp['rw']:.1f} + 마찰 {emp['rf']:.1f})")
        print(f"수렴        : {'예' if result.converged else '아니오 ⚠'} "
              f"({result.n_samples} 표본)")
        print(f"라벨 저장   : {args.labels}")
        return 0 if result.converged else 2

    build_case(report_dir, case, args.mode, grid_factor=args.grid_factor)
    print(f"케이스 생성 완료: {case}")
    print("다음 단계 (Docker 실행 — 수십 분):")
    print(f"  cfd/docker/run_case.sh {case} {SOLVER[args.mode]}")
    print("끝나면 라벨 수확:")
    print(f"  python -m src.cfd.hook --report {report_dir} "
          f"--mode {args.mode} --out {case} --parse-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
