"""능동 학습 표본 (스펙 §3) — Fn 고정, L/B만 스팬.

Michell 오차는 L/B와 Fn 둘 다에 의존 — 한 번에 한 변수만 움직인다
(변인 통제). Fn=0.341은 기존 데모 라벨(L/B=2.0)과 같은 값이라 4점이
한 곡선 위에 놓인다.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from src.ai.hull_generator import generate_hull_mesh, solve_exponents
from src.core.types import MainDimensions
from src.physics.resistance import total_resistance

FN_TARGET = 0.341
CB_STD = 0.444          # 표준 Wigley (n=m=2)
BT_STD = 1.6            # 표준 B/T

SAMPLES = [
    {"name": "wigley_lb4", "loa": 3.0, "beam": 0.75},
    {"name": "wigley_lb7", "loa": 3.0, "beam": 3.0 / 7.0},
    {"name": "wigley_lb10", "loa": 3.0, "beam": 0.30},
]


def speed_for(loa: float) -> float:
    """Fn=0.341 고정 속도 [m/s]."""
    return FN_TARGET * math.sqrt(9.81 * loa)


def build_sample(sample: dict, out_root: Path) -> Path:
    """표본 1척의 훅 입력 폴더(hull.stl + report.json) 생성."""
    draft = sample["beam"] / BT_STD
    dims = MainDimensions(loa=sample["loa"], beam=sample["beam"],
                          depth=2.0 * draft, draft_design=draft, cb=CB_STD)
    n, m = solve_exponents(CB_STD)
    mesh = generate_hull_mesh(dims)
    speed = speed_for(dims.loa)
    resist = total_resistance(mesh, dims, n, m, draft=draft, speed=speed)

    out = Path(out_root) / sample["name"]
    out.mkdir(parents=True, exist_ok=True)
    mesh.export(out / "hull.stl")
    report = {
        "goal": {"target_speed_ms": speed},
        "dimensions": dataclasses.asdict(dims),
        "hydrostatics": {"draft": draft},
        "mesh_file": "hull.stl",
        "resistance": dataclasses.asdict(resist),
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    return out


def main() -> int:
    out_root = Path("outputs/al_samples")
    for s in SAMPLES:
        d = build_sample(s, out_root)
        print(f"{s['name']}: {d}")
        print(f"  python -m src.cfd.hook --report {d} --mode simple")
        print(f"  python -m src.cfd.hook --report {d} --mode inter")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
