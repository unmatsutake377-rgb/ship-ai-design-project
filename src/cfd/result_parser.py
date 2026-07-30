"""OpenFOAM forces 함수 로그 파서 (스펙 §2, §4).

force.dat 형식 (v2406): 각 행 = 시각 + 벡터 3개 (전체/압력/점성 힘).
괄호를 벗기고 숫자만 뽑으면 [t, tx,ty,tz, px,py,pz, vx,vy,vz] 10개.
항력 = x성분 (유동이 +x 방향이므로).

수렴 판정: 마지막 window_frac 구간의 변동계수(CoV = 표준편차/|평균|)가
cov_tol 미만이면 수렴 — 정상상태 해가 "한 값에 눌러앉았다"는 뜻.
반쪽 도메인(y≥0)이라 힘에 symmetry_factor(기본 2)를 곱해 전선체 값으로.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_NUM = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class CfdResult:
    drag_total_n: float
    drag_pressure_n: float
    drag_viscous_n: float
    converged: bool
    n_samples: int


def find_force_file(case_dir: Path) -> Path:
    hits = sorted(Path(case_dir).glob("postProcessing/forces/*/force.dat"))
    if not hits:
        raise FileNotFoundError(f"force.dat 없음: {case_dir}")
    return hits[-1]


def parse_forces(case_dir: Path, window_frac: float = 0.2,
                 cov_tol: float = 0.02,
                 symmetry_factor: float = 2.0) -> CfdResult:
    rows = []
    for line in find_force_file(case_dir).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        nums = [float(x) for x in _NUM.findall(line)]
        if len(nums) >= 10:
            rows.append(nums[:10])
    if not rows:
        raise ValueError("force.dat에 데이터 행이 없음")
    arr = np.array(rows)
    n_win = max(2, int(len(arr) * window_frac))
    win = arr[-n_win:]
    total_x, press_x, visc_x = win[:, 1], win[:, 4], win[:, 7]
    mean = float(total_x.mean())
    cov = float(total_x.std() / abs(mean)) if mean != 0 else float("inf")
    return CfdResult(
        drag_total_n=symmetry_factor * mean,
        drag_pressure_n=symmetry_factor * float(press_x.mean()),
        drag_viscous_n=symmetry_factor * float(visc_x.mean()),
        converged=cov < cov_tol,
        n_samples=len(arr),
    )
