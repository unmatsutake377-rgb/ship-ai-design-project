"""MaxBox — 선체 내부 최대 직육면체 (#27, 스펙 2026-07-31).

"이 배에 짐 상자가 얼마나 크게 들어가나"의 척도. 무게 검사
(아르키메데스)가 놓치는 공간 검사를 메운다: 무게는 실려도 부피가
안 들어가는 배를 걸러냄.

탐색 가정 (스펙 §5 한계): 상자는 축 정렬·중심선 대칭·갑판 접촉.
아래로 좁아지는 보통 선형에서 폭 병목은 상자 바닥에서 생기지만,
가정하지 않고 [z0, 갑판] 전 높이의 최소 반폭을 쓴다 (보수적).

알고리즘: 내부 점 격자(contains) → z0 스캔 → 반폭 지형도 w(x) →
히스토그램 최대 직사각형 (고전 문제 — 손계산 정답지로 검증).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

# 용도별 화물 밀도 [kg/m³] — 개략 대표값 (명명 상수).
# 실측 수집(2026-08-02, data/payload_items.csv): 실제 조사 화물의
# 밀도는 88.5(플로트 패키지) ~ 1,843(리튬 배터리) kg/m³ — **21배 폭**.
# 단일 밀도로 짐 전체를 대표하는 것 자체가 거친 모형임이 실측으로
# 확인됨 → 프리셋은 "중간 대표값"이고, 부피가 중요한 사용자는
# --payload-volume 직접 입력이 정도(正道) (CLI 도움말에도 안내).
PAYLOAD_DENSITY = {"survey": 600.0, "patrol": 600.0, "workboat": 400.0,
                   "cargo": 500.0}  # 일반 화물 개략 (2단계, C급)


@dataclass(frozen=True)
class BoxReport:
    length: float
    width: float
    height: float
    volume: float
    x0: float
    x1: float
    z0: float


def largest_rectangle(half_widths: list[float], dx: float
                      ) -> tuple[float, float, float, float]:
    """반폭 히스토그램에서 최대 직사각형 (면적, x0, x1, 반폭).

    창 [i..j]의 면적 = 창 길이 × 창 안 최소 반폭. O(n²) 전수 —
    n~수십이라 충분."""
    n = len(half_widths)
    best = (0.0, 0.0, 0.0, 0.0)
    for i in range(n):
        w_min = float("inf")
        for j in range(i, n):
            w_min = min(w_min, half_widths[j])
            area = (j - i + 1) * dx * w_min
            if area > best[0]:
                best = (area, i * dx, (j + 1) * dx, w_min)
    return best


def largest_box(mesh: trimesh.Trimesh, depth: float,
                n_x: int = 36, n_y: int = 20, n_z: int = 12) -> BoxReport:
    """선체 내부 최대 직육면체 — 이산 탐색."""
    (xmin, ymin, zmin), (xmax, ymax, _) = mesh.bounds
    xs = np.linspace(xmin, xmax, n_x)
    ys = np.linspace(0.0, ymax, n_y)          # 중심선 대칭 — +y만 검사
    zs = np.linspace(zmin, depth, n_z, endpoint=False) + \
        (depth - zmin) / n_z * 0.5            # 셀 중심 (경계면 회피)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    grid = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    inside = mesh.contains(grid).reshape(n_x, n_y, n_z)

    # w[x, z] = y=0부터 연속으로 내부인 반폭
    consec = np.cumprod(inside, axis=1)       # 첫 바깥 이후 전부 0
    w_xz = consec.sum(axis=1) * dy            # [n_x, n_z]

    best = BoxReport(0, 0, 0, 0, 0, 0, 0)
    for iz0 in range(n_z):
        # 상자 [z0, 갑판]: 그 높이 구간의 최소 반폭이 병목
        w_eff = w_xz[:, iz0:].min(axis=1)
        area, x0, x1, w = largest_rectangle(list(w_eff), dx)
        height = depth - zs[iz0]
        volume = area * 2.0 * height          # 면적은 반폭 기준 → ×2
        if volume > best.volume:
            best = BoxReport(length=x1 - x0, width=2.0 * w, height=height,
                             volume=volume, x0=xmin + x0, x1=xmin + x1,
                             z0=float(zs[iz0]))
    return best


def payload_volume_for(payload_kg: float, purpose: str,
                       direct_volume: float | None = None) -> tuple[float, str]:
    """짐 부피 [m³] — 직접 입력 우선, 없으면 밀도 프리셋 환산."""
    if direct_volume is not None:
        return direct_volume, "직접 입력"
    density = PAYLOAD_DENSITY.get(purpose, 400.0)
    return payload_kg / density, f"밀도 가정 {density:.0f} kg/m³ (개략)"
