"""선체 메쉬 → 스테이션별 Lewis 단면 추출 (내항성 1단계 2차).

기존 단면 절단 인프라(holtrop_input_from_mesh 계보)를 재사용해
흘수선 아래 각 스테이션의 (B_wl, T, σ)를 실측 → fit_lewis.

검증 앵커: Wigley 수식 선체는 단면 성질이 해석적으로 알려짐 —
중앙 단면 σ = Cm (수직 지수 m의 성질 m/(m+1) = Cm).
"""
from __future__ import annotations

import numpy as np

from src.physics.seakeeping.lewis import (
    LewisRangeError,
    LewisSection,
    fit_lewis,
)


def station_geometry(mesh, x: float, draft: float,
                     nz: int = 60) -> tuple[float, float, float] | None:
    """스테이션 x에서 흘수선 아래 (B_wl, T_local, σ) 실측.

    반환 None = 절단 실패/퇴화 단면 (선수미 끝)."""
    sec = mesh.section(plane_origin=[float(x), 0, 0],
                       plane_normal=[1, 0, 0])
    if sec is None or not len(sec.entities):
        return None
    pts = np.vstack([e.discrete(sec.vertices) for e in sec.entities])
    below = pts[pts[:, 2] <= draft + 1e-9]
    pos = below[below[:, 1] > 1e-12]
    if len(pos) < 4:
        return None
    order = np.argsort(pos[:, 2])
    zs_raw = pos[order][:, 2]
    ys_raw = pos[order][:, 1]
    # 킬은 y=0 수렴점 — y>0 필터 전 전체 점에서 판독 (얕은 흘수 사고 방지)
    z_keel = float(below[:, 2].min())
    t_local = draft - z_keel
    if t_local < 1e-6:
        return None
    zs = np.linspace(z_keel + 1e-4 * t_local, draft - 1e-4 * t_local, nz)
    halves = np.interp(zs, zs_raw, ys_raw)
    b_wl = 2.0 * float(halves[-1])          # 수선 폭
    b_max = 2.0 * float(halves.max())
    if b_wl < 1e-6 or b_wl < 0.5 * b_max:
        return None      # 수선이 좁아지는 벌브형 단면 — Lewis 밖
    area = 2.0 * float(np.trapezoid(halves, zs))
    sigma = area / (b_wl * t_local)
    return b_wl, t_local, min(sigma, 1.0)


def extract_stations(mesh, draft: float, n_stations: int = 21,
                     margin: float = 0.02) -> list[tuple[float,
                                                         LewisSection]]:
    """선체 전장을 스테이션 분할 → (x, LewisSection) 목록.

    Lewis 범위 밖/퇴화 스테이션은 건너뜀 (선수미 끝 통상) —
    호출측은 유효 스테이션만으로 스트립 적분."""
    (xmin, _, _), (xmax, _, _) = mesh.bounds
    span = xmax - xmin
    out = []
    for i in range(n_stations):
        x = xmin + span * (margin + (1.0 - 2.0 * margin)
                           * i / (n_stations - 1))
        g = station_geometry(mesh, x, draft)
        if g is None:
            continue
        try:
            out.append((x, fit_lewis(*g)))
        except LewisRangeError:
            continue
    return out
