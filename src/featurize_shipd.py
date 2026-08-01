"""Ship-D 3만 척 물리 특징 사전 계산 (대리모델 특징 공학, 2026-08-01).

배경: 45개 원시 매개변수 직통 학습은 분류 0.535(동전 수준)로 실패 —
"원시 좌표 대신 물리가 아는 축을 줘라"는 처방의 실행.

특징 22개 (전부 기존 검증된 물리 함수 재사용):
- 전체 형상 7: B, D, V_total, S_total, 충전율 V/(L·B·D), L/B, B/D
- 흘수별 정역학 12: T ∈ {0.3D, 0.5D, 0.7D} 각각 (잠긴 부피, 젖은
  면적, 수선면적, 수선면 관성모멘트) — 배수량 필터·복원력의 원료
- 선형 계수 3: Cm(중앙단면), Cp(주형), LCB/L (부력중심 위치)

주의: Ship-D 파생물 — 재배포 금지 폴더 data/shipd/ 에만 저장.
계산 실패 선체는 NaN 행 (학습 시 제외) — 조용한 0 채움 금지.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

from data import shipd_loader
from src.physics.hydrostatics import (
    immersed_volume,
    kb_bm,
    waterplane_properties,
)
from src.physics.resistance import wetted_surface

TARGET_LOA = 3.0                      # 선별과 동일 상사 기준
DRAFT_FRACS = (0.3, 0.5, 0.7)         # 기준 흘수 3단 (형심 대비)
N_FEATURES = 22

FEATURE_NAMES = (
    ["beam", "depth", "vol_total", "area_total", "fill", "l_over_b",
     "b_over_d"]
    + [f"{name}_t{int(f*100)}" for f in DRAFT_FRACS
       for name in ("vol", "wet", "awp", "ixx")]
    + ["cm", "cp", "lcb_frac"]
)


def _midship_area(submerged: trimesh.Trimesh) -> float:
    """물속 부분 메쉬의 길이 중앙 단면적 — Cm의 분자.

    물속만 미리 잘라낸 메쉬를 받으므로 흘수 자르기가 불필요 —
    면적은 2D 펼침의 좌표틀 이동에 불변이라 안전.
    중앙은 (xmin+xmax)/2 — Ship-D는 x가 0~L 범위라 x=0은 뱃머리 끝
    (x=0 고정으로 잘랐다가 80% NaN 났던 실측 버그, 2026-08-01)."""
    x_mid = 0.5 * (submerged.bounds[0][0] + submerged.bounds[1][0])
    sec = submerged.section(plane_origin=[x_mid, 0, 0],
                            plane_normal=[1, 0, 0])
    if sec is None:
        return float("nan")
    planar, _ = sec.to_2D()
    return float(sum(p.area for p in planar.polygons_full))


def hull_features(mesh: trimesh.Trimesh) -> np.ndarray:
    """선체 1척 → 특징 22개. 실패 항목은 NaN."""
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    loa, beam, depth = xmax - xmin, ymax - ymin, zmax - zmin
    feats = [beam, depth, float(mesh.volume), float(mesh.area),
             float(mesh.volume) / (loa * beam * depth),
             loa / beam, beam / depth]

    t_mid = DRAFT_FRACS[1] * depth
    for frac in DRAFT_FRACS:
        t = frac * depth
        try:
            vol = immersed_volume(mesh, t)
            wet = wetted_surface(mesh, t)
            awp, ixx = waterplane_properties(mesh, t)
            feats += [vol, wet, awp, ixx]
        except Exception:
            feats += [np.nan] * 4

    try:
        sub = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal=[0, 0, -1], plane_origin=[0, 0, t_mid],
            cap=True)
        am = _midship_area(sub)
        vol_mid = immersed_volume(mesh, t_mid)
        cm = am / (beam * t_mid)
        cp = vol_mid / (am * loa) if am > 0 else np.nan
        # LCB: 물속 부분 무게중심의 길이방향 위치 (선체 길이 비율)
        lcb = (sub.center_mass[0] - xmin) / loa
        feats += [cm, cp, lcb]
    except Exception:
        feats += [np.nan] * 3
    return np.array(feats, dtype=np.float64)


def featurize_all(out_path: Path, limit: int | None = None,
                  report_every: int = 1000) -> np.ndarray:
    vectors, _ = shipd_loader.load_vectors()
    n = len(vectors) if limit is None else min(limit, len(vectors))
    feats = np.full((n, N_FEATURES), np.nan)
    for i in range(n):
        try:
            feats[i] = hull_features(
                shipd_loader.scaled_mesh(vectors[i], TARGET_LOA))
        except Exception:
            pass                       # NaN 행 유지 — 학습 시 제외
        if (i + 1) % report_every == 0:
            ok = np.isfinite(feats[: i + 1]).all(axis=1).sum()
            print(f"  {i + 1}/{n} — 완전 특징 {ok}척", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feats)
    return feats


def main() -> int:
    out = Path("data/shipd/features_loa3.npy")
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    feats = featurize_all(out, limit=limit)
    ok = np.isfinite(feats).all(axis=1).sum()
    print(f"저장: {out} — {len(feats)}척 중 완전 특징 {ok}척 "
          f"({N_FEATURES}차원)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
