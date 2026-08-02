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
N_FEATURES = 32

# v3 (2026-08-03, 스펙 features-v3): 평형 정합 + 다충실도
# ① 평형 흘수를 고정 150 kg 개략 대신 라벨 파이프라인과 같은 무게
#    모델(estimate_weights 고정비율 경로)로 배마다 계산 — 흘수
#    불일치 소음 제거. 잔여 불일치: 라벨은 설계 나선(실모터 반복),
#    특징은 고정 비율 (정직 한계, 스펙 §2)
# ② 저해상 물리 2종 (r_wave_lo, r_michell_lo) — 다충실도 특징.
#    해상도 60/30/80: 척간 비율 변동계수 0.11·척당 54 ms 실측
#    (30/15는 CV 0.29로 기각, 라벨 평가 대비 ~1/20 비용)
LABEL_SPEED_MS = 1.2                  # 라벨 생성 조건과 동일
LABEL_PAYLOAD_KG = 100.0
LO_N_X, LO_N_Z, LO_N_U = 60, 30, 80

# v2 (2026-08-01 2회차): 저항·안정 라벨용 8종 추가 — 평형 흘수 근사
# 기준 배수량. 라벨 생성 조건(payload 100 kg 나선 수렴 전체 중량
# ~120-200 kg)의 중앙 개략값. 실무 근거 아님 — 명명 상수.
REF_MASS_KG = 150.0   # v2 유물 — v3는 배별 무게 모델 사용 (이력 보존용)
RHO = 1025.0
KB_FRAC = 0.53    # 상자~V형 사이 개략 (KB ≈ 0.5~0.55 T)
KG_FRAC = 0.65    # 중량 모델의 KG/D 개략 — 라벨 파이프라인과 동일 계보

FEATURE_NAMES = (
    ["beam", "depth", "vol_total", "area_total", "fill", "l_over_b",
     "b_over_d"]
    + [f"{name}_t{int(f*100)}" for f in DRAFT_FRACS
       for name in ("vol", "wet", "awp", "ixx")]
    + ["cm", "cp", "lcb_frac"]
    # v2: 평형 자세 5 + 안정 프록시 1 + 입사각 1 + 능력비 1
    + ["t_eq", "wet_eq", "awp_eq", "ixx_eq", "bm_eq",
       "gmb_proxy", "entrance_deg", "capacity_ratio"]
    # v3: 다충실도 2 (저해상 조파 / 저해상 총저항)
    + ["r_wave_lo", "r_michell_lo"]
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

    # ---- v2→v3: 평형 자세 특징 (라벨과 같은 무게 모델로 정합) ----
    try:
        from src.physics.weights import estimate_weights

        # v3 ①: 배마다 무게 모델 계산 (v2는 150 kg 고정 개략이었음)
        total_mass = estimate_weights(float(mesh.area), depth,
                                      LABEL_PAYLOAD_KG).total_mass
        v_ref = total_mass / RHO
        ts = np.array([f * depth for f in DRAFT_FRACS])
        vols = np.array(feats[7:19:4])       # vol_t30/50/70
        wets = np.array(feats[8:19:4])
        awps = np.array(feats[9:20:4])
        ixxs = np.array(feats[10:20:4])
        # V(T) 3점 선형 보간·외삽의 역함수로 t*. np.interp는 범위 밖을
        # 못박아(외삽 안 함) 평형 흘수가 0.3D 아래인 다수 선체가 전부
        # 0.3D로 뭉개짐 — 끝 구간 기울기로 직접 외삽 (바지선은 V(T)가
        # 선형이라 이 방식이 해석 정답과 정확히 일치)
        t_eq = _interp_extrap(v_ref, vols, ts)
        wet_eq = _interp_extrap(t_eq, ts, wets)
        awp_eq = _interp_extrap(t_eq, ts, awps)
        ixx_eq = _interp_extrap(t_eq, ts, ixxs)
        bm_eq = ixx_eq / v_ref
        gmb = (KB_FRAC * t_eq + bm_eq - KG_FRAC * depth) / beam
        capacity = float(vols[-1]) / v_ref   # <1이면 만재 불가 경향
        feats += [t_eq, wet_eq, awp_eq, ixx_eq, bm_eq, gmb,
                  _entrance_angle_deg(mesh, min(t_eq, 0.9 * depth)),
                  capacity]
    except Exception:
        feats += [np.nan] * 8
        t_eq = np.nan

    # ---- v3 ②: 다충실도 — 저해상 물리 저항 (스펙 §2a) ----
    try:
        from src.physics.resistance import (
            frictional_resistance,
            michell_wave_resistance_mesh,
        )

        t_lo = float(min(max(t_eq, 0.05 * depth), 0.9 * depth))
        rw_lo = michell_wave_resistance_mesh(
            mesh, t_lo, LABEL_SPEED_MS,
            n_u=LO_N_U, n_x=LO_N_X, n_z=LO_N_Z)
        wet_lo = wetted_surface(mesh, t_lo)
        rf_lo = frictional_resistance(LABEL_SPEED_MS, loa, wet_lo)
        feats += [rw_lo, rf_lo + rw_lo]
    except Exception:
        feats += [np.nan] * 2
    return np.array(feats, dtype=np.float64)


def _interp_extrap(x: float, xs, ys) -> float:
    """단조 xs 기준 선형 보간 + 양끝 기울기 외삽 (np.interp는 못박음)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if x <= xs[0]:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return float(ys[0] + slope * (x - xs[0]))
    if x >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return float(ys[-1] + slope * (x - xs[-1]))
    return float(np.interp(x, xs, ys))


def _entrance_angle_deg(mesh: trimesh.Trimesh, draft: float) -> float:
    """수선 입사 반각 [도] — 뱃머리가 물을 가르는 각도 (조파저항 주인).

    수선(흘수 높이) 반폭 곡선의 앞쪽 10% 구간 기울기 atan(Δ반폭/Δx).
    상자(뭉툭)는 90°, 날씬한 배일수록 작다."""
    from src.physics.resistance import hull_offsets

    xs, zs, y_half = hull_offsets(mesh, draft, n_x=40, n_z=8)
    wl = y_half[:, -1]                       # 수선 근처 반폭
    span = float(xs[-1] - xs[0])
    margin = 1e-4 * span                     # hull_offsets의 끝단 여백
    n_bow = max(2, len(xs) // 10)

    def end_angle(seg_x, seg_y):
        """끝점(반폭 0 가상 측점 포함) 구간의 최대 기울기 각도.

        뭉툭한 상자는 끝점→첫 측점에서 반폭이 수직 점프 → ~90°,
        매끈한 배는 접선 기울기 → 작다. 구간 최대를 쓰는 이유:
        평균(현 기울기)은 뭉툭함을 희석함."""
        px = np.concatenate([[seg_x[0] - margin], seg_x])
        py = np.concatenate([[0.0], seg_y])
        dx = np.diff(px)
        dy = np.abs(np.diff(py))
        return float(np.degrees(np.arctan2(dy, dx).max()))

    a_fwd = end_angle(xs[:n_bow], wl[:n_bow])
    a_aft = end_angle((xs[-n_bow:])[::-1] * -1, wl[-n_bow:][::-1])
    return min(a_fwd, a_aft)


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
    # v3 별도 파일 — v2(features_loa3.npy) 보존 (A/B 재현성, 스펙 §3)
    out = Path("data/shipd/features_v3_loa3.npy")
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    feats = featurize_all(out, limit=limit)
    ok = np.isfinite(feats).all(axis=1).sum()
    print(f"저장: {out} — {len(feats)}척 중 완전 특징 {ok}척 "
          f"({N_FEATURES}차원)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
