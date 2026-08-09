"""정수 종강도 하중 곡선 (구조 강도 1단계, 스펙 2026-08-09 §2).

배 = 보(beam). 중량 w(x)와 부력 b(x)의 길이 방향 어긋남이 전단력
V(x)·굽힘 모멘트 M(x)를 만든다.

부호 관례 (프로젝트 공통): q = w − b, V = ∫q dx, M = ∫V dx,
**M > 0 = 호깅** (IACS hog 양수 정합). 중앙 화물 몰림 → 새깅(음수).

중량 분포 = 성분별 균일 블록 (C급 개략 — 정밀 분포는 백로그):
구조·의장 = 전장 균일, 기관·연료 = 선미 10~30% 구간,
화물(payload) = 중앙 25~85% 구간.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RHO_SEAWATER = 1025.0
G_ACC = 9.81

# 성분별 (선미 기준 시작 분율, 끝 분율) — 상선 통상 배치 (C급)
_BLOCK_FRACS = {
    "structure": (0.0, 1.0),
    "outfit": (0.0, 1.0),
    "machinery": (0.10, 0.30),
    "fuel": (0.10, 0.30),
    "payload": (0.25, 0.85),
}

WeightBlock = tuple[float, float, float]      # (mass_kg, x0, x1)


def standard_weight_blocks(component_masses_kg: dict[str, float],
                           xmin: float, loa: float) -> list[WeightBlock]:
    """성분 질량 → 통상 배치 균일 블록 목록. 미등록 성분 = 전장 균일."""
    out = []
    for name, mass in component_masses_kg.items():
        if mass <= 0.0:
            continue
        f0, f1 = _BLOCK_FRACS.get(name, (0.0, 1.0))
        out.append((float(mass), xmin + f0 * loa, xmin + f1 * loa))
    return out


def weight_linear_density(xs: np.ndarray,
                          blocks: list[WeightBlock]) -> np.ndarray:
    """블록 합성 w(x) [N/m] — 격자 적분이 총중량과 정확히 폐합하게
    정규화 (격자-블록 경계 불일치 오차 제거)."""
    w = np.zeros_like(xs, dtype=float)
    for mass, x0, x1 in blocks:
        span = max(x1 - x0, 1e-9)
        w += np.where((xs >= x0 - 1e-12) & (xs <= x1 + 1e-12),
                      mass * G_ACC / span, 0.0)
    total = sum(m for m, _, _ in blocks) * G_ACC
    integ = float(np.trapezoid(w, xs))
    if integ > 0.0:
        w *= total / integ
    return w


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """누적 사다리꼴 적분 — V·M 조립 공용."""
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate([[0.0], np.cumsum(seg)])


def station_area(mesh, x: float, waterline_z: float,
                 nz: int = 60) -> float:
    """스테이션 x, 수선 z 아래 몰수 단면적 [m²].

    수평 스캔라인 — **짝수 교차 규칙 일반화** (2026-08-10 쌍동
    2단계): 각 z에서 윤곽 변과의 교차 y들을 정렬해 (y₁,y₂),(y₃,y₄)…
    구간 폭 합산. 단동 대칭 선체는 기존 max-y 방식과 동일 결과,
    쌍동(폐곡선 2개)은 두 몸통 폭이 정확히 합산됨 (구 방식은 한쪽
    데미헐만 읽어 부력 절반 오독 — 시험이 검거). 성긴 폴리곤·수평
    변에 강건, Lewis 제약 없음."""
    sec = mesh.section(plane_origin=[float(x), 0, 0],
                       plane_normal=[1, 0, 0])
    if sec is None or not len(sec.entities):
        return 0.0
    edges = []
    z_keel = np.inf
    for e in sec.entities:
        d = e.discrete(sec.vertices)          # (N,3) 폴리라인
        z_keel = min(z_keel, float(d[:, 2].min()))
        for p0, p1 in zip(d[:-1], d[1:]):
            edges.append((p0[1], p0[2], p1[1], p1[2]))   # (y0,z0,y1,z1)
    if not edges or waterline_z - z_keel < 1e-6:
        return 0.0
    zs = np.linspace(z_keel + 1e-6, waterline_z - 1e-9, nz)
    widths = np.zeros(nz)
    for i, z in enumerate(zs):
        ys = []
        for y0, z0, y1, z1 in edges:
            zlo, zhi = min(z0, z1), max(z0, z1)
            if zhi - zlo < 1e-12 or not (zlo <= z <= zhi):
                continue                       # 수평 변·비교차
            t = (z - z0) / (z1 - z0)
            ys.append(y0 + t * (y1 - y0))
        if len(ys) < 2:
            continue
        ys.sort()
        n_pairs = len(ys) // 2
        widths[i] = sum(ys[2 * k + 1] - ys[2 * k]
                        for k in range(n_pairs))
    return float(np.trapezoid(widths, zs))


@dataclass(frozen=True)
class LoadCurves:
    xs: np.ndarray            # 스테이션 [m] (선미→선수)
    weight_npm: np.ndarray    # w(x) [N/m]
    buoy_npm: np.ndarray      # b(x) [N/m]
    shear_n: np.ndarray       # V(x) [N]
    moment_nm: np.ndarray     # M(x) [N·m] — M>0 호깅
    buoy_scale: float         # 부력 폐합 배율 (1 근방 = 절단 건강)
    shear_residual_n: float   # 보정 전 V(L) 잔차 (정직 기록)
    moment_residual_nm: float


def still_water_curves(mesh, draft: float, blocks,
                       n: int = 101) -> LoadCurves:
    """정수 전단력·굽힘 모멘트 곡선.

    draft = 메쉬 좌표계 수선 z. 부력은 총중량으로 폐합 정규화
    (배율 기록 — 1에서 멀면 절단·평형 이상 신호). 양끝 잔차는
    선형 보정 후 원값 기록 (통상 관행·정직 표기)."""
    (xmin, _, _), (xmax, _, _) = mesh.bounds
    xs = np.linspace(xmin, xmax, n)
    w = weight_linear_density(xs, blocks)
    total_w = sum(m for m, _, _ in blocks) * G_ACC

    b = np.array([station_area(mesh, x, draft) for x in xs])
    b *= RHO_SEAWATER * G_ACC
    integ_b = float(np.trapezoid(b, xs))
    scale = total_w / integ_b if integ_b > 0 else 1.0
    b = b * scale

    q = w - b
    shear = _cumtrapz(q, xs)
    moment = _cumtrapz(shear, xs)
    v_res, m_res = float(shear[-1]), float(moment[-1])
    ramp = (xs - xs[0]) / (xs[-1] - xs[0])
    shear = shear - v_res * ramp
    moment = moment - m_res * ramp
    return LoadCurves(xs=xs, weight_npm=w, buoy_npm=b, shear_n=shear,
                      moment_nm=moment, buoy_scale=scale,
                      shear_residual_n=v_res, moment_residual_nm=m_res)
