"""정역학: 침수부 계산, KB/BM, 평형 흘수, 복원성 필터 (spec §2.3).

필터 판정은 밴드 기반:
- 배수량 일치는 equilibrium_draft가 보장 (흘수를 중량에서 역산하므로),
  대신 침수 여부(SinksError)와 건현·GM 밴드를 검사한다.
- GM > 0 만으로는 부족 — GM/B 밴드로 판정 (너무 작으면 위험, 너무 크면 급횡동요).
모든 중간값을 리포트에 기록해 가정이 보이게 한다 ("그럴듯한 오답" 방지).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import math

import numpy as np
import trimesh

RHO_SEAWATER = 1025.0  # [kg/m^3]


class SinksError(ValueError):
    """갑판까지 잠겨도 부력이 중량에 못 미침."""


def immersed_mesh(mesh: trimesh.Trimesh, draft: float) -> trimesh.Trimesh:
    """흘수 아래 부분을 잘라 watertight 메쉬로 반환."""
    return trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal=[0, 0, -1], plane_origin=[0, 0, draft], cap=True
    )


def immersed_volume(mesh: trimesh.Trimesh, draft: float) -> float:
    below = immersed_mesh(mesh, draft)
    return 0.0 if below is None else float(below.volume)


def _waterplane_polygons(mesh: trimesh.Trimesh, draft: float) -> list[np.ndarray]:
    """수선면 단면 폐곡선들의 (x, y) 좌표 배열 목록."""
    section = mesh.section(plane_origin=[0, 0, draft], plane_normal=[0, 0, 1])
    if section is None:
        return []
    return [np.asarray(loop)[:, :2] for loop in section.discrete]


def _polygon_area_ixx(coords: np.ndarray) -> tuple[float, float]:
    """다각형 면적과 x축(y=0, 종축) 기준 2차 모멘트 ∫y²dA.

    Green 정리 기반 공식. 선체는 좌우대칭이라 centerline이 도심을 지나므로
    y=0 축 기준값이 곧 횡메타센터용 I_T.
    """
    x, y = coords[:, 0], coords[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y2 - x2 * y
    area = 0.5 * np.sum(cross)
    ixx = np.sum(cross * (y * y + y * y2 + y2 * y2)) / 12.0
    return abs(float(area)), abs(float(ixx))


def waterplane_properties(mesh: trimesh.Trimesh, draft: float) -> tuple[float, float]:
    """수선면 면적 Aw와 종축 기준 횡 2차 모멘트 I_T."""
    area_total, ixx_total = 0.0, 0.0
    for coords in _waterplane_polygons(mesh, draft):
        a, ixx = _polygon_area_ixx(coords)
        area_total += a
        ixx_total += ixx
    return area_total, ixx_total


def kb_bm(mesh: trimesh.Trimesh, draft: float) -> tuple[float, float]:
    below = immersed_mesh(mesh, draft)
    vol = float(below.volume)
    kb = float(below.center_mass[2])  # 부심 높이 (킬 기준)
    _, ixx = waterplane_properties(mesh, draft)
    bm = ixx / vol
    return kb, bm


def equilibrium_draft(mesh: trimesh.Trimesh, mass_kg: float,
                      rho: float = RHO_SEAWATER) -> float:
    """중량 = 부력이 되는 흘수를 이분법으로 역산 (spec §2.3)."""
    target = mass_kg / rho
    z_max = float(mesh.bounds[1][2])
    if immersed_volume(mesh, z_max) < target:
        raise SinksError(
            f"중량 {mass_kg:.1f} kg에 필요한 배수용적 {target:.3f} m³가 "
            f"갑판까지 잠긴 부피보다 큽니다 — 설계 침수."
        )
    lo, hi = 0.0, z_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if immersed_volume(mesh, mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-5:
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class StabilityCriteria:
    disp_tolerance: float = 0.02                      # |Δ−W|/W 허용오차
    gm_over_beam: tuple[float, float] = (0.04, 0.40)  # GM/B 밴드 (소형정 개략)
    freeboard_over_depth_min: float = 0.10            # 최소 건현/형심


@dataclass
class HydrostaticReport:
    draft: float
    freeboard: float
    displacement_volume: float
    displacement_mass: float
    kb: float
    bm: float
    kg: float
    gm: float
    waterplane_area: float
    waterplane_ixx: float
    checks: dict = field(default_factory=dict)
    passed: bool = False


def evaluate(mesh: trimesh.Trimesh, total_mass: float, kg: float,
             beam: float, depth: float, rho: float = RHO_SEAWATER,
             criteria: StabilityCriteria | None = None) -> HydrostaticReport:
    """평형 흘수 역산 → KB/BM/GM → 밴드 필터 판정.

    불합격은 예외가 아니라 정상 결과 — 리포트로 사유를 남긴다 (spec §3).
    """
    crit = criteria or StabilityCriteria()
    draft = equilibrium_draft(mesh, total_mass, rho)
    vol = immersed_volume(mesh, draft)
    kb, bm = kb_bm(mesh, draft)
    aw, ixx = waterplane_properties(mesh, draft)
    gm = kb + bm - kg
    freeboard = depth - draft

    disp_ok = abs(vol * rho - total_mass) / total_mass <= crit.disp_tolerance
    gm_ok = crit.gm_over_beam[0] <= gm / beam <= crit.gm_over_beam[1]
    fb_ok = freeboard / depth >= crit.freeboard_over_depth_min

    # bool(): numpy bool_이 섞이면 JSON 직렬화가 깨진다
    checks = {"displacement": bool(disp_ok), "gm_band": bool(gm_ok),
              "freeboard": bool(fb_ok)}
    return HydrostaticReport(
        draft=draft, freeboard=freeboard,
        displacement_volume=vol, displacement_mass=vol * rho,
        kb=kb, bm=bm, kg=kg, gm=gm,
        waterplane_area=aw, waterplane_ixx=ixx,
        checks=checks, passed=all(checks.values()),
    )


# ── 세로 방향 (3단계 트림, 스펙 multibay-maxbox §3) ────────────────

def _polygon_area_iyy_cx(coords: np.ndarray) -> tuple[float, float, float]:
    """다각형 면적, 도심 x, 도심 기준 세로 2차 모멘트 ∫(x−x̄)²dA."""
    x, y = coords[:, 0], coords[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y2 - x2 * y
    area = 0.5 * np.sum(cross)
    if abs(area) < 1e-12:
        return 0.0, 0.0, 0.0
    cx = np.sum(cross * (x + x2)) / (6.0 * area)
    iyy0 = np.sum(cross * (x * x + x * x2 + x2 * x2)) / 12.0
    iyy = iyy0 - area * cx * cx          # 평행축 정리 (원점 → 도심)
    return abs(float(area)), float(cx), abs(float(iyy))


def longitudinal_properties(mesh: trimesh.Trimesh, draft: float
                            ) -> tuple[float, float]:
    """(LCB x, BML) — 세로 부심 위치와 세로 메타센터 반경.

    BML = I_L/∇ (수선면 도심 기준 세로 2차 모멘트 / 배수용적).
    통상 BML ≫ KG라 세로 복원은 수선면 형상이 지배."""
    below = immersed_mesh(mesh, draft)
    vol = float(below.volume)
    lcb = float(below.center_mass[0])
    area_t, cx_num, iyy_t = 0.0, 0.0, 0.0
    polys = [(a, cx, iyy) for a, cx, iyy in
             (_polygon_area_iyy_cx(c) for c in _waterplane_polygons(mesh, draft))]
    area_t = sum(p[0] for p in polys)
    if area_t > 0.0:
        cx_all = sum(p[0] * p[1] for p in polys) / area_t
        iyy_t = sum(p[2] + p[0] * (p[1] - cx_all) ** 2 for p in polys)
    return lcb, iyy_t / max(vol, 1e-12)


def trim_angle_deg(lcg_x: float, lcb_x: float, kb: float, bml: float,
                   kg: float) -> float:
    """소각 선형 트림각 [deg]: θ = (LCG−LCB)/GML, GML = KB+BML−KG.

    자유 자세 완전판(기울인 물면 2변수 평형)의 소각 근사 — 소형정
    운용 트림(수 도 이내) 영역에서 표준 설계법. 부호: +x 쪽 무거우면 +."""
    gml = kb + bml - kg
    if gml <= 1e-9:
        return float("inf")
    return math.degrees((lcg_x - lcb_x) / gml)
