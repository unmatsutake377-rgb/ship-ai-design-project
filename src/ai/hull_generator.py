"""일반화 Wigley 선형 생성기 (spec §2.2, M3 1단 로켓).

반폭: y(x,z) = (B/2)·(1−|2x/L|ⁿ)·(1−((T−z)/T)ᵐ)  (z ≤ T)
       z > T 구간은 수직 현측 (wall-sided), z=D에서 갑판 마감.
계수 관계: Cp = n/(n+1), Cm = m/(m+1), Cb = Cp·Cm.

한계 (spec §2.2에 명시): 선수미가 뾰족함(트랜섬 없음) —
저속 배수량 영역에서만 대표성 있음. 반배수량 지원 전에
트랜섬 선미 계열 필요 (2차 사이클).
"""
from __future__ import annotations

import numpy as np
import trimesh

from src.core.types import MainDimensions

DEFAULT_CM = 0.78          # 소형 배수량정 대표 중앙단면계수 (가정, 리포트에 기록)
CP_RANGE = (0.35, 0.85)    # 지수 역산이 건전한 프리즘계수 범위
CB_ENVELOPE = (CP_RANGE[0] * DEFAULT_CM, CP_RANGE[1] * DEFAULT_CM)  # (0.273, 0.663)


class CbOutOfRangeError(ValueError):
    """생성기 도달범위 밖의 Cb 요구 (spec §2.1 — 조용히 다른 형상 금지)."""


def solve_exponents(cb: float, cm: float = DEFAULT_CM) -> tuple[float, float]:
    """목표 Cb에서 Wigley 지수 (n, m)을 역산한다."""
    cp = cb / cm
    if not CP_RANGE[0] <= cp <= CP_RANGE[1]:
        raise CbOutOfRangeError(
            f"Cb={cb:.3f} (Cp={cp:.3f})는 생성기 범위 밖입니다. "
            f"도달 가능한 Cb: [{CB_ENVELOPE[0]:.3f}, {CB_ENVELOPE[1]:.3f}] (Cm={cm})"
        )
    m = cm / (1.0 - cm)
    n = cp / (1.0 - cp)
    return n, m


def _half_breadth(x: float, z: float, dims: MainDimensions,
                  n: float, m: float) -> float:
    u = abs(2.0 * x / dims.loa)
    longitudinal = max(0.0, 1.0 - u ** n)
    if z >= dims.draft_design:
        vertical = 1.0  # 수직 현측
    else:
        vertical = 1.0 - ((dims.draft_design - z) / dims.draft_design) ** m
    return 0.5 * dims.beam * longitudinal * max(0.0, vertical)


def generate_hull_mesh(dims: MainDimensions, n_stations: int = 61,
                       n_below: int = 15, n_above: int = 9) -> trimesh.Trimesh:
    """watertight 선체 메쉬 생성. 좌표계: x 선수미(+선수), y 좌우, z 상방(0=킬)."""
    n, m = solve_exponents(dims.cb)
    return _build_mesh(lambda x, z: _half_breadth(x, z, dims, n, m),
                       dims, n_stations, n_below, n_above, cap_stern=False)


def _build_mesh(half_breadth_fn, dims: MainDimensions, n_stations: int,
                n_below: int, n_above: int,
                cap_stern: bool = False) -> trimesh.Trimesh:
    """반폭 함수 → watertight 메쉬 (Wigley·트랜섬 공용 빌더).

    cap_stern: 선미 폭이 0이 아닌 형상(트랜섬)은 선미면 캡 필요.
    """
    xs = np.linspace(-dims.loa / 2, dims.loa / 2, n_stations)
    zs = np.concatenate([
        np.linspace(0.0, dims.draft_design, n_below),
        np.linspace(dims.draft_design, dims.depth, n_above)[1:],
    ])
    nz = len(zs)

    # 정점: 우현 그리드 + 좌현 그리드
    verts = []
    for x in xs:
        for z in zs:
            verts.append((x, half_breadth_fn(x, z), z))
    for x in xs:
        for z in zs:
            verts.append((x, -half_breadth_fn(x, z), z))
    # + 0.0: 좌현의 -0.0을 +0.0으로 정규화 — 아니면 y=0 정점(킬·선수미)이
    # 우현 +0.0과 해시가 달라 병합되지 않아 watertight가 깨진다
    verts = np.array(verts) + 0.0

    def sid(i: int, j: int) -> int:
        return i * nz + j

    def pid(i: int, j: int) -> int:
        return n_stations * nz + i * nz + j

    faces = []
    for i in range(n_stations - 1):
        for j in range(nz - 1):
            a, b, c, d = sid(i, j), sid(i + 1, j), sid(i + 1, j + 1), sid(i, j + 1)
            faces += [[a, b, c], [a, c, d]]
            a, b, c, d = pid(i, j), pid(i + 1, j), pid(i + 1, j + 1), pid(i, j + 1)
            faces += [[a, c, b], [a, d, c]]  # 좌현은 감기 방향 반대
    top = nz - 1
    for i in range(n_stations - 1):  # 갑판 마감
        a, b = sid(i, top), sid(i + 1, top)
        c, d = pid(i + 1, top), pid(i, top)
        faces += [[a, b, c], [a, c, d]]
    if cap_stern:  # 트랜섬면 마감 (선미 스테이션 i=0, 폭>0)
        for j in range(nz - 1):
            a, b = sid(0, j), sid(0, j + 1)
            c, d = pid(0, j + 1), pid(0, j)
            faces += [[a, c, b], [a, d, c]]

    # process=True: 중복 정점 병합(킬/선수미의 y=0 접합), 퇴화 면 제거
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)

    # 선수·선미 스테이션은 폭 0으로 퇴화 — 그 모서리에서 좌·우현이
    # 같은 삼각형을 반대 감김으로 한 번씩 만든다 (중심면 슬리버 쌍).
    # 두 장 모두 제거해야 manifold가 된다 (한 장만 남기면 edge가 3면 공유).
    sorted_faces = np.sort(mesh.faces, axis=1)
    # 인덱스 반복 면(퇴화) 제거
    nondegenerate = (
        (sorted_faces[:, 0] != sorted_faces[:, 1])
        & (sorted_faces[:, 1] != sorted_faces[:, 2])
    )
    # 동일 정점 삼중쌍이 2번 이상 나오면 전부 제거
    _, inverse, counts = np.unique(
        sorted_faces, axis=0, return_inverse=True, return_counts=True
    )
    unique_only = counts[inverse] == 1
    mesh.update_faces(nondegenerate & unique_only)
    mesh.remove_unreferenced_vertices()

    trimesh.repair.fix_normals(mesh)
    if mesh.volume < 0:  # 법선이 안쪽을 향하면 뒤집는다
        mesh.invert()
    return mesh


# ---------- 트랜섬 선미 계열 (Phase C-1, spec v2 §2.2 예고) ----------
# 반배수량 USV의 실제 형상: 뾰족 선수 + 평행부 + 잘린 선미(트랜섬).
# 세로 형상 F(x): 선수부(길이 BOW_FRACTION·L)는 1−ξⁿ, 선미부는
# 1−(1−β_t)·vᵖ (v=선미 방향 정규화, x=−L/2에서 F=β_t — 트랜섬 폭비).
# 단면 H(z)는 Wigley와 동일. Cm은 반배수량 V형 단면을 반영해 낮게.

TRANSOM_BOW_FRACTION = 0.45  # 선수부 길이/전장
TRANSOM_TAPER_P = 2.0        # 선미 테이퍼 지수
TRANSOM_BETA = 0.55          # 트랜섬 폭 / 최대폭
TRANSOM_CM = 0.65            # 반배수량 V형 단면 개략 중앙단면계수


def _transom_cp_bounds(beta_t: float = TRANSOM_BETA,
                       p: float = TRANSOM_TAPER_P,
                       bow_fraction: float = TRANSOM_BOW_FRACTION
                       ) -> tuple[float, float]:
    k_aft = 1.0 - (1.0 - beta_t) / (p + 1.0)
    l_aft = 1.0 - bow_fraction
    # 선수 지수 n의 건전 범위 (CP_RANGE와 동일 사상)
    return (bow_fraction * CP_RANGE[0] + l_aft * k_aft,
            bow_fraction * CP_RANGE[1] + l_aft * k_aft)


def solve_transom_exponents(cb: float, cm: float = TRANSOM_CM
                            ) -> tuple[float, float]:
    """목표 Cb → (선수 지수 n, 단면 지수 m). 범위 밖은 명시적 거절."""
    cp = cb / cm
    lo, hi = _transom_cp_bounds()
    if not lo <= cp <= hi:
        raise CbOutOfRangeError(
            f"Cb={cb:.3f} (Cp={cp:.3f})는 트랜섬 계열 범위 밖입니다. "
            f"도달 가능 Cb: [{lo * cm:.3f}, {hi * cm:.3f}] (Cm={cm})"
        )
    k_aft = 1.0 - (1.0 - TRANSOM_BETA) / (TRANSOM_TAPER_P + 1.0)
    a = (cp - (1.0 - TRANSOM_BOW_FRACTION) * k_aft) / TRANSOM_BOW_FRACTION
    n = a / (1.0 - a)
    m = cm / (1.0 - cm)
    return n, m


def _transom_half_breadth(x: float, z: float, dims: MainDimensions,
                          n: float, m: float) -> float:
    x_b = dims.loa / 2 - TRANSOM_BOW_FRACTION * dims.loa  # 선수부 시작
    if x >= x_b:  # 선수부
        xi = (x - x_b) / (dims.loa / 2 - x_b)
        longitudinal = max(0.0, 1.0 - xi ** n)
    else:  # 평행~트랜섬
        v = (x_b - x) / (x_b + dims.loa / 2)
        longitudinal = 1.0 - (1.0 - TRANSOM_BETA) * v ** TRANSOM_TAPER_P
    if z >= dims.draft_design:
        vertical = 1.0
    else:
        vertical = 1.0 - ((dims.draft_design - z) / dims.draft_design) ** m
    return 0.5 * dims.beam * longitudinal * max(0.0, vertical)


def generate_transom_hull_mesh(dims: MainDimensions, n_stations: int = 61,
                               n_below: int = 15, n_above: int = 9
                               ) -> trimesh.Trimesh:
    """트랜섬 선미 watertight 메쉬 (반배수량용)."""
    n, m = solve_transom_exponents(dims.cb)
    return _build_mesh(lambda x, z: _transom_half_breadth(x, z, dims, n, m),
                       dims, n_stations, n_below, n_above, cap_stern=True)


def submerged_transom_area(dims: MainDimensions, draft: float) -> float:
    """흘수 아래 트랜섬 면적 [m²] — 반배수량 저항의 트랜섬 항 입력.

    A_t = ∫₀^T 2·y_t(z) dz,  y_t(z) = β_t·(B/2)·H(z)
    해석 적분: H의 z-적분 = z − T_d·(1−(1−z/T_d)^{m+1})/(m+1) 형태 대신
    수치 적분 (draft가 설계흘수 초과 시 wall-side 구간 포함).
    """
    _, m = solve_transom_exponents(dims.cb)
    zs = np.linspace(0.0, draft, 200)
    td = dims.draft_design
    h = np.where(zs >= td, 1.0, 1.0 - ((td - zs) / td) ** m)
    return float(TRANSOM_BETA * dims.beam * np.trapezoid(h, zs))
