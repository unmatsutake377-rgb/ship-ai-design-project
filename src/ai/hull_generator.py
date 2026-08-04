"""일반화 Wigley 선형 생성기 (spec §2.2, M3 1단 로켓).

반폭: y(x,z) = (B/2)·(1−|2x/L|ⁿ)·(1−((T−z)/T)ᵐ)  (z ≤ T)
       z > T 구간은 수직 현측 (wall-sided), z=D에서 갑판 마감.
계수 관계: Cp = n/(n+1), Cm = m/(m+1), Cb = Cp·Cm.

한계 (spec §2.2에 명시): 선수미가 뾰족함(트랜섬 없음) —
저속 배수량 영역에서만 대표성 있음. 반배수량 지원 전에
트랜섬 선미 계열 필요 (2차 사이클).
"""
from __future__ import annotations

import math

import numpy as np
import trimesh

from src.core.types import MainDimensions

DEFAULT_CM = 0.78          # 구세계 기본 (하위 호환 — 구 리포트 재현용)

# 용도별 중앙단면계수 (2026-08-04, 스펙 hull-bottoms — 오너 승인):
# Ship-D 300척 바닥 폭비율 실측 역산 (f10 25/50/75분위 = Cm
# 0.73/0.85/0.92) + 문헌 (배수량·반배수량 표준 선저 = round bilge,
# 반배수량은 fine 쪽). 활주는 Cm 개념 밖 (V바닥 별도 계열).
CM_BY_PURPOSE = {"survey": 0.85, "workboat": 0.92, "patrol": 0.80}


# 용도별 LCB 오프셋 (2026-08-05, 스펙 asym-hull): (LCB−중앙)/L,
# +가 선수쪽. 근거 = 저속(Fn 0.2~0.25)은 LCB 전방 유리 (문헌 최대
# +3%L) + Ship-D 30,000척 실측 중앙 전방 2.3%L. patrol·활주는 자체
# 비대칭 계열이라 미적용.
LCB_BY_PURPOSE = {"survey": 0.02, "workboat": 0.025}


def lcb_for_purpose(purpose: str) -> float:
    """용도 → LCB 오프셋 목표 (배수량 계열 전용, 그 외 0)."""
    return LCB_BY_PURPOSE.get(purpose, 0.0)


def cm_for_purpose(purpose: str, override: float | None = None) -> float:
    """용도 → 중앙단면계수 (선저 풍만도). override 우선."""
    if override is not None:
        return float(override)
    return CM_BY_PURPOSE.get(purpose, DEFAULT_CM)
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


def _cp_lcb_of(n_b: float, n_s: float) -> tuple[float, float]:
    """비대칭 지수 → (Cp, LCB_frac) 해석식.

    한쪽 면적비 a(n) = n/(n+1), 중앙 기준 모멘트비(반길이²·B/2 단위)
    m(n) = ∫u(1−uⁿ)du = 1/2 − 1/(n+2). LCB_frac = (M_b−M_s)/(A·L)
    — 선수(+x)가 u∈[0,1]에 대응, x = (L/2)u."""
    a_b, a_s = n_b / (n_b + 1.0), n_s / (n_s + 1.0)
    m_b = 0.5 - 1.0 / (n_b + 2.0)
    m_s = 0.5 - 1.0 / (n_s + 2.0)
    cp = 0.5 * (a_b + a_s)
    lcb = (m_b - m_s) / (2.0 * (a_b + a_s))   # (L/2)/L = 1/2 배율
    return cp, lcb


def solve_asym_exponents(cb: float, cm: float = DEFAULT_CM,
                         lcb_frac: float = 0.0,
                         ) -> tuple[float, float]:
    """목표 (Cb, LCB 오프셋) → (선수 지수 n_bow, 선미 지수 n_stern).

    2원 수치 역산 (뉴턴 없이 좌표 강하 이분 — 단조성 이용).
    lcb_frac > 0 = LCB 선수쪽 = 선수 풍만 (n_bow > n_stern)."""
    cp = cb / cm
    if not CP_RANGE[0] <= cp <= CP_RANGE[1]:
        raise CbOutOfRangeError(
            f"Cb={cb:.3f} (Cp={cp:.3f})는 생성기 범위 밖입니다.")
    # 초기값: 대칭 해
    n_b = n_s = cp / (1.0 - cp)
    for _ in range(80):
        # ① n_b/n_s 비율로 LCB 맞추기 (Cp 고정 근사) — r 이분
        lo, hi = 0.02, 50.0
        for _ in range(60):
            r = (lo * hi) ** 0.5              # n_b = r·n_s 기하 탐색
            nb, ns = n_s * r, n_s
            # Cp 보존 재정규화: ns를 Cp에 맞게 이분
            ns_lo, ns_hi = 0.05, 60.0
            for _ in range(50):
                ns_mid = 0.5 * (ns_lo + ns_hi)
                cp_now, _ = _cp_lcb_of(ns_mid * r, ns_mid)
                if cp_now < cp:
                    ns_lo = ns_mid
                else:
                    ns_hi = ns_mid
            ns = 0.5 * (ns_lo + ns_hi)
            nb = ns * r
            _, lcb_now = _cp_lcb_of(nb, ns)
            if lcb_now < lcb_frac:
                lo = r
            else:
                hi = r
        n_b, n_s = nb, ns
        break
    return float(n_b), float(n_s)


def _half_breadth(x: float, z: float, dims: MainDimensions,
                  n: float, m: float, n_stern: float | None = None
                  ) -> float:
    u = abs(2.0 * x / dims.loa)
    n_use = n if (n_stern is None or x >= 0.0) else n_stern
    longitudinal = max(0.0, 1.0 - u ** n_use)
    if z >= dims.draft_design:
        vertical = 1.0  # 수직 현측
    else:
        vertical = 1.0 - ((dims.draft_design - z) / dims.draft_design) ** m
    return 0.5 * dims.beam * longitudinal * max(0.0, vertical)


def generate_hull_mesh(dims: MainDimensions, n_stations: int = 61,
                       n_below: int = 15, n_above: int = 9,
                       cm: float = DEFAULT_CM,
                       lcb_frac: float = 0.0) -> trimesh.Trimesh:
    """watertight 선체 메쉬 생성. 좌표계: x 선수미(+선수), y 좌우, z 상방(0=킬).

    cm: 중앙단면계수 — 선저 풍만도 (CM_BY_PURPOSE, 2026-08-04).
    lcb_frac: LCB 오프셋 목표 (+선수쪽, 스펙 asym-hull 2026-08-05).
    기본 0.0 = 선수미 대칭 — Michell 해석해 대조의 표준기 채널.
    파이프라인 설계 경로의 기본은 용도 프리셋(비대칭)."""
    if abs(lcb_frac) < 1e-9:
        n, m = solve_exponents(dims.cb, cm)
        return _build_mesh(lambda x, z: _half_breadth(x, z, dims, n, m),
                           dims, n_stations, n_below, n_above,
                           cap_stern=False)
    n_b, n_s = solve_asym_exponents(dims.cb, cm, lcb_frac)
    _, m = solve_exponents(dims.cb, cm)
    return _build_mesh(
        lambda x, z: _half_breadth(x, z, dims, n_b, m, n_stern=n_s),
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


# ---------- 활주 계열 (Phase C-2) ----------
# V바닥 데드라이즈 프리즘 + 하드 차인 + 넓은 트랜섬 — Savitsky 전제 형상.
# 주의: 이 계열은 Cb를 목표로 역산하지 않음 — 데드라이즈·차인 기하가
# 단면을 결정 (dims.cb는 참고값). 부양·필터는 메쉬 기반이라 무관.

# 데드라이즈 15° — A급 근거 승격 (2026-08-04 조사): Savitsky 방법
# 자체가 10~30° 프리즘 실험 회귀 (Davidson Lab); 실선 트랜섬 통상
# 10~24° (semi-V 10~17 / deep-V 21~25); 소형·RC 표준 15°.
# deep-V 20° (거친 물) 옵션은 백로그.
PLANING_DEADRISE_DEG = 15.0
PLANING_TRANSOM_RATIO = 0.90  # 트랜섬 폭비 (활주정은 선미가 거의 전폭)
PLANING_BOW_FRACTION = 0.35   # 선수 테이퍼 구간 / 전장


def _planing_half_breadth(x: float, z: float, dims: MainDimensions) -> float:
    # 단면: V바닥 (y = z/tanβ) → 차인 높이에서 수직 현측
    chine_z = (dims.beam / 2.0) * math.tan(
        math.radians(PLANING_DEADRISE_DEG))
    section = min(z / math.tan(math.radians(PLANING_DEADRISE_DEG)),
                  dims.beam / 2.0) if z > 0 else 0.0
    if z > chine_z:
        section = dims.beam / 2.0
    # 세로: 선수 테이퍼 + 전폭 선미 (트랜섬)
    x_b = dims.loa / 2 - PLANING_BOW_FRACTION * dims.loa
    if x >= x_b:
        xi = (x - x_b) / (dims.loa / 2 - x_b)
        longitudinal = max(0.0, 1.0 - xi ** 2.5)
    else:
        v = (x_b - x) / (x_b + dims.loa / 2)
        longitudinal = 1.0 - (1.0 - PLANING_TRANSOM_RATIO) * v ** 2
    return section * longitudinal


def generate_planing_hull_mesh(dims: MainDimensions, n_stations: int = 61,
                               n_below: int = 15, n_above: int = 9
                               ) -> trimesh.Trimesh:
    """활주 선체 watertight 메쉬 (Savitsky 전제 — 프리즘 데드라이즈)."""
    return _build_mesh(lambda x, z: _planing_half_breadth(x, z, dims),
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
