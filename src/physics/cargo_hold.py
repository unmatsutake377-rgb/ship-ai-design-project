"""다구획 MaxBox — 적재 공간 모형 + 분산 적재 (스펙 2026-08-03, 1단계).

오너 발상: "각 배마다 실제 적재 공간을 먼저 만들고, 그 안에서 N개
자유로 채우면 확실한 부피와 무게가 나온다."

층 구조:
  1층 적재 공간 = 선체 내부 − 예약 구역 (모터·배터리·전장).
     예약 위치는 중량 분포모델(M4a)과 같은 배치: 추진은 선미
     (LCG −0.45L) → 선미 x-구간을 부피 일치로 통째 절단 (보수적).
  2층 다구획 = 남은 공간의 반폭 지형도에서 최대 상자를 반복 채취.
     칸막이는 지형도 골짜기(병목)에서 자연 발생. 상자당 최소 치수
     제약으로 "종잇장 상자 합산" 과관대를 방지.

단일 MaxBox(maxbox.py)는 Wigley 최적화기 경로에 그대로 — 이 모듈은
Ship-D 실선형 경로용 (단일 상자 가정이 과혹했던 실측이 유래, #27).

정직 한계 (스펙 §6): 상자 축 정렬·중심선 대칭·갑판 접촉, 짐은 균질
주입 근사, 예약 여유계수 3.0 개략, z0는 전 구획 공유.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from src.physics.maxbox import BoxReport

# 예약 부피 환산 — 배터리 밀도는 A급 실측 (data/payload_items.csv,
# Blue Robotics Li-ion 1,843 kg/m³). 여유계수는 케이블·마운트·접근
# 공간 개략 (#17 수집으로 승격 대상).
RESERVED_DENSITY_KG_M3 = 1843.0
RESERVE_PACKING_FACTOR = 3.0

# 상자 최소 치수 [m] — 실측 화물 참고 (배터리 0.15 m, 플로트 패키지
# 폭 0.8 m). 이보다 작은 변을 가진 "종잇장 상자"는 실제 짐이 못
# 들어가므로 합산 금지 (과혹→과관대 반전 방지, 스펙 §2).
MIN_BOX_DIM = 0.25

# 구획 수 상한 — 소형정 관행 개략 (앞/중/뒤+예비). 구획별 z 차등
# DP가 상한 없이는 3 m 보트에 9칸을 세우는 과관대 실측 (2026-08-03,
# 칸막이 중량·공사는 미모형이므로 개수로 현실성 방어 — 종잇장 가드와
# 같은 철학). #17 실선 수집으로 승격 대상.
MAX_BAYS = 4


@dataclass(frozen=True)
class HoldReport:
    boxes: tuple            # BoxReport 튜플 (구획들)
    total_volume: float     # Σ 상자 부피 [m³]
    reserved_volume: float  # 예약 구역 부피 [m³]
    stern_cut_x: float      # 절단 경계 (이보다 선미쪽 = 예약) [m]
    z0: float               # 공유 상자 바닥 높이 [m]
    interior_volume: float = 0.0   # 선체 내부 부피 (단면 적분) [m³]
    interior_cx: float = 0.0       # 내부 부피 도심 x — 구조 LCG 근사용


def reserved_volume_for(propulsion_mass_kg: float) -> float:
    """추진(모터·배터리·전장) 질량 → 예약 부피 [m³]."""
    return propulsion_mass_kg / RESERVED_DENSITY_KG_M3 \
        * RESERVE_PACKING_FACTOR


def pack_bays(half_widths: list[float], dx: float, height: float,
              min_dim: float = MIN_BOX_DIM) -> list[tuple]:
    """반폭 지형도에서 상자 반복 채취 — N개 자유 + 최소 치수.

    반환: (길이, 폭, 높이, 부피, x0, x1) 목록 (지형도 좌표계).
    알고리즘: 최대 직사각형 → 그 구간 소거(반폭 0) → 반복.
    채택 조건: 길이·폭·높이 모두 ≥ min_dim."""
    from src.physics.maxbox import largest_rectangle

    if height < min_dim:
        return []
    w = list(half_widths)
    boxes = []
    while True:
        area, x0, x1, half_w = largest_rectangle(w, dx)
        length, width = x1 - x0, 2.0 * half_w
        if area <= 0.0 or length < min_dim or width < min_dim:
            break
        boxes.append((length, width, height, length * width * height,
                      x0, x1))
        i0, i1 = int(round(x0 / dx)), int(round(x1 / dx))
        for i in range(i0, i1):
            w[i] = 0.0
    return boxes


def pack_bays_z(w_xz: np.ndarray, dx: float, zs: np.ndarray, dz: float,
                mask: np.ndarray, min_dim: float = MIN_BOX_DIM,
                max_bays: int = MAX_BAYS) -> list[tuple]:
    """구획별 [z0, z1] 차등 채우기 — x-분할 DP (최적 보장).

    모형: 구획은 x-구간 분할 (세로 칸막이 원칙 — 수직 적층은 범위 밖),
    각 구획이 자기만의 z-띠를 고름. 그리디는 "낮고 넓은 상자"가
    동점에서 전 구간을 선점하는 함정 (계단 지형도 실측 0.4 < 최적
    0.6) — 분할 DP로 교체: 구간값 v[i,j] = 그 x-구간의 최적 띠 부피,
    best[j] = max(best[j-1], max_i best[i] + v[i,j-1]).
    반환: (길이, 폭, 높이, 부피, x0, x1, z0_abs) 목록."""
    n_x, n_z = w_xz.shape

    # 구간값: v[i][j] = 셀 i..j(포함)를 한 구획으로 쓸 때 최적 부피
    NEG = (0.0, 0, 0)          # (부피, iz0, iz1)
    val = [[NEG] * n_x for _ in range(n_x)]
    for i in range(n_x):
        if not mask[i]:
            continue
        run_min = w_xz[i].copy()
        for j in range(i, n_x):
            if not mask[j]:
                break
            if j > i:
                run_min = np.minimum(run_min, w_xz[j])
            length = (j - i + 1) * dx
            if length < min_dim:
                continue
            best_v = NEG
            for iz0 in range(n_z):
                m = float("inf")
                for iz1 in range(iz0, n_z):
                    m = min(m, float(run_min[iz1]))
                    height = float(zs[iz1] - zs[iz0]) + dz
                    width = 2.0 * m
                    if height < min_dim or width < min_dim:
                        continue
                    vol = length * width * height
                    if vol > best_v[0]:
                        best_v = (vol, iz0, iz1)
            val[i][j] = best_v

    # 분할 DP (구획 수 차원 포함) + 경로 복원. SPLIT_EPS = 칸막이
    # 1개당 미세 벌점 — 부피 동점이면 구획 적은 분할 선택.
    SPLIT_EPS = 1e-9
    K = max_bays
    best = [[0.0] * (K + 1) for _ in range(n_x + 1)]
    choice: list = [[None] * (K + 1) for _ in range(n_x + 1)]
    for j in range(n_x):
        for k in range(K + 1):
            best[j + 1][k] = best[j][k]
            choice[j + 1][k] = None
            if k == 0:
                continue
            for i in range(j + 1):
                cand = best[i][k - 1] + val[i][j][0] - SPLIT_EPS
                if val[i][j][0] > 0.0 and cand > best[j + 1][k]:
                    best[j + 1][k] = cand
                    choice[j + 1][k] = (i, j)

    boxes = []
    j, k = n_x, K
    while j > 0 and k > 0:
        if choice[j][k] is None:
            j -= 1
            continue
        i, j1 = choice[j][k]
        vol, iz0, iz1 = val[i][j1]
        # 최소 부피 구간으로 축약: 같은 띠에서 더 짧아도 부피 동일한
        # 경우는 없음 (부피 ∝ 길이) — 구간 그대로 상자화
        length = (j1 - i + 1) * dx
        height = float(zs[iz1] - zs[iz0]) + dz
        width = vol / (length * height)
        boxes.append((length, width, height, vol,
                      i * dx, (j1 + 1) * dx, float(zs[iz0]) - dz / 2.0))
        j = i
        k -= 1
    boxes.reverse()
    return boxes


def multibay_hold(mesh: trimesh.Trimesh, depth: float,
                  reserved_volume: float, stern: str = "xmax",
                  n_x: int = 36, n_z: int = 12,
                  min_dim: float = MIN_BOX_DIM) -> HoldReport:
    """적재 공간(예약 절단) + 다구획 채우기.

    stern: 예약 구역을 절단할 끝 ("xmax" | "xmin").
    Ship-D 관례: 뱃머리 x=0 → 선미 = xmax.

    내부 판정은 contains 격자가 아니라 **단면 절단법**: Ship-D
    메쉬는 비수밀이라 contains(광선 홀짝 판정)가 오작동함 (2026-08-03
    실측: 정중앙 셀 내부율 7% — 물리 불가). 단면은 수밀 불요 —
    정역학 모듈과 같은 접근. 가정: 단면이 중심선에서 별 모양
    (y-단조) — 통상 선형에서 성립."""
    (xmin, ymin, zmin), (xmax, ymax, _) = mesh.bounds
    # 스테이션(끝 회피 셀 중심)에서 단면 → 높이별 반폭 지형도
    dx = (xmax - xmin) / n_x
    xs = xmin + (np.arange(n_x) + 0.5) * dx
    dz = (depth - zmin) / n_z
    zs = zmin + (np.arange(n_z) + 0.5) * dz

    w_xz = np.zeros((n_x, n_z))
    for i, x in enumerate(xs):
        sec = mesh.section(plane_origin=[float(x), 0.0, 0.0],
                           plane_normal=[1.0, 0.0, 0.0])
        if sec is None or not len(sec.entities):
            continue
        # sec.discrete가 빈 리스트를 주는 trimesh 판 존재 — entity
        # 직접 순회. 반폭은 스캔라인 교차 보간 (z-빈 방식은 꼭짓점
        # 빈틈으로 새는 박제 버그, 2026-07-27 — resistance.py 규약)
        for e in sec.entities:
            pts = np.asarray(e.discrete(sec.vertices))
            if len(pts) < 2:
                continue
            z1, z2 = pts[:-1, 2], pts[1:, 2]
            y1, y2 = pts[:-1, 1], pts[1:, 1]
            dz_e = z2 - z1
            valid = np.abs(dz_e) > 1e-12
            for k, z in enumerate(zs):
                crossing = valid & (np.minimum(z1, z2) <= z) \
                    & (z <= np.maximum(z1, z2))
                if crossing.any():
                    t = (z - z1[crossing]) / dz_e[crossing]
                    y_at = y1[crossing] + t * (y2[crossing] - y1[crossing])
                    w_xz[i, k] = max(w_xz[i, k], float(np.abs(y_at).max()))

    # ① 부피 일치 선미 절단: 단면적 적분으로 열당 부피
    col_vol = w_xz.sum(axis=1) * dz * 2.0 * dx     # [m³]
    order = range(n_x - 1, -1, -1) if stern == "xmax" else range(n_x)
    cut_idx = n_x if stern == "xmax" else -1
    acc = 0.0
    for i in order:
        if acc >= reserved_volume:
            break
        acc += float(col_vol[i])
        cut_idx = i
    # 경계는 상자 좌표와 같은 눈금 (히스토그램 칸 i = [i·dx, (i+1)·dx])
    if stern == "xmax":
        keep = slice(0, cut_idx)
        stern_cut_x = float(xmin + cut_idx * dx)
    else:
        keep = slice(cut_idx + 1, n_x)
        stern_cut_x = float(xmin + (cut_idx + 1) * dx)

    # ② 구획별 [z0, z1] 차등 채우기 (2026-08-03 후속 확장) —
    # 천장=갑판 가정 해제에 이어 바닥·천장을 구획마다 따로 고름.
    # 정직 한계: 해치(짐 넣는 구멍) 접근성은 무시 — 균질 주입 근사.
    mask = np.zeros(n_x, dtype=bool)
    mask[keep] = True
    bays = pack_bays_z(w_xz, dx, zs, dz, mask, min_dim=min_dim)

    total = sum(b[3] for b in bays)
    boxes = tuple(
        BoxReport(length=b[0], width=b[1], height=b[2], volume=b[3],
                  x0=xmin + b[4], x1=xmin + b[5], z0=b[6])
        for b in bays)
    z0_min = min((b[6] for b in bays), default=float(zs[0]))
    iv = float(col_vol.sum())
    icx = float((col_vol * xs).sum() / iv) if iv > 0 else float(xmin)
    return HoldReport(boxes=boxes, total_volume=total,
                      reserved_volume=reserved_volume,
                      stern_cut_x=stern_cut_x, z0=z0_min,
                      interior_volume=iv, interior_cx=icx)


# ── 2단계: KG/GM 구간 판정 (스펙 §3) ─────────────────────────────
#
# 짐을 구획들에 나누는 비율에 따라 짐 무게중심(KG_c)이 움직인다.
# 채움 물리: 구획에 부은 짐은 바닥부터 h_i = m_i/(ρ·A_i)로 차오름
# (균질 주입 근사) → 그 몫의 중심 = z0 + h_i/2.
#   최저 KG_c = 균일 수위 (전 구획에 얇게 폄)
#   최고 KG_c = 바닥면적 작은 구획부터 기둥으로 쌓기 (그리디 —
#               같은 부피로 가장 높은 기둥을 만드는 순서)
# 총 KG는 짐 KG의 1차 함수 → 구간 끝점이 그대로 사상. GM = KM − KG
# 도 1차 → "GM/B 도달 구간 ∩ 합격 밴드 ≠ ∅" 겹침 검사로 존재 판정.


@dataclass(frozen=True)
class BayGeom:
    """구간 판정용 구획 기하 (바닥면적·바닥 높이·천장까지 높이)."""
    floor_area: float   # [m²]
    z0: float           # 바닥 높이 (킬 기준) [m]
    height: float       # 구획 높이 [m]


def bays_from_hold(hold: HoldReport) -> list[BayGeom]:
    """HoldReport → 구간 판정용 기하 (z0·높이는 전 구획 공유)."""
    return [BayGeom(floor_area=b.length * b.width, z0=b.z0,
                    height=b.height) for b in hold.boxes]


def cargo_kg_interval(bays: list[BayGeom], payload_kg: float,
                      density: float) -> tuple[float, float] | None:
    """짐 무게중심 높이 도달 구간 [최저, 최고]. 용량 초과면 None."""
    if not bays or payload_kg <= 0.0:
        return None
    vol_need = payload_kg / density
    caps = [b.floor_area * b.height for b in bays]
    if vol_need > sum(caps) * (1.0 + 1e-9):
        return None

    # 최저: 균일 수위 — 낮은 천장 구획이 먼저 차면 넘치는 몫을 나머지에
    # (여기선 z0·height 공유 가정이라 단순 수위 계산으로 충분하지만,
    #  일반형으로 구현: 수위를 이분 탐색)
    zs0 = min(b.z0 for b in bays)
    lo_lv, hi_lv = 0.0, max(b.z0 + b.height for b in bays) - zs0
    for _ in range(60):
        mid = 0.5 * (lo_lv + hi_lv)
        filled = sum(b.floor_area * min(max(mid - (b.z0 - zs0), 0.0),
                                        b.height) for b in bays)
        if filled < vol_need:
            lo_lv = mid
        else:
            hi_lv = mid
    level = zs0 + hi_lv
    m_tot = 0.0
    zm_tot = 0.0
    for b in bays:
        h = min(max(level - b.z0, 0.0), b.height)
        m = b.floor_area * h * density
        m_tot += m
        zm_tot += m * (b.z0 + h / 2.0)
    kg_min = zm_tot / max(m_tot, 1e-12)

    # 최고: 바닥면적 오름차순으로 꽉꽉 채움 (좁은 기둥 우선)
    remain = vol_need
    m_tot = 0.0
    zm_tot = 0.0
    for b in sorted(bays, key=lambda b: b.floor_area):
        v = min(remain, b.floor_area * b.height)
        if v <= 0.0:
            break
        h = v / b.floor_area
        m = v * density
        m_tot += m
        zm_tot += m * (b.z0 + h / 2.0)
        remain -= v
    kg_max = zm_tot / max(m_tot, 1e-12)
    return float(kg_min), float(kg_max)


def gm_band_reachable(cargo_kg_lo: float, cargo_kg_hi: float, km: float,
                      fixed_moment: float, cargo_mass: float,
                      total_mass: float, beam: float,
                      band: tuple[float, float]
                      ) -> tuple[bool, float]:
    """GM/B 밴드 겹침 판정 → (합격 배치 존재?, 최선 여유).

    fixed_moment = Σ(짐 외 성분 질량 × 그 VCG) [kg·m].
    총 KG(k) = (fixed_moment + 짐질량·k)/총질량, GM = KM − KG.
    여유 = 도달 구간 안에서 만들 수 있는 최대 밴드 여유 (음수 = 불합격,
    크기는 밴드까지 거리)."""
    kg_lo = (fixed_moment + cargo_mass * cargo_kg_lo) / total_mass
    kg_hi = (fixed_moment + cargo_mass * cargo_kg_hi) / total_mass
    gmb_hi = (km - kg_lo) / beam          # 짐 낮게 = GM 크게
    gmb_lo = (km - kg_hi) / beam
    lo, hi = band
    # 도달 구간 [gmb_lo, gmb_hi] 안에서 밴드 중앙에 최대한 접근
    best = min(max((lo + hi) / 2.0, gmb_lo), gmb_hi)
    margin = min(best - lo, hi - best)
    return bool(margin >= 0.0), float(margin)


# ── 3단계: 트림 구간 판정 (스펙 §3) ──────────────────────────────
#
# 짐 배분에 따라 짐 세로 무게중심(LCG_c)이 구획 중심들 사이를
# 미끄러진다. 상자 안 짐의 x중심 = 상자 x중심 (균질 주입) → 도달
# 구간 끝점 = "한쪽 끝 구획부터 용량껏 채우기" 그리디 (용량 상한이
# 있는 분수 배낭 — 정확해).


@dataclass(frozen=True)
class BayGeomX:
    """트림 판정용 구획 기하 (x중심·용량 부피)."""
    x_center: float     # [m]
    cap_volume: float   # [m³]


def bays_x_from_hold(hold: HoldReport) -> list[BayGeomX]:
    return [BayGeomX(x_center=0.5 * (b.x0 + b.x1), cap_volume=b.volume)
            for b in hold.boxes]


def cargo_lcg_interval(bays: list[BayGeomX], payload_kg: float,
                       density: float) -> tuple[float, float] | None:
    """짐 세로 무게중심 도달 구간 [최소 x, 최대 x]. 용량 초과면 None."""
    if not bays or payload_kg <= 0.0:
        return None
    vol_need = payload_kg / density
    if vol_need > sum(b.cap_volume for b in bays) * (1.0 + 1e-9):
        return None

    def extreme(reverse: bool) -> float:
        remain = vol_need
        moment = 0.0
        for b in sorted(bays, key=lambda b: b.x_center, reverse=reverse):
            v = min(remain, b.cap_volume)
            moment += v * b.x_center
            remain -= v
            if remain <= 0.0:
                break
        return moment / vol_need

    return extreme(False), extreme(True)   # (x 작은 쪽 몰기, 큰 쪽 몰기)


# 운용 트림 한계 [deg] — 소형정 개략 명명 상수 (#17 승격 대상).
# 실선은 만재 트림을 L의 0.5~1% 기울기 이내로 관리하는 관행.
TRIM_MAX_DEG = 2.0
