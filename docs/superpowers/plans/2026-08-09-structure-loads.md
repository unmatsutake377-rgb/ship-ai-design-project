# 구조 강도 1단계 — 하중 곡선 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 중량−부력 분포에서 정수 전단력·굽힘 모멘트 곡선을 만들고, IACS 파랑 굽힘 공식 + 표준파 준정적 + 스트립 동적 굽힘 3중 교차검증으로 1단계 성적표를 낸다.

**Architecture:** 신설 `src/physics/structure/` 패키지. `loads.py`(정수 곡선) → `wave_loads.py`(IACS 정본 + 표준파 준정적) → `strip_loads.py`(스트립 동적 굽힘 RAO — 기존 `seakeeping/strip.py` 헬퍼 추출 재사용). 검증 앵커: 상자 바지선 해석해 (|M_mid| = WL/16, 파랑 ρgBaL²/2π²) + 폐합 항등식 (V·M 양끝 0).

**Tech Stack:** Python 3, numpy, trimesh (기존 메쉬 절단 인프라), pytest.

## Global Constraints

- 부호 관례: **q(x) = w(x) − b(x), V = ∫q, M = ∫V, M > 0 = 호깅** (IACS hog 양수와 정합) — 모든 모듈 공통, docstring 명기
- 한국어 docstring (프로젝트 관례)
- IACS 공식 상수는 Task 1에서 원전 대조 후 확정 (기억 하드코딩 금지 — 다르면 코드가 아니라 계획이 틀린 것, 원전이 정답)
- 기존 시험 350개 전부 통과 유지 (특히 strip.py 리팩터 후 내항 앵커 시험)
- e2e급 느린 시험은 별도 파일 격리 (슈트 11분 악화 방지)
- main 직커밋·직푸시 (프로젝트 관례)

---

### Task 1: 원전 확보 — IACS UR S11

**Files:**
- Create: `references/IACS_UR_S11.pdf` (gitignore 대상 — 커밋 안 됨)
- Modify: `docs/superpowers/specs/2026-08-09-structure-design.md` (§3에 구현 인덱스 추가)

**Interfaces:**
- Produces: 확정된 파랑 굽힘 공식 (Cw 구간식·Mw 호깅/새깅 상수) — Task 4가 이 값을 박제

- [ ] **Step 1: UR S11 PDF 다운로드 시도**

```bash
cd "/Users/isong-eon/Desktop/선박 ai 모델 프로젝트"
# IACS 공식 사이트의 UR S 목록에서 S11 최신 개정판 URL 확인 후 다운로드
curl -sL "https://iacs.org.uk/resolutions/unified-requirements/ur-s" -o /tmp/urs_list.html || true
# 목록 페이지에서 S11 PDF 링크 추출 (링크 형식 바뀌면 WebFetch로 페이지 읽고 수동 확인)
grep -io 'href="[^"]*S11[^"]*\.pdf"' /tmp/urs_list.html | head -5
```

링크 확인 후:

```bash
curl -sL "<찾은 S11 PDF URL>" -o "references/IACS_UR_S11.pdf" && file "references/IACS_UR_S11.pdf"
```

Expected: `PDF document`. 실패 시 폴백: DNV-RU-SHIP Pt.3 Ch.4 (dnv.com 공개 다운로드 — 동일 IACS 계보 공식 수록). 둘 다 실패면 정직 기록 후 공개 교재 공식으로 C급 표기 (SEAWAY 대체 관례).

- [ ] **Step 2: 공식 검증 — Read 도구로 PDF 열람**

확인 항목 (0-기준 페이지와 함께 기록):
1. 파랑계수 Cw 구간식 (90≤L≤300: `10.75 − ((300−L)/100)^1.5`, 300<L≤350: `10.75`, 350<L≤500: `10.75 − ((L−350)/150)^1.5` — 예상값, 원전과 다르면 원전이 정답)
2. 호깅: `Mw = +190·Cw·L²·B·Cb·10⁻³ [kN·m]`
3. 새깅: `Mw = −110·Cw·L²·B·(Cb+0.7)·10⁻³ [kN·m]`
4. 적용 범위 (L 하한 — 90m 예상) 및 Cb 하한 (0.60 예상 — 우리 소형은 범위 밖, 정직 거절 근거)

- [ ] **Step 3: 스펙에 구현 인덱스 추가**

`docs/superpowers/specs/2026-08-09-structure-design.md` §3 끝에 추가:

```markdown
### §3 보강 — IACS UR S11 구현 인덱스 (2026-08-09 확보)

- 확보 경로: <실제 URL> → references/IACS_UR_S11.pdf (gitignore)
- p<N>: Cw 파랑계수 구간식 <확인한 식 그대로>
- p<N>: Mw 호깅/새깅 <확인한 상수>
- 적용 범위: L <하한>~<상한> m, Cb ≥ <하한> — 범위 밖 정직 거절
```

- [ ] **Step 4: 커밋**

```bash
git add docs/superpowers/specs/2026-08-09-structure-design.md
git commit -m "docs: IACS UR S11 원전 확보 — 파랑 굽힘 공식 구현 인덱스"
```

---

### Task 2: `loads.py` — 중량 길이 분포

**Files:**
- Create: `src/physics/structure/__init__.py` (빈 파일)
- Create: `src/physics/structure/loads.py`
- Test: `tests/test_structure_loads.py`

**Interfaces:**
- Produces:
  - `WeightBlock = tuple[float, float, float]` — (mass_kg, x0_m, x1_m) 균일 블록
  - `standard_weight_blocks(component_masses_kg: dict[str, float], xmin: float, loa: float) -> list[WeightBlock]`
  - `weight_linear_density(xs: np.ndarray, blocks: list[WeightBlock]) -> np.ndarray` — w(x) [N/m], 격자 정규화로 ∫w = Σm·g 정확 폐합
  - `_cumtrapz(y, x) -> np.ndarray` — 누적 사다리꼴 (Task 3·6 공용)

- [ ] **Step 1: 실패 시험 작성**

```python
# tests/test_structure_loads.py
"""구조 강도 1단계 — 하중 곡선 시험 (스펙 2026-08-09 §5-1)."""
import numpy as np
import pytest
import trimesh

G = 9.81
RHO = 1025.0


def test_weight_blocks_closure():
    """성분 블록 합 = 총중량 (폐합 항등식)."""
    from src.physics.structure.loads import (
        standard_weight_blocks, weight_linear_density)
    comp = {"structure": 800.0, "outfit": 200.0, "machinery": 300.0,
            "fuel": 100.0, "payload": 600.0}
    blocks = standard_weight_blocks(comp, xmin=-40.0, loa=80.0)
    xs = np.linspace(-40.0, 40.0, 201)
    w = weight_linear_density(xs, blocks)
    total = np.trapezoid(w, xs)
    assert total == pytest.approx(sum(comp.values()) * G, rel=1e-9)
    assert np.all(w >= 0)


def test_weight_blocks_placement():
    """기관·연료 = 선미 구간, 화물 = 중앙 구간 (통상 배치)."""
    from src.physics.structure.loads import standard_weight_blocks
    blocks = standard_weight_blocks(
        {"machinery": 100.0, "payload": 100.0}, xmin=0.0, loa=100.0)
    named = {}
    for (m, x0, x1), name in zip(blocks, ["machinery", "payload"]):
        named[name] = (x0, x1)
    m0, m1 = named["machinery"]
    p0, p1 = named["payload"]
    assert m1 <= 30.0          # 기관실 = 선미 30% 안
    assert 20.0 <= p0 and p1 <= 90.0   # 화물창 = 중앙부
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_structure_loads.py -v`
Expected: FAIL `ModuleNotFoundError: src.physics.structure`

- [ ] **Step 3: 구현**

```python
# src/physics/structure/loads.py
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
```

`src/physics/structure/__init__.py`는 빈 파일로 생성.

- [ ] **Step 4: 통과 확인**

Run: `python -m pytest tests/test_structure_loads.py -v`
Expected: 2 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/physics/structure/ tests/test_structure_loads.py
git commit -m "feat: 구조 하중 1차 — 성분별 중량 길이 분포 (폐합 정규화)"
```

---

### Task 3: `loads.py` — 부력 분포 + 정수 전단·모멘트 곡선

**Files:**
- Modify: `src/physics/structure/loads.py` (함수 추가)
- Test: `tests/test_structure_loads.py` (시험 추가)

**Interfaces:**
- Consumes: `weight_linear_density`, `_cumtrapz` (Task 2)
- Produces:
  - `station_area(mesh, x: float, waterline_z: float) -> float` — 수선 아래 단면적 [m²] (Lewis 제약 없는 전용 절단 — Task 5·6 재사용)
  - `LoadCurves` dataclass: `xs, weight_npm, buoy_npm, shear_n, moment_nm, buoy_scale, shear_residual_n, moment_residual_nm`
  - `still_water_curves(mesh, draft: float, blocks, n: int = 101) -> LoadCurves`

- [ ] **Step 1: 실패 시험 작성 — 상자 바지선 해석해 앵커**

`tests/test_structure_loads.py`에 추가:

```python
def _box_barge(loa=80.0, beam=10.0, depth=6.0):
    return trimesh.creation.box(extents=[loa, beam, depth])


def test_still_water_uniform_barge_zero_moment():
    """균일 중량 상자 바지선 → M(x) ≈ 0 전역 (분포 일치 항등식)."""
    from src.physics.structure.loads import still_water_curves
    mesh = _box_barge()
    draft_T = 2.0
    mass = RHO * 80.0 * 10.0 * draft_T
    wl_z = -3.0 + draft_T                      # z ∈ [-3, 3] 상자
    curves = still_water_curves(mesh, wl_z, [(mass, -40.0, 40.0)])
    scale = mass * G * 80.0 / 16.0             # WL/16 기준 스케일
    assert np.max(np.abs(curves.moment_nm)) < 0.01 * scale


def test_still_water_midship_cargo_analytic():
    """중앙 절반 몰림 → |M_mid| = WL/16, 부호 음(새깅) — 손계산 앵커."""
    from src.physics.structure.loads import still_water_curves
    mesh = _box_barge()
    draft_T = 2.0
    mass = RHO * 80.0 * 10.0 * draft_T
    wl_z = -3.0 + draft_T
    curves = still_water_curves(mesh, wl_z, [(mass, -20.0, 20.0)],
                                n=201)
    m_mid = curves.moment_nm[len(curves.xs) // 2]
    m_analytic = mass * G * 80.0 / 16.0
    assert m_mid == pytest.approx(-m_analytic, rel=0.02)
    # 폐합 항등식: 양끝 V·M 잔차가 최대값 대비 미소
    assert abs(curves.shear_residual_n) < 0.02 * np.max(
        np.abs(curves.shear_n))


def test_station_area_box():
    """상자 단면적 = B×t 해석해."""
    from src.physics.structure.loads import station_area
    mesh = _box_barge()
    a = station_area(mesh, 0.0, -1.0)          # 흘수 2m (킬 -3)
    assert a == pytest.approx(10.0 * 2.0, rel=0.01)
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_structure_loads.py -v`
Expected: 신규 3개 FAIL `ImportError: still_water_curves`

- [ ] **Step 3: 구현**

`loads.py`에 추가:

```python
from dataclasses import dataclass


def station_area(mesh, x: float, waterline_z: float,
                 nz: int = 60) -> float:
    """스테이션 x, 수선 z 아래 몰수 단면적 [m²].

    seakeeping.sections.station_geometry 계보 절단이지만 Lewis 제약
    (수선폭 필터) 없음 — 하중 적분은 벌브·선수미 단면도 셈."""
    sec = mesh.section(plane_origin=[float(x), 0, 0],
                       plane_normal=[1, 0, 0])
    if sec is None or not len(sec.entities):
        return 0.0
    pts = np.vstack([e.discrete(sec.vertices) for e in sec.entities])
    below = pts[pts[:, 2] <= waterline_z + 1e-9]
    pos = below[below[:, 1] > 1e-12]
    if len(pos) < 4:
        return 0.0
    order = np.argsort(pos[:, 2])
    zs_raw = pos[order][:, 2]
    ys_raw = pos[order][:, 1]
    z_keel = float(below[:, 2].min())
    if waterline_z - z_keel < 1e-6:
        return 0.0
    zs = np.linspace(z_keel, waterline_z, nz)
    halves = np.interp(zs, zs_raw, ys_raw)
    return 2.0 * float(np.trapezoid(halves, zs))


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
```

- [ ] **Step 4: 통과 확인**

Run: `python -m pytest tests/test_structure_loads.py -v`
Expected: 전부 PASS. 부호 확인 포인트: 중앙 몰림 = 새깅 = 음수 — 틀리면 q 부호부터 의심 (관례 M>0 호깅).

- [ ] **Step 5: 커밋**

```bash
git add src/physics/structure/loads.py tests/test_structure_loads.py
git commit -m "feat: 정수 전단·굽힘 곡선 — 상자 바지선 WL/16 해석해 앵커"
```

---

### Task 4: `wave_loads.py` — IACS 파랑 굽힘 정본

**Files:**
- Create: `src/physics/structure/wave_loads.py`
- Test: `tests/test_wave_loads.py`

**Interfaces:**
- Produces:
  - `iacs_wave_coefficient(l_m: float) -> float` — Cw (범위 밖 `IACSRangeError`)
  - `iacs_wave_bending_knm(l_m, b_m, cb) -> tuple[float, float]` — (호깅 +, 새깅 −) [kN·m]
  - `class IACSRangeError(ValueError)`

주의: 아래 상수는 Task 1 원전 대조 결과로 확정 — 원전과 다르면 원전 값으로 교체하고 시험 기대값도 갱신.

- [ ] **Step 1: 실패 시험 작성**

```python
# tests/test_wave_loads.py
"""IACS UR S11 파랑 굽힘 + 표준파 준정적 (스펙 2026-08-09 §2·§3)."""
import numpy as np
import pytest
import trimesh

G = 9.81
RHO = 1025.0


def test_iacs_cw_anchor_values():
    """원전 구간식 재현 — L=300에서 10.75 (최대 구간 진입점)."""
    from src.physics.structure.wave_loads import iacs_wave_coefficient
    assert iacs_wave_coefficient(300.0) == pytest.approx(10.75)
    assert iacs_wave_coefficient(320.0) == pytest.approx(10.75)
    # 90~300 구간: L=200 → 10.75 − 1.0 = 9.75
    assert iacs_wave_coefficient(200.0) == pytest.approx(9.75)
    # 단조 증가 (90~300)
    ls = np.linspace(90.0, 300.0, 50)
    cws = [iacs_wave_coefficient(l) for l in ls]
    assert all(a <= b + 1e-12 for a, b in zip(cws, cws[1:]))


def test_iacs_bending_signs_and_magnitude():
    """호깅 양수·새깅 음수, 100m 화물선 자릿수 (1e5 kN·m 대역)."""
    from src.physics.structure.wave_loads import iacs_wave_bending_knm
    hog, sag = iacs_wave_bending_knm(100.0, 15.0, 0.75)
    assert hog > 0 > sag
    assert 5e4 < hog < 5e5


def test_iacs_range_honest_rejection():
    """적용 범위 밖 (소형선) = 정직 거절."""
    from src.physics.structure.wave_loads import (
        IACSRangeError, iacs_wave_coefficient)
    with pytest.raises(IACSRangeError):
        iacs_wave_coefficient(10.0)
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_wave_loads.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 구현**

```python
# src/physics/structure/wave_loads.py
"""파랑 굽힘 모멘트 — IACS UR S11 정본 + 표준파 준정적 교차검증.

IACS: 선급 통일 규칙 설계 파랑 굽힘 (극치 통계 내장, 10⁻⁸ 확률
수준). 적용 범위 밖(소형선)은 정직 거절 — 소형 종강도는
quasi_static_wave_moment(전 크기 유효)로.

원전: references/IACS_UR_S11.pdf (확보 2026-08-09, 스펙 §3 인덱스).
"""
from __future__ import annotations

import math

import numpy as np

from src.physics.structure.loads import (
    G_ACC,
    RHO_SEAWATER,
    _cumtrapz,
    station_area,
    still_water_curves,
    weight_linear_density,
)

IACS_L_MIN = 90.0      # 원전 적용 하한 (Task 1 대조값으로 확정)
IACS_L_MAX = 500.0


class IACSRangeError(ValueError):
    """UR S11 적용 범위 밖 — 소형선은 표준파 준정적으로."""


def iacs_wave_coefficient(l_m: float) -> float:
    """파랑계수 Cw — UR S11 구간식 (원전 대조 박제)."""
    if not (IACS_L_MIN <= l_m <= IACS_L_MAX):
        raise IACSRangeError(
            f"L {l_m:.1f} m는 UR S11 범위({IACS_L_MIN:.0f}~"
            f"{IACS_L_MAX:.0f}) 밖 — quasi_static_wave_moment 사용.")
    if l_m <= 300.0:
        return 10.75 - ((300.0 - l_m) / 100.0) ** 1.5
    if l_m <= 350.0:
        return 10.75
    return 10.75 - ((l_m - 350.0) / 150.0) ** 1.5


def iacs_wave_bending_knm(l_m: float, b_m: float,
                          cb: float) -> tuple[float, float]:
    """설계 파랑 굽힘 (호깅 +, 새깅 −) [kN·m] — UR S11.

    cb 하한(원전 0.60) 미만은 0.60으로 클램프 (원전 지시)."""
    cw = iacs_wave_coefficient(l_m)
    cb_eff = max(cb, 0.60)
    hog = 0.19 * cw * l_m ** 2 * b_m * cb_eff
    sag = -0.11 * cw * l_m ** 2 * b_m * (cb_eff + 0.7)
    return hog, sag
```

- [ ] **Step 4: 통과 확인**

Run: `python -m pytest tests/test_wave_loads.py -v`
Expected: 3 PASS. Task 1 대조에서 상수가 다르게 확인됐다면 코드·시험 기대값을 원전 값으로 맞춘 뒤 통과시킬 것.

- [ ] **Step 5: 커밋**

```bash
git add src/physics/structure/wave_loads.py tests/test_wave_loads.py
git commit -m "feat: IACS UR S11 파랑 굽힘 정본 — Cw 구간식·호깅/새깅 (원전 대조)"
```

---

### Task 5: `wave_loads.py` — 표준파 준정적 굽힘 (전 크기 유효)

**Files:**
- Modify: `src/physics/structure/wave_loads.py`
- Test: `tests/test_wave_loads.py` (추가)

**Interfaces:**
- Consumes: `station_area`, `weight_linear_density`, `_cumtrapz`, `still_water_curves` (Task 3)
- Produces: `quasi_static_wave_moment(mesh, draft, blocks, wave_amp, wavelength, crest_mid=True, n=101) -> dict` — 키: `m_wave_mid_nm`(파랑 성분만, 호깅 +), `m_total_mid_nm`, `sinkage_m`, `buoy_scale`

- [ ] **Step 1: 실패 시험 작성 — 바지선 파랑 해석해**

`tests/test_wave_loads.py`에 추가:

```python
def _box_barge(loa=80.0, beam=10.0, depth=6.0):
    return trimesh.creation.box(extents=[loa, beam, depth])


def test_quasi_static_barge_analytic():
    """상자 바지선, λ=L 파정 중앙: |M_wave| = ρ·g·B·a·L²/(2π²).

    직벽 상자는 침하 보정 0 (cos 한 주기 적분 = 0) — 손계산 앵커."""
    from src.physics.structure.wave_loads import quasi_static_wave_moment
    loa, beam, t = 80.0, 10.0, 2.0
    mesh = _box_barge(loa, beam)
    mass = RHO * loa * beam * t
    wl_z = -3.0 + t
    amp = 0.5
    r = quasi_static_wave_moment(mesh, wl_z, [(mass, -40.0, 40.0)],
                                 wave_amp=amp, wavelength=loa, n=201)
    m_analytic = RHO * G * beam * amp * loa ** 2 / (2.0 * np.pi ** 2)
    assert r["m_wave_mid_nm"] == pytest.approx(m_analytic, rel=0.03)
    assert abs(r["sinkage_m"]) < 0.01 * amp


def test_quasi_static_hog_sag_mirror():
    """파정 중앙 = 호깅(+), 파곡 중앙 = 새깅(−) — 부호 거울."""
    from src.physics.structure.wave_loads import quasi_static_wave_moment
    mesh = _box_barge()
    mass = RHO * 80.0 * 10.0 * 2.0
    wl_z = -1.0
    hog = quasi_static_wave_moment(mesh, wl_z, [(mass, -40.0, 40.0)],
                                   wave_amp=0.5, wavelength=80.0,
                                   crest_mid=True)
    sag = quasi_static_wave_moment(mesh, wl_z, [(mass, -40.0, 40.0)],
                                   wave_amp=0.5, wavelength=80.0,
                                   crest_mid=False)
    assert hog["m_wave_mid_nm"] > 0 > sag["m_wave_mid_nm"]
    assert abs(hog["m_wave_mid_nm"]) == pytest.approx(
        abs(sag["m_wave_mid_nm"]), rel=0.05)
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_wave_loads.py -v`
Expected: 신규 2개 FAIL `ImportError`

- [ ] **Step 3: 구현**

`wave_loads.py`에 추가:

```python
def quasi_static_wave_moment(mesh, draft: float, blocks,
                             wave_amp: float, wavelength: float,
                             crest_mid: bool = True,
                             n: int = 101) -> dict:
    """표준파 준정적 굽힘 — 정현 파면에 배를 정적으로 얹고
    (침하 이분법 재평형, 트림 보정은 생략·기록) 부력 재적분.

    고전 표준파 계산법 (IACS 이전 세대 정통) — 전 크기 유효,
    IACS(대형 전용)·스트립(동적)과 3중 교차검증 축.
    반환 m_wave_mid_nm = 총 모멘트 − 정수 모멘트 (파랑 성분만)."""
    (xmin, _, _), (xmax, _, _) = mesh.bounds
    xs = np.linspace(xmin, xmax, n)
    xmid = 0.5 * (xmin + xmax)
    total_w = sum(m for m, _, _ in blocks) * G_ACC
    phase = 0.0 if crest_mid else math.pi
    wave = wave_amp * np.cos(
        2.0 * math.pi * (xs - xmid) / wavelength + phase)

    def buoy_curve(delta: float) -> np.ndarray:
        wl = draft + delta + wave
        areas = [station_area(mesh, x, float(z))
                 for x, z in zip(xs, wl)]
        return np.array(areas) * RHO_SEAWATER * G_ACC

    lo, hi = -abs(wave_amp) - 0.5, abs(wave_amp) + 0.5
    for _ in range(60):                     # 침하 이분법
        mid = 0.5 * (lo + hi)
        if float(np.trapezoid(buoy_curve(mid), xs)) < total_w:
            lo = mid
        else:
            hi = mid
    sinkage = 0.5 * (lo + hi)
    b = buoy_curve(sinkage)
    integ_b = float(np.trapezoid(b, xs))
    scale = total_w / integ_b if integ_b > 0 else 1.0
    b = b * scale

    w = weight_linear_density(xs, blocks)
    shear = _cumtrapz(w - b, xs)
    moment = _cumtrapz(shear, xs)
    ramp = (xs - xs[0]) / (xs[-1] - xs[0])
    moment = moment - moment[-1] * ramp

    still = still_water_curves(mesh, draft, blocks, n=n)
    i_mid = n // 2
    return {
        "m_total_mid_nm": float(moment[i_mid]),
        "m_wave_mid_nm": float(moment[i_mid]
                               - still.moment_nm[i_mid]),
        "sinkage_m": sinkage,
        "buoy_scale": scale,
        "note": "준정적 (동적 증폭 없음)·트림 재평형 생략 — "
                "스트립 동적과 교차검증",
    }
```

- [ ] **Step 4: 통과 확인**

Run: `python -m pytest tests/test_wave_loads.py -v`
Expected: 전부 PASS. 해석해 3% 안 맞으면 침하 이분법 수렴(60회)·격자 n 먼저 확인.

- [ ] **Step 5: 커밋**

```bash
git add src/physics/structure/wave_loads.py tests/test_wave_loads.py
git commit -m "feat: 표준파 준정적 굽힘 — 바지선 ρgBaL²/2π² 해석해 앵커"
```

---

### Task 6: 스트립 동적 굽힘 RAO + 3중 교차검증

**Files:**
- Modify: `src/physics/seakeeping/strip.py` (내부 헬퍼 추출 리팩터)
- Create: `src/physics/structure/strip_loads.py`
- Test: `tests/test_strip_loads.py`

**Interfaces:**
- Consumes: `heave_pitch_rao` 내부 로직 (리팩터로 노출), `weight_linear_density`, `_cumtrapz`
- Produces:
  - strip.py에 추가: `sectional_setup(mesh, draft, n_stations, contour_n) -> (xs, secs, yw)` (G 기준 xs), `sectional_coeffs(secs, we, contour_n) -> (m2d, n2d)`, `sectional_excitation(secs, xs, yw, k, we) -> np.ndarray[complex]`
  - `wave_bending_rao(mesh, draft, mass, iyy, blocks, omegas, n_stations=21, contour_n=12, restrained=False) -> list[dict]` — 키: `omega, m_mid_per_amp_nm`(ζa=1 기준), `balance_v, balance_m`(폐합 항등식 잔차비)

- [ ] **Step 1: strip.py 리팩터 — 기존 시험이 회귀 그물**

`heave_pitch_rao`의 ① 스테이션 준비부(스펙 §2 사슬 앞단)를 `sectional_setup`으로, ② 단면 계수 루프를 `sectional_coeffs`로, ③ 기진력 루프를 `sectional_excitation`으로 추출. `heave_pitch_rao`는 세 함수를 호출하는 조립부만 남긴다. **동작 불변** — 수치가 1비트도 달라질 이유 없는 순수 추출.

```python
# strip.py에 추가 (기존 코드 이동 — 시그니처만 제시)
def sectional_setup(mesh, draft, n_stations=11, contour_n=12):
    """스테이션 추출 + G 기준 x + 수선 반폭 — RAO·하중 공용."""
    from src.physics.seakeeping.sections import extract_stations
    stations = extract_stations(mesh, draft, n_stations=n_stations)
    if len(stations) < 5:
        raise ValueError("유효 스테이션 부족 — 흘수·메쉬 확인")
    xmid = 0.5 * (mesh.bounds[0][0] + mesh.bounds[1][0])
    xs = np.array([x - xmid for x, _ in stations])
    secs = [s for _, s in stations]
    yw = np.array([s.beam / 2.0 for s in secs])
    return xs, secs, yw


def sectional_coeffs(secs, we, contour_n=12):
    """단면 2D 부가질량·감쇠 (Frank) — 기존 루프 추출."""
    from src.physics.seakeeping.frank import heave_coefficients_frank
    m2d = np.zeros(len(secs))
    n2d = np.zeros(len(secs))
    for i, s in enumerate(secs):
        f = heave_coefficients_frank(_station_contour(s, contour_n),
                                     we, gauss_n=2)
        m2d[i], n2d[i] = f.added_mass, f.damping
    return m2d, n2d


def sectional_excitation(secs, xs, yw, k, we,
                         rho=RHO_SEAWATER):
    """스테이션별 기진력 밀도 x3 (FK 유효진폭 C3) — 기존 루프 추출."""
    from src.physics.seakeeping.lewis import section_points
    x3 = np.zeros(len(secs), dtype=complex)
    for i, s in enumerate(secs):
        pts = section_points(s, n=30)
        zs = np.array([p[1] - s.draft for p in pts])
        ys = np.array([p[0] for p in pts])
        integ = float(np.trapezoid(ys * np.exp(k * zs), zs))
        c3 = max(1.0 - (k / max(yw[i], 1e-9)) * abs(integ), 0.0)
        zeta = c3 * np.exp(1j * k * xs[i])
        acc = -k * G_ACC * zeta
        vel = 1j * (k * G_ACC / we) * zeta
        # 이 시점 m2d·n2d 필요 — 호출측에서 곱하도록 성분 분리 반환이
        # 깔끔하나 기존 수치 보존 우선: (acc, vel, zeta) 튜플 반환으로
        # 조립은 호출측 책임
        x3[i] = 0  # 자리 — 아래 조립 규약 참조
    return x3
```

**조립 규약 (실제 구현 지침):** 기존 `heave_pitch_rao` 루프에서 `x3[i] = m2d[i]*acc + n2d[i]*vel + 2ρg·yw[i]·zeta`가 m2d를 쓰므로, `sectional_excitation(secs, xs, yw, k, we, m2d, n2d)`로 **m2d·n2d를 인자로 받게** 시그니처 확정 (위 스케치의 자리 표시 제거). `heave_pitch_rao`는 `sectional_coeffs` 결과를 그대로 넘긴다.

Run: `python -m pytest tests/ -k "strip or seakeeping" -v`
Expected: 기존 내항 시험 전부 PASS (수치 불변 회귀 확인)

- [ ] **Step 2: 리팩터 커밋**

```bash
git add src/physics/seakeeping/strip.py
git commit -m "refactor: 스트립 내부 헬퍼 추출 — 하중 적분과 공용 (동작 불변)"
```

- [ ] **Step 3: 실패 시험 작성**

```python
# tests/test_strip_loads.py
"""스트립 동적 굽힘 RAO — 폐합 항등식 + 준정적 교차 (스펙 §2·§3)."""
import numpy as np
import pytest
import trimesh

G = 9.81
RHO = 1025.0


def _barge_setup(loa=80.0, beam=10.0, t=2.0):
    mesh = trimesh.creation.box(extents=[loa, beam, 6.0])
    mass = RHO * loa * beam * t
    iyy = mass * (0.25 * loa) ** 2
    wl_z = -3.0 + t
    blocks = [(mass, -loa / 2.0, loa / 2.0)]
    return mesh, mass, iyy, wl_z, blocks


def test_bending_balance_identity():
    """자유 운동 해에서 V·M 양끝 잔차 미소 — 운동방정식 총평형의
    내부 하중 버전 (상시 자기검증)."""
    from src.physics.structure.strip_loads import wave_bending_rao
    mesh, mass, iyy, wl_z, blocks = _barge_setup()
    omega = float(np.sqrt(2.0 * np.pi * G / 80.0))    # λ = L
    out = wave_bending_rao(mesh, wl_z, mass, iyy, blocks, [omega])
    assert out[0]["balance_v"] < 0.05
    assert out[0]["balance_m"] < 0.05


def test_restrained_matches_quasi_static():
    """구속 모드(운동 0) 스트립 굽힘 ≈ 준정적 표준파 — 서로 다른
    두 적분(FK 압력 vs 메쉬 재침수)의 λ=L 교차검증."""
    from src.physics.structure.strip_loads import wave_bending_rao
    from src.physics.structure.wave_loads import quasi_static_wave_moment
    mesh, mass, iyy, wl_z, blocks = _barge_setup()
    amp = 0.5
    omega = float(np.sqrt(2.0 * np.pi * G / 80.0))
    strip = wave_bending_rao(mesh, wl_z, mass, iyy, blocks, [omega],
                             restrained=True)
    qs = quasi_static_wave_moment(mesh, wl_z, blocks, wave_amp=amp,
                                  wavelength=80.0, n=201)
    m_strip = strip[0]["m_mid_per_amp_nm"] * amp
    assert m_strip == pytest.approx(abs(qs["m_wave_mid_nm"]), rel=0.20)


def test_free_less_than_restrained():
    """자유 운동은 파면을 타서 하중 경감 — |M_free| < |M_restrained|
    (장파 방향 성질)."""
    from src.physics.structure.strip_loads import wave_bending_rao
    mesh, mass, iyy, wl_z, blocks = _barge_setup()
    omega = float(np.sqrt(2.0 * np.pi * G / 160.0))   # λ = 2L 장파측
    free = wave_bending_rao(mesh, wl_z, mass, iyy, blocks, [omega])
    rest = wave_bending_rao(mesh, wl_z, mass, iyy, blocks, [omega],
                            restrained=True)
    assert free[0]["m_mid_per_amp_nm"] < rest[0]["m_mid_per_amp_nm"]
```

- [ ] **Step 4: 실패 확인**

Run: `python -m pytest tests/test_strip_loads.py -v`
Expected: FAIL `ModuleNotFoundError: strip_loads`

- [ ] **Step 5: 구현**

```python
# src/physics/structure/strip_loads.py
"""스트립 동적 굽힘 모멘트 RAO (구조 강도 1단계 — 3중 교차검증 축).

원리: 자유 운동 해 (ẑ, θ̂)에서 스테이션별 내부 수직 하중 밀도
q(x) = 관성 − 유체력 = μ(x)·s̈(x) − [기진 − (부가질량·s̈ + 감쇠·ṡ
+ 복원·s)] 를 두 번 적분 → V(x)·M(x) 복소 진폭.

앵커: ① 폐합 항등식 (양끝 V·M ≈ 0 — 운동방정식 총평형의 내부
버전) ② 구속 모드 = 준정적 표준파 교차 (λ=L) ③ 자유 < 구속
(파면 타기 하중 경감).
부호: |M| 진폭만 반환 (위상별 호깅/새깅은 IACS·준정적 축이 판정).
"""
from __future__ import annotations

import numpy as np

from src.physics.seakeeping.strip import (
    G_ACC,
    RHO_SEAWATER,
    sectional_coeffs,
    sectional_excitation,
    sectional_setup,
)
from src.physics.structure.loads import _cumtrapz, weight_linear_density


def wave_bending_rao(mesh, draft: float, mass: float, iyy: float,
                     blocks, omegas, n_stations: int = 21,
                     contour_n: int = 12,
                     restrained: bool = False) -> list[dict]:
    """규칙파 ζa=1 기준 미드십 굽힘 모멘트 진폭 [N·m/m]."""
    xs, secs, yw = sectional_setup(mesh, draft,
                                   n_stations=n_stations,
                                   contour_n=contour_n)
    xmid_g = 0.5 * (mesh.bounds[0][0] + mesh.bounds[1][0])
    # 중량 밀도 μ(x) [kg/m] — 블록을 G 기준 좌표로 이동, 질량 폐합
    blocks_g = [(m, x0 - xmid_g, x1 - xmid_g) for m, x0, x1 in blocks]
    mu = weight_linear_density(xs, blocks_g) / G_ACC
    integ_mu = float(np.trapezoid(mu, xs))
    if integ_mu > 0:
        mu *= mass / integ_mu

    out = []
    for omega in omegas:
        k = omega * omega / G_ACC
        we = omega                                  # V=0
        m2d, n2d = sectional_coeffs(secs, we, contour_n=contour_n)
        x3 = sectional_excitation(secs, xs, yw, k, we, m2d, n2d)

        if restrained:
            s = np.zeros(len(xs), dtype=complex)
        else:
            # 자유 운동 해 — 기존 RAO 조립과 동일 계수 (V=0 항)
            def tr(arr):
                return float(np.trapezoid(arr, xs))
            a33, b33 = tr(m2d), tr(n2d)
            c33 = 2.0 * RHO_SEAWATER * G_ACC * tr(yw)
            a35 = -tr(m2d * xs)
            b35 = -tr(n2d * xs)
            c35 = -2.0 * RHO_SEAWATER * G_ACC * tr(yw * xs)
            a55 = tr(m2d * xs ** 2)
            b55 = tr(n2d * xs ** 2)
            c55 = 2.0 * RHO_SEAWATER * G_ACC * tr(yw * xs ** 2)
            xw3 = complex(np.trapezoid(x3, xs))
            xw5 = complex(np.trapezoid(-x3 * xs, xs))
            lhs = np.array([
                [-we ** 2 * (mass + a33) + 1j * we * b33 + c33,
                 -we ** 2 * a35 + 1j * we * b35 + c35],
                [-we ** 2 * a35 + 1j * we * b35 + c35,
                 -we ** 2 * (iyy + a55) + 1j * we * b55 + c55]])
            z_hat, th_hat = np.linalg.solve(lhs, np.array([xw3, xw5]))
            s = z_hat - xs * th_hat

        acc = -we ** 2 * s
        vel = 1j * we * s
        f_hydro = x3 - (m2d * acc + n2d * vel
                        + 2.0 * RHO_SEAWATER * G_ACC * yw * s)
        q = mu * acc - f_hydro
        shear = _cumtrapz(q, xs)
        moment = _cumtrapz(shear, xs)
        bal_v = float(abs(shear[-1])
                      / max(np.max(np.abs(shear)), 1e-12))
        bal_m = float(abs(moment[-1])
                      / max(np.max(np.abs(moment)), 1e-12))
        ramp = (xs - xs[0]) / (xs[-1] - xs[0])
        moment = moment - moment[-1] * ramp
        m_mid = abs(complex(np.interp(0.0, xs, moment.real))
                    + 1j * complex(np.interp(0.0, xs, moment.imag)))
        out.append({"omega": float(omega),
                    "m_mid_per_amp_nm": float(m_mid),
                    "balance_v": bal_v, "balance_m": bal_m})
    return out
```

구현 주의 (사고 다발 지점 — Frank 관례):
- `sectional_excitation`의 x3는 리팩터에서 m2d·n2d를 인자로 받도록 확정한 시그니처와 일치시킬 것
- 자유 해 연성 행렬은 V=0이라 a35=a53·b35=b53 대칭 — strip.py 원식과 동일해야 함 (다르면 리팩터 회귀 시험이 먼저 잡음)
- 폐합 잔차 5% 초과 시: 스테이션 수(n_stations)·격자 우선 의심, 그다음 μ 정규화

- [ ] **Step 6: 통과 확인**

Run: `python -m pytest tests/test_strip_loads.py -v`
Expected: 3 PASS

- [ ] **Step 7: 3중 교차검증 e2e (느린 시험 — 별도 파일 유지)**

`tests/test_strip_loads.py`에 추가:

```python
def test_three_way_cross_check_100m_cargo(tmp_path):
    """100m 화물선 — IACS vs 준정적 vs 스트립 같은 자릿수 (성적표).

    표준파 a = L/40 (H = L/20 고전 관례). 극치 통계 유무가 달라
    수배 차이는 정상 — 자릿수(0.2~5배) 합의만 판정, 값은 성적표
    기록."""
    from src.core.types import GoalSpec
    from src.pipeline import run_pipeline
    from src.physics.structure.loads import still_water_curves
    from src.physics.structure.strip_loads import wave_bending_rao
    from src.physics.structure.wave_loads import (
        iacs_wave_bending_knm, quasi_static_wave_moment)
    import trimesh

    goal = GoalSpec(target_speed_ms=7.0, payload_kg=5_000_000.0,
                    purpose="cargo")
    report = run_pipeline(goal, tmp_path, seakeeping=False)
    mesh = trimesh.load(tmp_path / report["mesh_file"])
    d = report["dimensions"]
    loa, beam = d["loa"], d["beam"]
    draft = report["hydrostatics"]["draft"]
    zmin = float(mesh.bounds[0][2])
    wl_z = zmin + draft
    lt = report["weights_large"]["lightship_t"]
    comp = {"structure": lt["structure"] * 1e3,
            "outfit": lt["outfit"] * 1e3,
            "machinery": lt["machinery"] * 1e3,
            "fuel": report["weights_large"]["fuel_t"] * 1e3,
            "payload": report["weights_large"]["payload_t"] * 1e3}
    from src.physics.structure.loads import standard_weight_blocks
    xmin = float(mesh.bounds[0][0])
    blocks = standard_weight_blocks(comp, xmin, loa)

    amp = loa / 40.0
    hog, sag = iacs_wave_bending_knm(loa, beam, d["cb"])
    qs = quasi_static_wave_moment(mesh, wl_z, blocks, wave_amp=amp,
                                  wavelength=loa)
    mass = sum(comp.values())
    iyy = mass * (0.25 * loa) ** 2
    omega = float(np.sqrt(2.0 * np.pi * G / loa))
    st = wave_bending_rao(mesh, wl_z, mass, iyy, blocks, [omega],
                          restrained=True)
    m_strip = st[0]["m_mid_per_amp_nm"] * amp

    m_iacs = hog * 1e3                       # kN·m → N·m
    m_qs = abs(qs["m_wave_mid_nm"])
    print(f"\n[3중 교차] IACS {m_iacs:.3e} / 준정적 {m_qs:.3e} "
          f"/ 스트립(구속) {m_strip:.3e} N·m")
    assert 0.2 < m_qs / m_iacs < 5.0
    assert 0.2 < m_strip / m_iacs < 5.0
```

주의: `report["weights_large"]` 키 이름은 실제 리포트 구조 확인 후 맞출 것 (`src/pipeline.py`의 대형 분기 반환 — 다르면 시험 쪽을 수정, 파이프라인 개조 금지). `d["cb"]` 키도 동일.

Run: `python -m pytest tests/test_strip_loads.py -v`
Expected: 전부 PASS + 3중 교차 수치 출력

- [ ] **Step 8: 커밋**

```bash
git add src/physics/structure/strip_loads.py tests/test_strip_loads.py
git commit -m "feat: 스트립 동적 굽힘 RAO — 3중 교차검증 (IACS·준정적·스트립)"
```

---

### Task 7: 1단계 성적표 + 회귀 확인

**Files:**
- Modify: `docs/worklog/2026-08-09.md` (성적표 기록)

**Interfaces:**
- Consumes: Task 2~6 전체

- [ ] **Step 1: 전체 회귀**

Run: `python -m pytest tests/ -x -q --ignore=tests/test_strip_loads.py` 후 `python -m pytest tests/test_strip_loads.py -q`
Expected: 기존 350 + 신규 전부 PASS (느린 e2e 분할 실행 — 슈트 관례)

- [ ] **Step 2: 성적표 작성**

worklog에 1단계 성적표 기록:
- 바지선 해석해 2종 (WL/16·ρgBaL²/2π²) 오차
- 폐합 항등식 잔차 (V·M)
- 3중 교차검증 표 (IACS/준정적/스트립 — 100m 화물선)
- 다음 단계 (단면 설계) 진입 판단 재료

- [ ] **Step 3: 커밋·푸시**

```bash
git add docs/worklog/
git commit -m "docs: 구조 1단계 성적표 — 하중 곡선 3중 교차검증"
git push
```

성적표를 오너에게 보고하고 계속/보류 판단을 받는다 (단계 분할 관례).
