# 구조 강도 2단계 — 단면 설계 (재료·판 두께·미드십) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 재료 3종 물성 → DNV 계보 설계 압력·판 두께·종늑골 배치 → 미드십 단면계수 조립. 1단계 하중과 만나 3단계 판정의 "실제값" 축을 완성한다.

**Architecture:** `src/physics/structure/`에 `materials.py` → `scantlings.py` → `midship.py` 순 적층. 원전 = DNV Rules for Ships Pt.3 Ch.1(≥100m)/Ch.2(<100m) (references/ 확보됨, 스펙 §3 보강 2 인덱스). 앵커: 단순 상자+보강재 단면계수 손계산, 실선 두께 대역.

**Tech Stack:** Python 3, numpy, pytest. 외부 의존 없음.

## Global Constraints

- 원전 상수는 구현 직전 PDF 재판독으로 확정 (스펙 §3 보강 2 인덱스 페이지) — 계획 수치와 다르면 원전이 정답
- 알루미늄 = 강 규칙식 + 재료 계수 환산 C급 표기, FRP = 공개 적층 물성 C급 표기 (스펙 명시)
- 한국어 docstring, 기존 시험 전체 통과 유지, main 직커밋
- 단위 관례: 두께 mm, 압력 kN/m², 단면계수 cm³(보강재)·m³(선체 거더), 응력 N/mm² — DNV 원전 단위 그대로 (변환 사고 방지)

---

### Task 1: `materials.py` — 재료 3종 물성

**Files:**
- Create: `src/physics/structure/materials.py`
- Test: `tests/test_structure_materials.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Material: name, yield_nmm2, f1, density_kgm3, e_nmm2, grade, note`
  - `MATERIALS: dict[str, Material]` — 키 "mild_steel", "ah36", "al5083", "frp_eglass"
  - `select_material(loa: float, purpose: str) -> Material` — 대형(≥60m)=연강, 소형=알루 기본, patrol 고속 소형=알루 (FRP는 명시 지정 시)

- [ ] **Step 1: 실패 시험 작성**

```python
# tests/test_structure_materials.py
"""재료 3종 물성 — f1 계수·용접부 함정 (스펙 2026-08-09 §2)."""
import pytest


def test_material_catalog_values():
    """물성 앵커: 연강 235/f1=1.0 (UR S11 k=1 정합), AH36 355,
    알루 5083 용접부 강도 < 모재 (함정 지점 — 용접하면 항복 저하)."""
    from src.physics.structure.materials import MATERIALS
    ms = MATERIALS["mild_steel"]
    assert ms.yield_nmm2 == pytest.approx(235.0)
    assert ms.f1 == pytest.approx(1.0)
    ah = MATERIALS["ah36"]
    assert ah.yield_nmm2 == pytest.approx(355.0)
    assert ah.f1 > 1.0                     # 고장력 = 얇게 허용
    al = MATERIALS["al5083"]
    assert al.yield_nmm2 < 200.0           # 용접부 기준 (모재 아님)
    assert "용접" in al.note
    frp = MATERIALS["frp_eglass"]
    assert frp.grade == "C"                # 원전 미확보 정직 표기


def test_select_material_by_size():
    """대형 = 강, 소형 = 알루 (실선 관행)."""
    from src.physics.structure.materials import select_material
    assert select_material(100.0, "cargo").name == "mild_steel"
    assert select_material(3.0, "survey").name == "al5083"
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_structure_materials.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 구현**

```python
# src/physics/structure/materials.py
"""구조 재료 3종 물성 (구조 강도 2단계, 스펙 §2).

f1 = DNV 재료 계수 (Pt.3 Ch.1 Sec.2 — 연강 1.0 기준, 고장력강은
허용 응력 상향 → 판 얇게). 알루미늄·FRP는 원전 규칙 미확보 —
등가 f1 = σy/235 환산 C급 (스펙 §3 보강 2 정직 표기).

함정 지점: 알루 5083은 **용접부(HAZ) 강도**가 설계 기준 — 모재
항복(215~228)이 아니라 용접 후 ~125 N/mm² (5083-H116 용접부 통상
대역). 강은 용접해도 모재 강도 유지 — 알루만의 함정.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    yield_nmm2: float      # 설계 기준 항복 (알루 = 용접부)
    f1: float              # DNV 재료 계수 (연강 1.0)
    density_kgm3: float
    e_nmm2: float          # 탄성계수
    grade: str             # 출처 등급 A/B/C
    note: str


MATERIALS: dict[str, Material] = {
    "mild_steel": Material(
        "mild_steel", 235.0, 1.0, 7850.0, 2.06e5, "A",
        "DNV NS 연강 — UR S11 k=1 정합"),
    "ah36": Material(
        "ah36", 355.0, 1.39, 7850.0, 2.06e5, "A",
        "DNV NV-36 고장력강 — f1=1.39 (Pt.3 Ch.1 Sec.2, 구현 시 "
        "원전 재판독 확정)"),
    "al5083": Material(
        "al5083", 125.0, 125.0 / 235.0, 2660.0, 0.69e5, "C",
        "알루 5083-H116 용접부(HAZ) 기준 — HSLC 원전 미확보, "
        "등가 f1 환산 C급"),
    "frp_eglass": Material(
        "frp_eglass", 100.0, 100.0 / 235.0, 1700.0, 0.15e5, "C",
        "E-glass 적층 설계 허용 대역 (극한 ~200/안전율 2) — "
        "ISO 12215-5 미확보 C급"),
}


def select_material(loa: float, purpose: str) -> Material:
    """크기·용도 → 기본 재료 (사용자 재지정 가능).

    실선 관행: 대형 상선 = 강, 소형 USV/작업정 = 알루미늄."""
    if loa >= 60.0:
        return MATERIALS["mild_steel"]
    return MATERIALS["al5083"]
```

- [ ] **Step 4: f1 원전 확정**

`references/DNV_Pt3Ch1_hull.pdf` Sec.2 (p18 근방) 재판독 — NV-32/NV-36 f1 값 확인 (예상 1.28/1.39). 다르면 코드·시험 갱신.

- [ ] **Step 5: 통과 확인 + 커밋**

Run: `python -m pytest tests/test_structure_materials.py -v`
Expected: 2 PASS

```bash
git add src/physics/structure/materials.py tests/test_structure_materials.py
git commit -m "feat: 구조 재료 3종 — f1 계수·알루 용접부 함정 박제"
```

---

### Task 2: `scantlings.py` — 설계 압력 + 판 두께 + 종늑골

**Files:**
- Create: `src/physics/structure/scantlings.py`
- Test: `tests/test_structure_scantlings.py`

**Interfaces:**
- Consumes: `Material` (Task 1)
- Produces:
  - `design_pressure_bottom(loa, beam, draft, cw=None) -> float` [kN/m²] — DNV p1 = 10T + pdp(z=0) 계보 (CW = UR S11 파랑계수 재사용, L<90은 소형 보간)
  - `design_pressure_side(loa, beam, draft, depth) -> float` — 수선 상부 완화
  - `design_pressure_deck(loa) -> float` — 최소 5+0.025L1 계보
  - `plate_thickness_mm(pressure_knm2, spacing_m, span_m, sigma_nmm2, tk_mm=1.5) -> float` — **t = 15.8·ka·s·√(p/σ) + tk** (원전 p89)
  - `min_thickness_mm(loa, f1, location) -> float` — t0 + k·L1/f1 계보 (p89~90)
  - `stiffener_modulus_cm3(pressure_knm2, spacing_m, span_m, f1) -> float` — Z = 0.63·l²·s·p·wk/f1 계보 (p91, wk=1)
  - `default_spacing_m(loa) -> float` — 종늑골 간격 통상 (0.6~0.9m 대형, 소형 비례 축소)

- [ ] **Step 1: 실패 시험 작성**

```python
# tests/test_structure_scantlings.py
"""DNV 계보 국부 스캔틀링 — 실선 두께 대역 앵커 (스펙 §3)."""
import pytest


def test_plate_thickness_formula_anchor():
    """원전 식 손계산: p=100 kN/m², s=0.7m, σ=120, ka=1 (s/l 0.4↓)
    → t = 15.8·0.7·√(100/120) + 1.5 = 11.60 mm."""
    from src.physics.structure.scantlings import plate_thickness_mm
    t = plate_thickness_mm(100.0, 0.7, 2.5, 120.0, tk_mm=1.5)
    assert t == pytest.approx(15.8 * 0.7 * (100.0 / 120.0) ** 0.5
                              + 1.5, rel=1e-6)


def test_plate_thickness_aspect_correction():
    """정사각 패널(s/l=1) ka=0.72 하한 — 좁고 긴 패널보다 얇게 허용."""
    from src.physics.structure.scantlings import plate_thickness_mm
    t_long = plate_thickness_mm(100.0, 0.7, 2.5, 120.0)   # s/l 0.28
    t_sq = plate_thickness_mm(100.0, 0.7, 0.7, 120.0)     # s/l 1.0
    assert t_sq < t_long


def test_100m_cargo_bottom_band():
    """실선 sanity: 100m 화물선 선저 10~15mm 대역 (문헌 통상)."""
    from src.physics.structure.materials import MATERIALS
    from src.physics.structure.scantlings import (
        design_pressure_bottom, default_spacing_m, min_thickness_mm,
        plate_thickness_mm)
    p = design_pressure_bottom(100.0, 15.0, 5.5)
    s = default_spacing_m(100.0)
    t = max(plate_thickness_mm(p, s, 2.5, 120.0),
            min_thickness_mm(100.0, 1.0, "bottom"))
    assert 8.0 < t < 16.0


def test_small_alu_band():
    """소형 알루 3~8mm 대역 (실선 USV 관행) — 전 크기 유효 확인."""
    from src.physics.structure.materials import MATERIALS
    from src.physics.structure.scantlings import (
        design_pressure_bottom, default_spacing_m, min_thickness_mm,
        plate_thickness_mm)
    al = MATERIALS["al5083"]
    p = design_pressure_bottom(3.0, 1.2, 0.3)
    s = default_spacing_m(3.0)
    t = max(plate_thickness_mm(p, s, 0.5, 120.0 * al.f1, tk_mm=0.5),
            min_thickness_mm(3.0, al.f1, "bottom"))
    assert 2.0 < t < 9.0


def test_stiffener_modulus_positive_scaling():
    """늑골 단면계수 — 스팬 제곱·압력 비례 (원전 구조)."""
    from src.physics.structure.scantlings import stiffener_modulus_cm3
    z1 = stiffener_modulus_cm3(100.0, 0.7, 2.0, 1.0)
    z2 = stiffener_modulus_cm3(100.0, 0.7, 4.0, 1.0)
    assert z2 == pytest.approx(4.0 * z1, rel=1e-6)
    assert z1 > 0
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_structure_scantlings.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 원전 재판독 — 압력·최소 두께·종늑골 Z 정확식**

`references/DNV_Pt3Ch1_hull.pdf` p56(Sec.4 C200 해수 압력)·p89~91(Sec.6) + 종늑골 Z식 (Sec.6 C700 근방 — 페이지 탐색) 판독. 소형(<100m)은 Ch.2 대응식 확인. 아래 Step 4 코드의 상수를 원전 값으로 확정.

- [ ] **Step 4: 구현**

```python
# src/physics/structure/scantlings.py
"""DNV 계보 국부 스캔틀링 (구조 강도 2단계, 스펙 §2·§3 보강 2).

원전: DNV Rules for Ships Pt.3 Ch.1 (2016, ≥100m) — 판 두께
t = 15.8·ka·s·√(p/σ) + tk (p89), 최소 두께 t0 + k·L1/f1 (p89~90),
늑골 Z = 0.63·l²·s·p·wk/f1 (p91). 소형은 Ch.2 대응식 계보 —
100m 미만 외삽은 최소 두께가 지배 (정직 표기).

설계 압력: p1 = 10·h0 + pdp (수선 아래, Sec.4 C201) — pdp의
파랑 성분은 UR S11 CW 재사용 (같은 계보), ks=2 미드십.
"""
from __future__ import annotations

import math


def _cw_small_ok(loa: float) -> float:
    """파랑계수 CW — UR S11 구간식 (L≥90), 소형은 0.0792L 연속
    외삽 (DNV Ch.2 소형 계보, C급)."""
    if loa >= 90.0:
        from src.physics.structure.wave_loads import iacs_wave_coefficient
        return iacs_wave_coefficient(loa)
    return 0.0792 * loa


def design_pressure_bottom(loa: float, beam: float,
                           draft: float) -> float:
    """선저 설계 압력 [kN/m²] — p1 = 10·T + pdp(z=0).

    pdp = ks·CW − 1.2(T−z), z=0(선저), ks=2 (미드십 0.2~0.7L).
    kf 항은 보수 생략 (작음)."""
    cw = _cw_small_ok(loa)
    pdp = 2.0 * cw - 1.2 * draft
    return 10.0 * draft + max(pdp, 0.0)


def design_pressure_side(loa: float, beam: float, draft: float,
                         depth: float) -> float:
    """선측 설계 압력 [kN/m²] — 수선에서 z=T 평가 (p1의 z=T) +
    상부 최소 6.25+0.025L 계보."""
    cw = _cw_small_ok(loa)
    p_wl = 2.0 * cw                       # z=T: 정수항 0 + pdp
    return max(p_wl, 6.25 + 0.025 * min(loa, 300.0))


def design_pressure_deck(loa: float) -> float:
    """노천 갑판 최소 압력 [kN/m²] — 원전 'minimum 5' 계보."""
    return max(5.0, 6.25 + 0.025 * min(loa, 300.0) - 5.0)


def _ka(spacing_m: float, span_m: float) -> float:
    """종횡비 보정 ka = (1.1 − 0.25 s/l)², 0.72~1.0 클램프 (원전)."""
    r = spacing_m / max(span_m, 1e-9)
    ka = (1.1 - 0.25 * r) ** 2
    return min(1.0, max(0.72, ka))


def plate_thickness_mm(pressure_knm2: float, spacing_m: float,
                       span_m: float, sigma_nmm2: float,
                       tk_mm: float = 1.5) -> float:
    """판 두께 [mm] — t = 15.8·ka·s·√(p/σ) + tk (원전 p89)."""
    return (15.8 * _ka(spacing_m, span_m) * spacing_m
            * math.sqrt(pressure_knm2 / sigma_nmm2) + tk_mm)


def min_thickness_mm(loa: float, f1: float, location: str) -> float:
    """최소 두께 [mm] — t0 + k·L1/f1 계보 (원전 p89~90).

    location: 'keel' 7.0+0.05L1 / 'bottom' 5.0+0.04L1 /
    'side'·'deck' 5.0+0.03L1. L1 = min(L, 300). 소형은 상수항이
    지배 — 실선 알루 소형정 4~6mm 대역과 정합."""
    l1 = min(loa, 300.0)
    base = {"keel": (7.0, 0.05), "bottom": (5.0, 0.04),
            "side": (5.0, 0.03), "deck": (5.0, 0.03)}[location]
    t0, k = base
    # 소형 보정: L<20m는 규칙 하한이 과대 — 비례 완화 (C급 정직)
    if loa < 20.0:
        t0 = t0 * max(loa / 20.0, 0.4)
    return t0 + k * l1 / max(f1, 1e-9)


def stiffener_modulus_cm3(pressure_knm2: float, spacing_m: float,
                          span_m: float, f1: float,
                          wk: float = 1.0) -> float:
    """늑골(보강재) 요구 단면계수 [cm³] — Z = 0.63·l²·s·p·wk/f1
    (원전 p91 계열)."""
    return (0.63 * span_m ** 2 * spacing_m * pressure_knm2 * wk
            / max(f1, 1e-9))


def default_spacing_m(loa: float) -> float:
    """종늑골 간격 통상값 — 대형 0.7m 근방, 소형 비례 축소
    (2+ L/250 계보 관행, C급)."""
    return min(0.9, max(0.2, 0.48 + loa / 400.0))
```

- [ ] **Step 5: 통과 확인 + 커밋**

Run: `python -m pytest tests/test_structure_scantlings.py -v`
Expected: 5 PASS. 대역 시험 실패 시 압력식 kf 생략·소형 보정 계수부터 재검토 (원전 재판독).

```bash
git add src/physics/structure/scantlings.py tests/test_structure_scantlings.py
git commit -m "feat: DNV 계보 스캔틀링 — 판 두께 15.8ka·s·√(p/σ)·실선 대역 앵커"
```

---

### Task 3: `midship.py` — 미드십 단면계수 조립

**Files:**
- Create: `src/physics/structure/midship.py`
- Test: `tests/test_structure_midship.py`

**Interfaces:**
- Consumes: `Material`, `plate_thickness_mm` 계열 (Task 1·2)
- Produces:
  - `@dataclass PlateElement: name, width_m, thickness_mm, z_center_m` (수평 판) / 수직 판은 height_m 사용 — `area_m2`, `i_own_m4` 속성
  - `assemble_midship(beam, depth, t_bottom_mm, t_side_mm, t_deck_mm, n_bottom_long, n_deck_long, long_area_cm2) -> MidshipSection`
  - `MidshipSection: neutral_axis_m, inertia_m4, z_deck_m3, z_keel_m3, area_m2` — 단면계수 [m³]

- [ ] **Step 1: 실패 시험 작성 — 손계산 앵커**

```python
# tests/test_structure_midship.py
"""미드십 단면계수 조립 — 상자 손계산 앵커 (스펙 §3)."""
import pytest


def test_symmetric_box_hand_calc():
    """대칭 상자 (선저=갑판 t, 선측 t): 중립축 = D/2 정확,
    I 손계산 대조.

    B=10, D=6, t=10mm 전둘레: 선저·갑판 A=0.1 m² 각각 z=0,6 →
    I_판 = 2·0.1·3² = 1.8, 선측 2장 I = 2·(0.01·6³/12) = 0.36,
    합 2.16 m⁴. Z = I/3 = 0.72 m³."""
    from src.physics.structure.midship import assemble_midship
    sec = assemble_midship(10.0, 6.0, 10.0, 10.0, 10.0,
                           n_bottom_long=0, n_deck_long=0,
                           long_area_cm2=0.0)
    assert sec.neutral_axis_m == pytest.approx(3.0, rel=1e-6)
    assert sec.inertia_m4 == pytest.approx(2.16, rel=0.01)
    assert sec.z_deck_m3 == pytest.approx(0.72, rel=0.01)
    assert sec.z_keel_m3 == pytest.approx(0.72, rel=0.01)


def test_asymmetric_thickness_shifts_neutral_axis():
    """선저를 두껍게 → 중립축 하강, Z_deck < Z_keel 역전 방향."""
    from src.physics.structure.midship import assemble_midship
    sec = assemble_midship(10.0, 6.0, 20.0, 10.0, 10.0,
                           n_bottom_long=0, n_deck_long=0,
                           long_area_cm2=0.0)
    assert sec.neutral_axis_m < 3.0
    assert sec.z_deck_m3 < sec.z_keel_m3


def test_longitudinals_add_inertia():
    """종늑골 추가 → I 증가 (부재는 공짜가 아니다)."""
    from src.physics.structure.midship import assemble_midship
    bare = assemble_midship(10.0, 6.0, 10.0, 10.0, 10.0, 0, 0, 0.0)
    stiff = assemble_midship(10.0, 6.0, 10.0, 10.0, 10.0,
                             n_bottom_long=10, n_deck_long=10,
                             long_area_cm2=20.0)
    assert stiff.inertia_m4 > bare.inertia_m4
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_structure_midship.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 구현**

```python
# src/physics/structure/midship.py
"""미드십 단면계수 조립 (구조 강도 2단계, 스펙 §2).

모델: 상자보 — 선저판(z=0)·갑판(z=D)·선측 2장(수직) + 종늑골
(면적 점부재, 판 근방 배치). 평행축 정리로 중립축·I → 단면계수
Z_deck = I/(D−zn), Z_keel = I/zn.

한계 정직 표기: 이중저·해치 개구·전단 지연 없음 — 개구 큰 배
(컨테이너선)는 과대평가. 상선 폐단면 근사."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MidshipSection:
    neutral_axis_m: float    # 킬 기준 중립축 높이
    inertia_m4: float
    z_deck_m3: float
    z_keel_m3: float
    area_m2: float


def assemble_midship(beam: float, depth: float,
                     t_bottom_mm: float, t_side_mm: float,
                     t_deck_mm: float,
                     n_bottom_long: int, n_deck_long: int,
                     long_area_cm2: float) -> MidshipSection:
    """판 3종 + 종늑골 → 중립축·I·단면계수.

    종늑골: 선저군 z=0.05D, 갑판군 z=0.95D 점면적 (부착판 근방)."""
    tb, ts, td = (t_bottom_mm * 1e-3, t_side_mm * 1e-3,
                  t_deck_mm * 1e-3)
    a_long = long_area_cm2 * 1e-4
    # (면적, z중심, 자체 I)
    elems = [
        (beam * tb, 0.0, beam * tb ** 3 / 12.0),          # 선저
        (beam * td, depth, beam * td ** 3 / 12.0),        # 갑판
        (2.0 * ts * depth, depth / 2.0,
         2.0 * ts * depth ** 3 / 12.0),                   # 선측 2장
        (n_bottom_long * a_long, 0.05 * depth, 0.0),      # 선저 종늑골
        (n_deck_long * a_long, 0.95 * depth, 0.0),        # 갑판 종늑골
    ]
    area = sum(a for a, _, _ in elems)
    zn = sum(a * z for a, z, _ in elems) / max(area, 1e-12)
    inertia = sum(i0 + a * (z - zn) ** 2 for a, z, i0 in elems)
    return MidshipSection(
        neutral_axis_m=zn, inertia_m4=inertia,
        z_deck_m3=inertia / max(depth - zn, 1e-9),
        z_keel_m3=inertia / max(zn, 1e-9),
        area_m2=area)
```

- [ ] **Step 4: 통과 확인 + 커밋**

Run: `python -m pytest tests/test_structure_midship.py -v`
Expected: 3 PASS

```bash
git add src/physics/structure/midship.py tests/test_structure_midship.py
git commit -m "feat: 미드십 단면계수 조립 — 상자+종늑골 손계산 앵커"
```

---

### Task 4: 2단계 성적표 — 실선 대역 통합 확인

**Files:**
- Modify: `docs/worklog/2026-08-09.md` (2단계 성적표 추가)
- Test: `tests/test_structure_scantlings.py` (통합 시험 1개 추가)

- [ ] **Step 1: 통합 시험 — 100m 설계 두께로 단면 조립, UR S11 요구 단면계수와 비교**

```python
def test_100m_section_modulus_vs_iacs_requirement():
    """2단계 통합: 규칙 두께로 조립한 단면계수 vs UR S11 요구치
    Z_req = M_total/175 — 같은 자릿수 (3단계 판정 예고편).

    UR S11 요구 최소 단면계수 계보: Z_min = C·L²·B·(Cb+0.7) cm²·m
    ×10⁻⁶ [m³] 병기 확인."""
    from src.physics.structure.materials import MATERIALS
    from src.physics.structure.midship import assemble_midship
    from src.physics.structure.scantlings import (
        design_pressure_bottom, design_pressure_side, default_spacing_m,
        min_thickness_mm, plate_thickness_mm)
    from src.physics.structure.wave_loads import iacs_wave_bending_knm

    loa, beam, depth, draft, cb = 100.0, 15.0, 8.0, 5.5, 0.75
    s = default_spacing_m(loa)
    pb = design_pressure_bottom(loa, beam, draft)
    ps = design_pressure_side(loa, beam, draft, depth)
    tb = max(plate_thickness_mm(pb, s, 2.5, 120.0),
             min_thickness_mm(loa, 1.0, "bottom"))
    ts = max(plate_thickness_mm(ps, s, 2.5, 120.0),
             min_thickness_mm(loa, 1.0, "side"))
    td = max(plate_thickness_mm(5.0, s, 2.5, 120.0),
             min_thickness_mm(loa, 1.0, "deck"))
    sec = assemble_midship(beam, depth, tb, ts, td,
                           n_bottom_long=int(beam / s),
                           n_deck_long=int(beam / s),
                           long_area_cm2=30.0)
    hog, _ = iacs_wave_bending_knm(loa, beam, cb)
    m_total_knm = hog * 1.5          # 정수 성분 개략 가산 (자릿수용)
    z_req_m3 = m_total_knm / (175.0 * 1000.0)   # σ=175 N/mm² → kN/m²·m³
    print(f"\n[2단계] Z_deck {sec.z_deck_m3:.3f} / Z_keel "
          f"{sec.z_keel_m3:.3f} / Z_req {z_req_m3:.3f} m³ "
          f"(tb {tb:.1f} ts {ts:.1f} td {td:.1f} mm)")
    assert 0.2 < sec.z_deck_m3 / z_req_m3 < 5.0
```

주의: σ=175/k N/mm² → Z = M/σ 단위 환산 (kN·m ÷ N/mm² = 10⁻³ m³ 배율) 재확인 — 단위 사고 다발 지점. 1 N/mm² = 1000 kN/m² → Z[m³] = M[kN·m] / (σ[N/mm²]·1000).

- [ ] **Step 2: 통과 확인 + 성적표·커밋**

Run: `python -m pytest tests/test_structure_scantlings.py -v -s`
Expected: 전부 PASS + 두께·단면계수 수치 출력 → worklog 2단계 성적표 추가

```bash
git add tests/test_structure_scantlings.py docs/worklog/2026-08-09.md
git commit -m "feat: 2단계 통합 — 규칙 두께 단면계수 vs UR S11 요구 자릿수 합의"
git push
```
