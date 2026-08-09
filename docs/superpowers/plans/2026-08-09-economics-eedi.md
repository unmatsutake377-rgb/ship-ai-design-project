# 경제성 1단계 — EEDI 정본 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) 문법.

**Goal:** attained EEDI (CF·SFC·P75/(DWT·Vref)) + required EEDI (기준선 × Phase 3 감축) — 손계산·기준선 앵커로 검증된 판정 코어.

**Architecture:** 신설 `src/physics/economics/eedi.py` 단일 모듈 — 순수 수치 (파이프라인 결합은 3단계). 원전 3건 확보됨 (`references/`): MEPC.328(76) p37 Table 1 감축률·p39 Table 2 기준선, MEPC.364(79) CF 표, MEPC.231(65) 방법론.

**Tech Stack:** Python 3, pytest. 외부 의존 없음.

## Global Constraints

- 원전 박제값: General cargo **a=107.48·c=0.216** (Table 2 p39), Phase 3 (2022-04+) 감축 **30%** (15,000 DWT+, 3,000~15,000은 0→30 선형 보간, <3,000 적용 밖 — Table 1 p37 이미지 판독), CF: **MDO 3.206·HFO 3.114** tCO₂/t (MEPC.364 표)
- attained = CF·SFC·P_75 / (DWT·Vref) — P_75 = 0.75·MCR, Vref = 75% MCR에서 속도 (P∝V³ 역산, 정직 표기), SFC [g/kWh], 결과 [gCO₂/(t·nm)] — 1 nm = 1.852 km, 속도 knots 환산 주의 (단위 사고 다발 지점)
- 한국어 docstring, 기존 시험 396 통과 유지, main 직커밋

---

### Task 1: `eedi.py` — attained·required·판정

**Files:**
- Create: `src/physics/economics/__init__.py` (빈 파일)
- Create: `src/physics/economics/eedi.py`
- Test: `tests/test_eedi.py`

**Interfaces:**
- Produces:
  - `attained_eedi(mcr_kw, sfoc_g_per_kwh, dwt_t, v_service_ms, p_service_kw, cf_t_co2_per_t=3.206) -> dict` — 키: `eedi_g_per_tnm, v_ref_kn, p_75_kw` (Vref = v_service·(0.75·MCR/P_service)^(1/3))
  - `required_eedi(dwt_t, phase=3) -> dict` — 키: `reference_g_per_tnm, reduction_pct, required_g_per_tnm, applicable` (<3,000 DWT = False)
  - `eedi_verdict(attained, required) -> dict` — passed·margin_pct
  - `class EEDIRangeError(ValueError)` — DWT<3,000 정직 거절은 required가 applicable=False로 (예외 아님 — 게이트에서 성적표 강등)

- [ ] **Step 1: 실패 시험 작성**

```python
# tests/test_eedi.py
"""EEDI 정본 — 손계산·기준선 앵커 (경제성 1단계, 스펙 §3)."""
import pytest


def test_reference_line_hand_calc():
    """기준선 a·DWT^{-c} 손계산 — 원전 Table 2 (General cargo
    a=107.48, c=0.216). DWT 10,000 → 107.48·10000^-0.216."""
    from src.physics.economics.eedi import required_eedi
    r = required_eedi(10_000.0, phase=3)
    ref = 107.48 * 10_000.0 ** (-0.216)
    assert r["reference_g_per_tnm"] == pytest.approx(ref, rel=1e-9)
    assert 14.0 < ref < 15.5          # 자릿수 sanity (문헌 대역)


def test_reduction_factor_bands():
    """감축률 — 원전 Table 1: 15,000+ = 30%, 3,000~15,000 선형
    0→30, <3,000 적용 밖."""
    from src.physics.economics.eedi import required_eedi
    assert required_eedi(20_000.0)["reduction_pct"] == pytest.approx(30.0)
    mid = required_eedi(9_000.0)["reduction_pct"]
    assert 0.0 < mid < 30.0
    assert required_eedi(9_000.0)["applicable"] is True
    small = required_eedi(2_000.0)
    assert small["applicable"] is False


def test_attained_hand_calc():
    """attained 손계산 — MCR 4640 kW·SFOC 178.8·DWT 8000·
    설계 7 m/s @ 2900 kW: P75 = 3480 kW, Vref = 7·(3480/2900)^⅓,
    EEDI = 3.206e6·178.8e-3·3480 / (8000·Vref_kn)  [g/(t·nm)]."""
    from src.physics.economics.eedi import attained_eedi
    r = attained_eedi(mcr_kw=4640.0, sfoc_g_per_kwh=178.8,
                      dwt_t=8000.0, v_service_ms=7.0,
                      p_service_kw=2900.0)
    p75 = 0.75 * 4640.0
    v_ref_ms = 7.0 * (p75 / 2900.0) ** (1.0 / 3.0)
    v_ref_kn = v_ref_ms * 3600.0 / 1852.0
    expected = (3.206 * 178.8 * p75) / (8000.0 * v_ref_kn)
    assert r["eedi_g_per_tnm"] == pytest.approx(expected, rel=1e-9)
    assert r["v_ref_kn"] == pytest.approx(v_ref_kn, rel=1e-9)
    assert 5.0 < r["eedi_g_per_tnm"] < 40.0   # 일반화물선 자릿수


def test_verdict():
    """판정 — attained ≤ required 합격·여유 % 부호."""
    from src.physics.economics.eedi import eedi_verdict
    v = eedi_verdict(10.0, 12.0)
    assert v["passed"] is True
    assert v["margin_pct"] == pytest.approx(100.0 * (12.0 - 10.0) / 12.0)
    assert eedi_verdict(13.0, 12.0)["passed"] is False
```

- [ ] **Step 2: 실패 확인** — Run: `python -m pytest tests/test_eedi.py -q` → FAIL ModuleNotFoundError

- [ ] **Step 3: 구현**

```python
# src/physics/economics/eedi.py
"""IMO EEDI — attained·required·판정 (경제성 1단계, 스펙 §2).

원전 (references/, imo.org 공식 PDF):
- MEPC.328(76) p39 Table 2: 기준선 General cargo a=107.48·c=0.216
  (reference = a·DWT^{-c} [gCO₂/(t·nm)])
- MEPC.328(76) p37 Table 1 (이미지 판독): Phase 3 (2022-04+) 감축
  30% — General cargo 15,000 DWT+, 3,000~15,000 선형 0→30 (* 각주
  보간 계보), <3,000 적용 밖
- MEPC.364(79) CF 표: MDO 3.206·HFO 3.114 tCO₂/t-fuel

attained = CF·SFC·P75 / (DWT·Vref) — P75 = 75% MCR (원전 관례),
Vref는 설계점에서 P ∝ V³ 역산 (프로펠러 법칙 근사, 정직 표기).
단위: SFC g/kWh·CF t/t → 분자 gCO₂/h, 분모 t·kn = t·nm/h.
"""
from __future__ import annotations

CF_MDO = 3.206      # tCO₂/t-fuel (MEPC.364 표 — Diesel/Gas Oil)
CF_HFO = 3.114
REF_A_GENERAL_CARGO = 107.48    # MEPC.328(76) Table 2
REF_C_GENERAL_CARGO = 0.216
PHASE3_REDUCTION_PCT = 30.0     # Table 1, General cargo 15k+
DWT_FULL_REDUCTION = 15_000.0
DWT_MIN_APPLICABLE = 3_000.0
KN_PER_MS = 3600.0 / 1852.0


def attained_eedi(mcr_kw: float, sfoc_g_per_kwh: float, dwt_t: float,
                  v_service_ms: float, p_service_kw: float,
                  cf_t_co2_per_t: float = CF_MDO) -> dict:
    """설계 배의 attained EEDI [gCO₂/(t·nm)]."""
    p75 = 0.75 * mcr_kw
    v_ref_ms = v_service_ms * (p75 / max(p_service_kw, 1e-9)) ** (1.0 / 3.0)
    v_ref_kn = v_ref_ms * KN_PER_MS
    eedi = (cf_t_co2_per_t * sfoc_g_per_kwh * p75) \
        / (max(dwt_t, 1e-9) * max(v_ref_kn, 1e-9))
    return {"eedi_g_per_tnm": eedi, "v_ref_kn": v_ref_kn,
            "p_75_kw": p75,
            "note": "Vref = 설계점 P∝V³ 역산 (프로펠러 법칙 근사)"}


def required_eedi(dwt_t: float, phase: int = 3) -> dict:
    """required EEDI — 기준선 × (1 − 감축률). General cargo 계보만
    (타 선종 정직 미지원)."""
    ref = REF_A_GENERAL_CARGO * dwt_t ** (-REF_C_GENERAL_CARGO)
    if dwt_t < DWT_MIN_APPLICABLE:
        return {"reference_g_per_tnm": ref, "reduction_pct": 0.0,
                "required_g_per_tnm": ref, "applicable": False,
                "note": "DWT < 3,000 — EEDI 규제 적용 밖 (원전 Table 1)"}
    if dwt_t >= DWT_FULL_REDUCTION:
        red = PHASE3_REDUCTION_PCT
    else:
        red = PHASE3_REDUCTION_PCT * (
            (dwt_t - DWT_MIN_APPLICABLE)
            / (DWT_FULL_REDUCTION - DWT_MIN_APPLICABLE))
    return {"reference_g_per_tnm": ref, "reduction_pct": red,
            "required_g_per_tnm": ref * (1.0 - red / 100.0),
            "applicable": True,
            "note": "Phase 3 (2022-04+) — General cargo 계보"}


def eedi_verdict(attained_g_per_tnm: float,
                 required_g_per_tnm: float) -> dict:
    """합불 + 여유율 (%). margin > 0 = 여유 있음."""
    margin = 100.0 * (required_g_per_tnm - attained_g_per_tnm) \
        / max(required_g_per_tnm, 1e-9)
    return {"passed": bool(attained_g_per_tnm <= required_g_per_tnm),
            "margin_pct": margin}
```

- [ ] **Step 4: 통과 확인** — Run: `python -m pytest tests/test_eedi.py -v` → 4 PASS

- [ ] **Step 5: 우리 100m급 첫 판정 (성적표 수치)**

```python
# 일회 스크립트 — 8000t 화물선 실측값 대입
from src.physics.economics.eedi import attained_eedi, required_eedi, eedi_verdict
# 115.7m 설계 실측 (조종 게이트 회차): 8L32 MCR 4640 kW·SFOC 178.8·
# 부하 62% → P_service ≈ 2877 kW·7 m/s·DWT 8000
a = attained_eedi(4640.0, 178.8, 8000.0, 7.0, 0.62 * 4640.0)
r = required_eedi(8000.0)
print(a, r, eedi_verdict(a["eedi_g_per_tnm"], r["required_g_per_tnm"]))
```

- [ ] **Step 6: 커밋 + worklog 성적표**

```bash
git add src/physics/economics/ tests/test_eedi.py
git commit -m "feat: EEDI 정본 — 기준선·Phase3 감축·attained 손계산 앵커 (경제성 1단계)"
```

worklog에 1단계 성적표 (원전 3건 인덱스·손계산 앵커·첫 판정) 기록, 스펙 §3에 원전 인덱스 보강, 커밋·푸시 후 오너 보고.
