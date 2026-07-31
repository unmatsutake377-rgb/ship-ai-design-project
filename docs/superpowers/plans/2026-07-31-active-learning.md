# 능동 학습 1회전 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wigley L/B 스팬 3척 CFD 라벨로 Michell 보정 계수 ratio_w(B/L)=1+b·(B/L)를 적합하고, 보정 경험식으로 파레토 24척을 재평가한다.

**Architecture:** 새 모듈 2개 — `src/cfd/samples.py`(표본 산출물 생성)와 `src/cfd/calibration.py`(적합·재평가). CFD 실행은 기존 훅(hook.py + run_case.sh) 재사용. labels.py에 선택적 추가 열(loa_m, beam_m)만 확장. 스펙: `docs/superpowers/specs/2026-07-31-active-learning-design.md`.

**Tech Stack:** Python 3.13, trimesh, pandas, matplotlib(Agg), 기존 OpenFOAM Docker 훅.

## Global Constraints

- 한국어 docstring, 학습용 주석 (기존 관례)
- pytest는 OpenFOAM 없이 통과 (analytic-answer 원칙)
- 브랜치 `feat/active-learning` → main 직병합, 커밋 한국어 + Co-Authored-By
- 표본 공통: L=3.0 m, Fn=0.341 → v=1.850 m/s, Cb=0.444, T=B/1.6, depth=2T
- 적합 모형: ratio_w(B/L) = 1 + b·(B/L), 앵커 ratio(0)=1, 클리핑 [0.05, 1.5]
- 마찰(ITTC-57)은 무보정 — rw에만 ratio 적용

---

### Task 0: 브랜치

- [ ] **Step 1:**

```bash
cd "/Users/isong-eon/Desktop/선박 ai 모델 프로젝트"
git checkout -b feat/active-learning
```

---

### Task 1: 표본 생성기 (`samples.py`)

**Files:**
- Create: `src/cfd/samples.py`
- Test: `tests/test_cfd_samples.py`

**Interfaces:**
- Produces: `SAMPLES: list[dict]` — `{"name": "wigley_lb4", "loa": 3.0, "beam": 0.75}` 3건 (lb4/lb7/lb10, beam 0.75/0.4286/0.30)
- Produces: `FN_TARGET = 0.341`, `speed_for(loa: float) -> float` (= 0.341·√(9.81·L))
- Produces: `build_sample(sample: dict, out_root: Path) -> Path` — `<out_root>/<name>/`에 hull.stl + report.json (훅이 요구하는 필드: goal.target_speed_ms / dimensions{loa,beam,depth,draft_design,cb} / hydrostatics.draft / mesh_file / resistance{rf,rw,total,wetted_area})

- [ ] **Step 1: 실패하는 시험**

```python
"""능동 학습 표본 생성기 — Fn 고정·L/B 스팬 Wigley 3척."""
import json
import math

import pytest
import trimesh

from src.cfd.samples import FN_TARGET, SAMPLES, build_sample, speed_for


def test_samples_span_lb():
    lbs = [s["loa"] / s["beam"] for s in SAMPLES]
    assert lbs == pytest.approx([4.0, 7.0, 10.0], rel=0.01)
    assert all(s["loa"] == 3.0 for s in SAMPLES)


def test_speed_is_fixed_froude():
    v = speed_for(3.0)
    assert v / math.sqrt(9.81 * 3.0) == pytest.approx(FN_TARGET)


def test_build_sample_report_complete(tmp_path):
    d = build_sample(SAMPLES[0], tmp_path)
    report = json.loads((d / "report.json").read_text())
    assert report["goal"]["target_speed_ms"] == pytest.approx(speed_for(3.0))
    dims = report["dimensions"]
    assert dims["loa"] == 3.0 and dims["beam"] == 0.75
    # T = B/1.6 (표준 Wigley 비율)
    assert report["hydrostatics"]["draft"] == pytest.approx(0.75 / 1.6)
    r = report["resistance"]
    assert r["rw"] > 0 and r["rf"] > 0
    mesh = trimesh.load(d / report["mesh_file"])
    assert mesh.is_watertight
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_cfd_samples.py -q`
Expected: FAIL — `No module named 'src.cfd.samples'`

- [ ] **Step 3: 구현**

```python
"""능동 학습 표본 (스펙 §3) — Fn 고정, L/B만 스팬.

Michell 오차는 L/B와 Fn 둘 다에 의존 — 한 번에 한 변수만 움직인다
(변인 통제). Fn=0.341은 기존 데모 라벨(L/B=2.0)과 같은 값이라 4점이
한 곡선 위에 놓인다.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from src.ai.hull_generator import generate_hull_mesh, solve_exponents
from src.core.types import MainDimensions
from src.physics.resistance import total_resistance

FN_TARGET = 0.341
CB_STD = 0.444          # 표준 Wigley (n=m=2)
BT_STD = 1.6            # 표준 B/T

SAMPLES = [
    {"name": "wigley_lb4", "loa": 3.0, "beam": 0.75},
    {"name": "wigley_lb7", "loa": 3.0, "beam": 3.0 / 7.0},
    {"name": "wigley_lb10", "loa": 3.0, "beam": 0.30},
]


def speed_for(loa: float) -> float:
    """Fn=0.341 고정 속도 [m/s]."""
    return FN_TARGET * math.sqrt(9.81 * loa)


def build_sample(sample: dict, out_root: Path) -> Path:
    """표본 1척의 훅 입력 폴더(hull.stl + report.json) 생성."""
    draft = sample["beam"] / BT_STD
    dims = MainDimensions(loa=sample["loa"], beam=sample["beam"],
                          depth=2.0 * draft, draft_design=draft, cb=CB_STD)
    n, m = solve_exponents(CB_STD)
    mesh = generate_hull_mesh(dims)
    speed = speed_for(dims.loa)
    resist = total_resistance(mesh, dims, n, m, draft=draft, speed=speed)

    out = Path(out_root) / sample["name"]
    out.mkdir(parents=True, exist_ok=True)
    mesh.export(out / "hull.stl")
    report = {
        "goal": {"target_speed_ms": speed},
        "dimensions": dataclasses.asdict(dims),
        "hydrostatics": {"draft": draft},
        "mesh_file": "hull.stl",
        "resistance": dataclasses.asdict(resist),
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    return out


def main() -> int:
    out_root = Path("outputs/al_samples")
    for s in SAMPLES:
        d = build_sample(s, out_root)
        print(f"{s['name']}: {d}")
        print(f"  python -m src.cfd.hook --report {d} --mode simple")
        print(f"  python -m src.cfd.hook --report {d} --mode inter")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_cfd_samples.py -q`
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add src/cfd/samples.py tests/test_cfd_samples.py
git commit -m "feat: 능동학습 1/4 — L/B 스팬 표본 생성기 (Fn 0.341 고정)"
```

---

### Task 2: 라벨에 치수 열 추가 (`labels.py`·`hook.py` 확장)

**Files:**
- Modify: `src/cfd/labels.py` (append_label에 `extra` 인자)
- Modify: `src/cfd/hook.py` (dims 전달)
- Test: `tests/test_cfd_labels.py` (추가)

**Interfaces:**
- Produces: `append_label(..., extra: dict | None = None)` — extra 키가 열로 병합 (예: `{"loa_m": 3.0, "beam_m": 0.75}`)

- [ ] **Step 1: 실패하는 시험 추가** (tests/test_cfd_labels.py 끝에)

```python
def test_extra_columns_stored(tmp_path):
    csv = tmp_path / "cfd_labels.csv"
    df = append_label(csv, "wigley_lb4_simple", 1.85, 0.47, R1, EMP,
                      extra={"loa_m": 3.0, "beam_m": 0.75})
    assert df.iloc[0]["loa_m"] == 3.0
    assert df.iloc[0]["beam_m"] == 0.75
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_cfd_labels.py -q`
Expected: 1 failed (TypeError: unexpected keyword 'extra'), 3 passed

- [ ] **Step 3: 구현** — labels.py의 append_label 시그니처와 row 구성 변경:

```python
def append_label(csv_path: Path, case_name: str, speed: float, draft: float,
                 result: CfdResult, empirical: dict,
                 extra: dict | None = None) -> pd.DataFrame:
```

row dict 마지막에:

```python
    if extra:
        row.update(extra)
```

hook.py의 두 append_label 호출부(--parse-only 분기)를 다음으로 교체:

```python
        dims = report["dimensions"]
        append_label(Path(args.labels), name, speed, draft, result,
                     report["resistance"],
                     extra={"loa_m": dims["loa"], "beam_m": dims["beam"]})
```

- [ ] **Step 4: 통과 + 전체 회귀**

Run: `python3 -m pytest tests/test_cfd_labels.py tests/test_cfd_hook.py -q`
Expected: 전부 통과 (hook 시험의 report에는 dimensions.beam이 없으므로 `_fake_report_dir`의 report dict에 `"beam": 0.99` 추가 — dimensions를 `{"loa": 3.7, "beam": 0.99}`로)

- [ ] **Step 5: 기존 라벨 2행 치수 소급 기입**

```bash
python3 - <<'EOF'
import pandas as pd
df = pd.read_csv("data/cfd_labels.csv")
df.loc[df.case_name.str.startswith("demo_cfd"), "loa_m"] = 1.97
df.loc[df.case_name.str.startswith("demo_cfd"), "beam_m"] = 0.99
df.to_csv("data/cfd_labels.csv", index=False)
print(df[["case_name", "loa_m", "beam_m"]])
EOF
```

정확한 값은 `outputs/demo_cfd/report.json`의 dimensions에서 확인해 사용 (1.97/0.99는 반올림 표시 — 소수 전체 자리 기입).

- [ ] **Step 6: 커밋**

```bash
git add src/cfd/labels.py src/cfd/hook.py tests/ data/cfd_labels.csv
git commit -m "feat: 능동학습 2/4 — 라벨에 치수 열 (보정 계수의 설명변수)"
```

---

### Task 3: 보정 계수 적합 (`calibration.py`)

**Files:**
- Create: `src/cfd/calibration.py`
- Test: `tests/test_cfd_calibration.py`

**Interfaces:**
- Consumes: `data/cfd_labels.csv` (Task 2 스키마)
- Produces: `fit_wave_ratio(points: list[tuple[float, float]]) -> float` — (B/L, ratio) 점들로 앵커드 최소제곱 b
- Produces: `wave_ratio(bl: float, b: float, lo=0.05, hi=1.5) -> float`
- Produces: `ratios_from_labels(df: pd.DataFrame) -> list[tuple[float, float]]` — case_name의 `_simple_`/`_inter_` 짝을 묶어 ratio=(P_inter−P_simple)/rw_emp

- [ ] **Step 1: 실패하는 시험**

```python
"""Michell 보정 계수 — 앵커드 최소제곱 (손계산 정답지)."""
import pandas as pd
import pytest

from src.cfd.calibration import fit_wave_ratio, ratios_from_labels, wave_ratio


def test_fit_recovers_known_slope():
    """ratio = 1 - 1.6·(B/L) 인 합성 점 → b=-1.6 정확 복원."""
    pts = [(x, 1 - 1.6 * x) for x in (0.1, 0.25, 0.5)]
    assert fit_wave_ratio(pts) == pytest.approx(-1.6)


def test_fit_least_squares_hand_calc():
    """잡음 점: b = Σ(r-1)x / Σx² 손계산과 일치."""
    pts = [(0.5, 0.2), (0.1, 0.9)]
    b_hand = ((0.2 - 1) * 0.5 + (0.9 - 1) * 0.1) / (0.5**2 + 0.1**2)
    assert fit_wave_ratio(pts) == pytest.approx(b_hand)


def test_anchor_at_zero():
    """B/L=0 (무한히 얇음) → ratio=1: Michell이 정확한 극한."""
    assert wave_ratio(0.0, b=-1.6) == 1.0


def test_clipping():
    assert wave_ratio(1.0, b=-5.0) == 0.05    # 하한
    assert wave_ratio(1.0, b=+5.0) == 1.5     # 상한


def test_ratios_from_labels_pairs_modes():
    df = pd.DataFrame([
        {"case_name": "wigley_lb4_simple_1.85ms", "cfd_pressure_n": 4.0,
         "emp_rw_n": 10.0, "loa_m": 3.0, "beam_m": 0.75},
        {"case_name": "wigley_lb4_inter_1.85ms", "cfd_pressure_n": 9.0,
         "emp_rw_n": 10.0, "loa_m": 3.0, "beam_m": 0.75},
    ])
    pts = ratios_from_labels(df)
    assert pts == [pytest.approx((0.25, 0.5))]   # (B/L, (9-4)/10)
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_cfd_calibration.py -q`
Expected: FAIL — import 에러

- [ ] **Step 3: 구현**

```python
"""Michell 보정 계수 (스펙 §4) — 능동 학습 1회전의 '학습' 부분.

모형: ratio_w(B/L) = 1 + b·(B/L). 절편은 물리 앵커(B/L→0에서 Michell
정확 → ratio=1)로 고정 — 소표본(4점)에서 매개변수는 b 하나만 배운다.
앵커드 최소제곱: b = Σ(ratio_i − 1)·x_i / Σ x_i²  (x = B/L).

CFD 조파 추정 = P_자유수면 − P_이중모형: 같은 척을 두 모드로 돌려
형상(점성 압력) 성분을 빼내면 파도 몫만 남는다.
"""
from __future__ import annotations

import pandas as pd

RATIO_CLIP = (0.05, 1.5)   # 외삽 폭주 방지 (스펙 §4)


def fit_wave_ratio(points: list[tuple[float, float]]) -> float:
    num = sum((r - 1.0) * x for x, r in points)
    den = sum(x * x for x, _ in points)
    return num / den


def wave_ratio(bl: float, b: float,
               lo: float = RATIO_CLIP[0], hi: float = RATIO_CLIP[1]) -> float:
    return min(hi, max(lo, 1.0 + b * bl))


def ratios_from_labels(df: pd.DataFrame) -> list[tuple[float, float]]:
    """라벨 CSV에서 (B/L, ratio) 점 추출 — _simple_/_inter_ 짝 필수."""
    points = []
    df = df.dropna(subset=["loa_m", "beam_m"])
    bases = {}
    for _, row in df.iterrows():
        if "_simple_" in row.case_name:
            base = row.case_name.replace("_simple_", "_")
            bases.setdefault(base, {})["simple"] = row
        elif "_inter_" in row.case_name:
            base = row.case_name.replace("_inter_", "_")
            bases.setdefault(base, {})["inter"] = row
    for base, pair in sorted(bases.items()):
        if "simple" not in pair or "inter" not in pair:
            continue
        s, i = pair["simple"], pair["inter"]
        ratio = (i.cfd_pressure_n - s.cfd_pressure_n) / i.emp_rw_n
        points.append((float(i.beam_m / i.loa_m), float(ratio)))
    return points
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_cfd_calibration.py -q`
Expected: 5 passed

- [ ] **Step 5: 커밋**

```bash
git add src/cfd/calibration.py tests/test_cfd_calibration.py
git commit -m "feat: 능동학습 3/4 — 보정 계수 적합 (물리 앵커 + 클리핑)"
```

---

### Task 4: 파레토 재평가 (`calibration.py` 확장)

**Files:**
- Modify: `src/cfd/calibration.py` (reevaluate_pareto + plot + CLI)
- Test: `tests/test_cfd_calibration.py` (추가)

**Interfaces:**
- Produces: `reevaluate_pareto(pareto_csv: Path, b: float, speed: float = 1.2) -> pd.DataFrame` — 입력 열(loa, lb, bt, cb, resistance_n, …)에 `rw_orig, rf_orig, ratio, resistance_corrected, pareto_before, pareto_after` 추가
- Produces: `plot_reeval(df, b, points, path)` — 보정 전/후 전선 + ratio 곡선 그림
- Consumes: `_mark_pareto` 논리 (src/screen_shipd.py의 비지배 판정과 동일 — 재구현)

- [ ] **Step 1: 실패하는 시험 추가**

```python
def test_reevaluate_conserves_friction(tmp_path):
    """보정은 조파만 깎는다: 보정 전저항 = rf + rw×ratio 폐합."""
    import pandas as pd

    from src.cfd.calibration import reevaluate_pareto
    df_in = pd.DataFrame([
        {"loa": 3.0, "lb": 4.0, "bt": 1.6, "cb": 0.444,
         "resistance_n": 10.0, "total_mass_kg": 100.0,
         "stability_margin": 0.1, "feasible": True},
        {"loa": 3.0, "lb": 10.0, "bt": 1.6, "cb": 0.444,
         "resistance_n": 8.0, "total_mass_kg": 120.0,
         "stability_margin": 0.1, "feasible": True},
    ])
    csv = tmp_path / "pareto.csv"
    df_in.to_csv(csv, index=False)
    out = reevaluate_pareto(csv, b=-1.6, speed=1.2)
    for _, row in out.iterrows():
        assert row.resistance_corrected == pytest.approx(
            row.rf_orig + row.rw_orig * row.ratio, rel=1e-6)
        # 통통할수록(B/L↑) ratio↓
    assert out.iloc[0]["ratio"] < out.iloc[1]["ratio"]
    assert set(out.columns) >= {"pareto_before", "pareto_after"}
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_cfd_calibration.py::test_reevaluate_conserves_friction -q`
Expected: FAIL — reevaluate_pareto 없음

- [ ] **Step 3: 구현** — calibration.py에 추가:

```python
def _mark_pareto(f) -> "np.ndarray":
    """3목적 (저항↓, 중량↓, −안정여유↓) 비지배 표시 — screen_shipd와 동일 논리."""
    import numpy as np

    n = len(f)
    flags = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and (f[j] <= f[i]).all() and (f[j] < f[i]).any():
                flags[i] = False
                break
    return flags


def reevaluate_pareto(pareto_csv, b: float, speed: float = 1.2):
    """보정 경험식으로 파레토 후보 재평가 (스펙 §5 — Fn 외삽 시연).

    pareto.csv에는 rw/rf 분해가 없으므로 치수(loa, lb, bt, cb)로 선형을
    재생성해 설계 흘수에서 rw·rf를 재계산한다 (평형 흘수가 아니라 설계
    흘수 — 근사임을 리포트에 명시)."""
    import numpy as np

    from src.ai.hull_generator import generate_hull_mesh, solve_exponents
    from src.core.types import MainDimensions
    from src.physics.resistance import total_resistance

    df = pd.read_csv(pareto_csv)
    df = df[df["feasible"]].reset_index(drop=True)
    rows = []
    for _, c in df.iterrows():
        beam = c.loa / c.lb
        draft = beam / c.bt
        dims = MainDimensions(loa=c.loa, beam=beam, depth=2 * draft,
                              draft_design=draft, cb=c.cb)
        n_exp, m_exp = solve_exponents(c.cb)
        r = total_resistance(generate_hull_mesh(dims), dims, n_exp, m_exp,
                             draft=draft, speed=speed)
        ratio = wave_ratio(beam / c.loa, b)
        rows.append({**c, "rw_orig": r.rw, "rf_orig": r.rf, "ratio": ratio,
                     "resistance_corrected": r.rf + r.rw * ratio})
    out = pd.DataFrame(rows)
    before = np.column_stack([out.resistance_n, out.total_mass_kg,
                              -out.stability_margin])
    after = np.column_stack([out.resistance_corrected, out.total_mass_kg,
                             -out.stability_margin])
    out["pareto_before"] = _mark_pareto(before)
    out["pareto_after"] = _mark_pareto(after)
    return out
```

CLI(main)도 추가 — 라벨 로드 → 적합 → 재평가 → 그림 저장 → 변동 요약 출력:

```python
def plot_reeval(df, b: float, points, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
    xs = np.linspace(0, 0.55, 100)
    ax1.plot(xs, [wave_ratio(x, b) for x in xs], "-",
             color="#0f766e", label=f"적합: 1 + ({b:.2f})·(B/L)")
    px, py = zip(*points)
    ax1.scatter(px, py, s=70, color="#dc2626", zorder5=5 if False else 5,
                label="CFD 관측점")
    ax1.set_xlabel("B/L (통통함)")
    ax1.set_ylabel("ratio_w = CFD 조파 / Michell")
    ax1.set_title("Michell 보정 계수 (Fn≈0.34)")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.scatter(df.resistance_n, df.total_mass_kg, s=25, alpha=0.4,
                color="#94a3b8", label="보정 전")
    ax2.scatter(df.resistance_corrected, df.total_mass_kg, s=25, alpha=0.7,
                color="#0f766e", label="보정 후")
    fb = df[df.pareto_after]
    ax2.scatter(fb.resistance_corrected, fb.total_mass_kg, s=90,
                facecolor="none", edgecolor="#dc2626", label="보정 후 전선")
    ax2.set_xlabel("전저항 [N] (Fn 외삽 시연)")
    ax2.set_ylabel("전체 중량 [kg]")
    ax2.set_title("파레토 재평가")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
```

(`zorder5=5 if False else 5`는 오타 방지용이 아님 — **zorder=5로 쓸 것**. 계획 자가 수정: `ax1.scatter(px, py, s=70, color="#dc2626", zorder=5, label="CFD 관측점")`)

```python
def main() -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="능동 학습 1회전: 적합+재평가")
    parser.add_argument("--labels", default="data/cfd_labels.csv")
    parser.add_argument("--pareto", default="outputs/pareto/pareto.csv")
    parser.add_argument("--out", default="outputs/active_learning")
    args = parser.parse_args()

    points = ratios_from_labels(pd.read_csv(args.labels))
    if len(points) < 2:
        print(f"짝 라벨 부족 ({len(points)}점) — CFD 실행 먼저")
        return 2
    b = fit_wave_ratio(points)
    print(f"적합 b = {b:.3f}  (점 {len(points)}개)")
    for x, r in points:
        print(f"  B/L={x:.3f}: 관측 {r:.3f} vs 적합 {wave_ratio(x, b):.3f}")
    df = reevaluate_pareto(args.pareto, b)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "reeval.csv", index=False)
    plot_reeval(df, b, points, out / "reeval.png")
    changed = df[df.pareto_before != df.pareto_after]
    print(f"전선 변동: {len(changed)}척 (전 {df.pareto_before.sum()} → "
          f"후 {df.pareto_after.sum()})")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
```

- [ ] **Step 4: 통과 + 전체 회귀**

Run: `python3 -m pytest tests/test_cfd_calibration.py -q && python3 -m pytest -q`
Expected: 전부 통과

- [ ] **Step 5: 커밋**

```bash
git add src/cfd/calibration.py tests/test_cfd_calibration.py
git commit -m "feat: 능동학습 4/4 — 파레토 재평가 (조파만 보정, 전선 변동 리포트)"
```

---

### Task 5: CFD 실행 — 3척 × 2모드 (수동, 백그라운드)

- [ ] **Step 1: 표본 생성 + 케이스 6개 생성**

```bash
python3 -m src.cfd.samples
for lb in lb4 lb7 lb10; do
  python3 -m src.cfd.hook --report outputs/al_samples/wigley_$lb --mode simple
  python3 -m src.cfd.hook --report outputs/al_samples/wigley_$lb --mode inter
done
```

- [ ] **Step 2: 단상 3건 병렬 실행** (각 ~10분, 코어 여유 충분)

```bash
for lb in lb4 lb7 lb10; do
  ./cfd/docker/run_case.sh "outputs/cfd_cases/wigley_${lb}_simple_1.85ms" simpleFoam &
done
wait
```

주의: 케이스 폴더명의 속도 문자열은 hook가 `report_dir.name + mode + speed`로 만든다 — speed_for(3.0)=1.8499...가 이름에 그대로 들어가므로 실제 생성된 폴더명을 `ls outputs/cfd_cases/`로 확인 후 사용.

- [ ] **Step 3: 자유수면 3건 병렬 실행** (각 ~50분, 병렬로 벽시계 ~1시간)

```bash
for lb in lb4 lb7 lb10; do
  ./cfd/docker/run_case.sh "outputs/cfd_cases/wigley_${lb}_inter_1.85ms" interFoam &
done
wait
```

- [ ] **Step 4: 라벨 수확 6건**

```bash
for lb in lb4 lb7 lb10; do
  python3 -m src.cfd.hook --report outputs/al_samples/wigley_$lb --mode simple --parse-only
  python3 -m src.cfd.hook --report outputs/al_samples/wigley_$lb --mode inter --parse-only
done
```

Expected: data/cfd_labels.csv 8행 (기존 2 + 신규 6)

- [ ] **Step 5: 커밋**

```bash
git add data/cfd_labels.csv
git commit -m "data: 능동학습 CFD 라벨 6행 — L/B 4/7/10 × 단상·자유수면"
```

---

### Task 6: 적합·재평가 실행 + 기록 + 병합

- [ ] **Step 1: 사이클 실행**

```bash
python3 -m src.cfd.calibration
```

Expected: b 값, 4점 관측-적합 대조, 전선 변동 요약, outputs/active_learning/{reeval.csv, reeval.png}

- [ ] **Step 2: 결과 판독** — b가 음수(통통할수록 Michell 과대 깎음)인지, L/B=10 점이 ratio≈1 근방인지 (표준 Wigley에서 Michell 잘 맞음 = 물리 정합성 검사). 아니면 원인 분석 후 worklog에 기록

- [ ] **Step 3: 기록 3축 + README/설명서** — worklog 2026-07-31 (표·그림·b값·전선 변동), PROGRESS 마일스톤 행, feedback-log (오너 결정 3건: 보정 계수/L/B 스팬/3척 이유 질문), README 로드맵, manual

- [ ] **Step 4: 전체 회귀 + 병합**

```bash
python3 -m pytest -q
git add -A && git commit -m "docs: 능동 학습 1회전 완결 기록"
git checkout main && git merge --no-ff feat/active-learning -m "병합: feat/active-learning — 능동 학습 1회전 (Michell 보정 계수)"
python3 -m pytest -q && git push origin main
```

---

## Self-Review 결과

- 스펙 커버리지: §2 아키텍처(Task 1·3·4), §3 표본(Task 1·5), §4 수학(Task 3), §5 재평가(Task 4·6), §6 검증(각 태스크 pytest), §7 한계(Task 6 기록) — 누락 없음
- 플레이스홀더: 없음 (plot 코드의 zorder 오타는 본문에서 자가 수정 명시)
- 타입 일관성: fit_wave_ratio/wave_ratio/ratios_from_labels/reevaluate_pareto 시그니처 태스크 간 일치. hook의 extra dict 키(loa_m/beam_m)와 calibration의 사용 열 일치
- 리스크: 케이스 폴더명 속도 문자열(1.8499…) — Task 5 Step 2에 확인 절차 내장. interFoam 3건 병렬 메모리 — 실패 시 순차 전환
