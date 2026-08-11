# Streamlit 시연 앱 1단계 (라이브 설계 페이지) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** "속도·짐·용도 3입력 → 8중 게이트 검증된 배" 전 과정을 로컬 브라우저에서 체험하는 라이브 설계 페이지.

**Architecture:** `app/ui/*`는 streamlit 비의존 순수 함수 (단위시험 대상 — 메쉬→plotly 변환·게이트 신호등 가공·성적표 카드). streamlit 렌더는 `app/main.py`+`app/pages/`에만 (수동 브라우저 검증). 파이프라인은 `run_pipeline` 임포트 재사용 — 새 물리 0줄.

**Tech Stack:** streamlit 1.51 / plotly 6.3 (둘 다 설치 확인) / 기존 src.* 사슬. 시험 러너 `/opt/anaconda3/bin/python -m pytest`.

## Global Constraints

- 인터넷 공개 배포 절대 없음 — 로컬 실행 전용 (스펙 헤더)
- Ship-D 생성 형상 커밋·공개 금지 — 앱은 화면 표시만, 파일 저장은 기존 out_dir 관례 (outputs/ gitignore)
- 정직 스킵 표기: 게이트 3태 (합격/불합격/스킵 — "데이터 없음 ≠ 불합격" 문구)
- 코드 식별자 영어·docstring 한국어, 커밋 conventional prefix 한국어

---

### Task 1: `app/ui/mesh_view.py` — 메쉬 → plotly Mesh3d 데이터

**Files:**
- Create: `app/__init__.py`, `app/ui/__init__.py` (빈 파일)
- Create: `app/ui/mesh_view.py`
- Test: `tests/test_app_ui.py`

**Interfaces:**
- Produces: `mesh_to_plotly(mesh: trimesh.Trimesh) -> dict` — keys `x,y,z,i,j,k` (list[float]/list[int], plotly Mesh3d 인자 그대로)

- [ ] **Step 1: 실패 시험**

```python
"""Streamlit 앱 ui 헬퍼 — streamlit 비의존 순수 함수 시험."""
import pytest
import trimesh


def test_mesh_to_plotly_preserves_counts():
    """정점·면 수 보존 + plotly Mesh3d 키 계약."""
    from app.ui.mesh_view import mesh_to_plotly
    box = trimesh.creation.box(extents=[2.0, 1.0, 0.5])
    d = mesh_to_plotly(box)
    assert set(d) >= {"x", "y", "z", "i", "j", "k"}
    assert len(d["x"]) == len(box.vertices)
    assert len(d["i"]) == len(box.faces)
    assert max(d["i"] + d["j"] + d["k"]) < len(box.vertices)
```

- [ ] **Step 2: 실패 확인** — `/opt/anaconda3/bin/python -m pytest tests/test_app_ui.py -q` → FAIL (`No module named 'app'`)

- [ ] **Step 3: 구현**

```python
"""메쉬 → plotly Mesh3d 인자 변환 (streamlit 비의존)."""
from __future__ import annotations

import trimesh


def mesh_to_plotly(mesh: trimesh.Trimesh) -> dict:
    """정점 xyz + 삼각형 ijk — plotly.graph_objects.Mesh3d(**d)."""
    v = mesh.vertices
    f = mesh.faces
    return {
        "x": v[:, 0].tolist(), "y": v[:, 1].tolist(),
        "z": v[:, 2].tolist(),
        "i": f[:, 0].tolist(), "j": f[:, 1].tolist(),
        "k": f[:, 2].tolist(),
    }
```

- [ ] **Step 4: 통과 확인** — 같은 명령 → PASS
- [ ] **Step 5: 커밋** — `git add app tests/test_app_ui.py && git commit -m "feat: 앱 ui — 메쉬 plotly 변환"`

### Task 2: `app/ui/gates.py` — 리포트 → 8중 게이트 신호등

**Files:**
- Create: `app/ui/gates.py`
- Test: `tests/test_app_ui.py` (추가)

**Interfaces:**
- Consumes: `run_pipeline` 리포트 dict (키: hydrostatics/maxbox/seakeeping/structure/maneuvering/economics/space_large, 각 값 None|dict, dict는 passed·skipped 선택 보유)
- Produces: `gate_cards(report: dict) -> list[dict]` — 각 `{"name": str, "state": "pass"|"fail"|"skip"|"off", "detail": str}` (순서 고정 8개: 뜨나·실리나·안 넘어지나·안 흔들리나·안 부러지나·도나·규제·성적)

- [ ] **Step 1: 실패 시험**

```python
def test_gate_cards_three_states():
    """합격/불합격/스킵/미실행 4태 매핑 + 정직 스킵 문구."""
    from app.ui.gates import gate_cards
    report = {
        "hydrostatics": {"gm": 0.5, "freeboard_ok": True},
        "maxbox": {"feasible": True},
        "seakeeping": {"passed": False},
        "structure": {"passed": True, "skipped": False},
        "maneuvering": {"passed": True, "skipped": True,
                        "note": "조종 게이트 스킵 — L<20 m"},
        "economics": None,
        "passed": False,
    }
    cards = {c["name"]: c for c in gate_cards(report)}
    assert cards["내항"]["state"] == "fail"
    assert cards["구조"]["state"] == "pass"
    assert cards["조종"]["state"] == "skip"
    assert "≠ 불합격" in cards["조종"]["detail"]
    assert cards["경제"]["state"] == "off"
```

- [ ] **Step 2: 실패 확인** → FAIL (`gate_cards` 미정의)
- [ ] **Step 3: 구현**

```python
"""리포트 dict → 8중 게이트 신호등 카드 (streamlit 비의존)."""
from __future__ import annotations

_SKIP_NOTE = "정직 스킵 (데이터 없음 ≠ 불합격)"


def _state(entry) -> tuple[str, str]:
    if entry is None:
        return "off", "이번 실행에서 꺼짐"
    if isinstance(entry, dict):
        if entry.get("skipped"):
            return "skip", f"{_SKIP_NOTE} — {entry.get('note', '')}"
        if "passed" in entry:
            return ("pass" if entry["passed"] else "fail",
                    entry.get("note", ""))
        if "feasible" in entry:
            return ("pass" if entry["feasible"] else "fail", "")
    return "pass", ""


def gate_cards(report: dict) -> list[dict]:
    """8중 게이트 순서 고정 — 홈 화면 신호등."""
    hydro = report.get("hydrostatics")
    hydro_entry = (None if hydro is None
                   else {"passed": bool(hydro.get("gm", 0) > 0
                                        and hydro.get("freeboard_ok",
                                                      True))})
    space = report.get("space_large") or report.get("maxbox")
    items = [
        ("부양·복원", hydro_entry), ("공간", space),
        ("내항", report.get("seakeeping")),
        ("구조", report.get("structure")),
        ("조종", report.get("maneuvering")),
        ("경제", report.get("economics")),
    ]
    cards = []
    for name, entry in items:
        state, detail = _state(entry)
        cards.append({"name": name, "state": state, "detail": detail})
    return cards
```

(주: 시험의 카드 이름 "내항"·"구조"·"조종"·"경제"에 맞춰 name 키 사용 — 시험과 구현의 이름 목록은 동일해야 함. 8중 표기는 화면 쪽 제목에서.)

- [ ] **Step 4: 통과 확인** → PASS (기존 mesh 시험 포함 전부)
- [ ] **Step 5: 커밋** — `git commit -m "feat: 앱 ui — 게이트 신호등 가공"`

### Task 3: `app/ui/cards.py` — 성적표 카드 + 용어 툴팁

**Files:**
- Create: `app/ui/cards.py`
- Test: `tests/test_app_ui.py` (추가)

**Interfaces:**
- Consumes: 리포트 dict (dimensions/hydrostatics/resistance/motor|engine/economics 하위 키)
- Produces: `scorecards(report: dict) -> list[dict]` — 각 `{"label": str, "value": str, "help": str}` (값 없으면 항목 생략 — KeyError 금지)

- [ ] **Step 1: 실패 시험**

```python
def test_scorecards_tolerates_missing():
    """있는 값만 카드로 — 소형(경제 없음)에서도 죽지 않음."""
    from app.ui.cards import scorecards
    report = {
        "dimensions": {"loa": 3.0, "beam": 0.9, "draft": 0.3},
        "hydrostatics": {"gm": 0.15},
        "resistance": {"total": 12.5},
    }
    cards = scorecards(report)
    labels = [c["label"] for c in cards]
    assert "길이 LOA" in labels and "GM" in labels
    gm = next(c for c in cards if c["label"] == "GM")
    assert "복원" in gm["help"]          # 학부 눈높이 툴팁
    assert all(c["value"] for c in cards)
```

- [ ] **Step 2: 실패 확인** → FAIL
- [ ] **Step 3: 구현**

```python
"""리포트 → 성적표 카드 (streamlit 비의존) — 학부 2학년 툴팁."""
from __future__ import annotations

_SPECS = [
    (("dimensions", "loa"), "길이 LOA", "{:.2f} m",
     "배 전체 길이 (Length Over All)"),
    (("dimensions", "beam"), "폭", "{:.2f} m", "배의 최대 너비"),
    (("dimensions", "draft"), "흘수", "{:.2f} m",
     "물에 잠기는 깊이"),
    (("hydrostatics", "gm"), "GM", "{:.2f} m",
     "복원력 지표 — 클수록 덜 뒤집힘 (너무 크면 급한 흔들림)"),
    (("resistance", "total"), "저항", "{:,.0f} N",
     "이 속도로 밀고 갈 때 물이 미는 힘"),
    (("engine", "name"), "엔진", "{}", "선정된 실물 엔진"),
    (("economics", "attained_g_per_tnm"), "EEDI", "{:.2f} gCO₂/t·nm",
     "설계 탄소 효율 — 규제 상한 이하여야 합격"),
    (("economics", "cii", "rating_2026"), "CII 2026", "{}",
     "운항 탄소 등급 (A 최고 ~ E)"),
    (("economics", "transport_usd_per_tnm"), "수송단가",
     "{:.4f} $/t·nm", "짐 1톤을 1해리 나르는 연료비"),
]


def _dig(d: dict, path: tuple):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return None
        cur = cur[k]
    return cur


def scorecards(report: dict) -> list[dict]:
    cards = []
    for path, label, fmt, help_txt in _SPECS:
        v = _dig(report, path)
        if v is not None:
            cards.append({"label": label, "value": fmt.format(v),
                          "help": help_txt})
    return cards
```

- [ ] **Step 4: 통과 확인** → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: 앱 ui — 성적표 카드"`

### Task 4: `app/main.py` + `app/pages/1_라이브_설계.py` — 화면 배선

**Files:**
- Create: `app/main.py`, `app/pages/1_라이브_설계.py`
- Test: `tests/test_app_ui.py` (임포트 스모크 추가)

**Interfaces:**
- Consumes: Task 1~3 함수 + `run_pipeline(goal, out_dir, hull_source=..., catamaran=...)` + `GoalSpec(target_speed_ms, payload_kg, purpose)`
- Produces: `streamlit run app/main.py` 구동 화면

- [ ] **Step 1: 스모크 시험** (streamlit 파일은 임포트만 검증 — 렌더는 수동)

```python
def test_app_pages_importable():
    """페이지 모듈이 임포트 오류 없이 로드 (streamlit 스크립트는
    bare 실행 구조라 ast 파싱으로 문법만 — 실행은 수동 검증)."""
    import ast
    from pathlib import Path
    for p in ["app/main.py", "app/pages/1_라이브_설계.py"]:
        ast.parse(Path(p).read_text(), filename=p)
```

- [ ] **Step 2: 실패 확인** → FAIL (파일 없음)
- [ ] **Step 3: 구현** — `app/main.py`:

```python
"""선박 AI 설계 — 시연 앱 홈 (로컬 전용, 공개 배포 금지)."""
import streamlit as st

st.set_page_config(page_title="선박 AI 설계", page_icon="🚢",
                   layout="wide")
st.title("🚢 선박 AI 설계 파이프라인")
st.markdown(
    "**\"이런 배가 필요해요\" → 물리 법칙으로 검증된 배.**\n\n"
    "왼쪽 **라이브 설계**에서 속도·짐·용도 3개만 고르면 8중 관문"
    " (뜨나·실리나·안 넘어지나·안 흔들리나·안 부러지나·도나·규제"
    "·성적)을 통과한 배가 설계됩니다.\n\n"
    "- 시험 441개 전부 통과한 물리 엔진 재사용 — 이 앱은 화면만\n"
    "- 로컬 전용 (생성 형상 비공개 관례)")
```

`app/pages/1_라이브_설계.py`:

```python
"""라이브 설계 — 3입력 → run_pipeline → 신호등·성적표·3D."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from app.ui.cards import scorecards
from app.ui.gates import gate_cards
from app.ui.mesh_view import mesh_to_plotly

st.set_page_config(page_title="라이브 설계", page_icon="⚙️",
                   layout="wide")
st.title("⚙️ 라이브 설계")

PURPOSES = {"조사선 (survey)": "survey", "경비정 (patrol)": "patrol",
            "작업선 (workboat)": "workboat", "화물선 (cargo)": "cargo"}

with st.sidebar:
    purpose_label = st.radio("용도", list(PURPOSES))
    purpose = PURPOSES[purpose_label]
    speed = st.slider("목표 속도 [m/s]", 0.5, 8.0,
                      6.0 if purpose == "cargo" else 1.5, 0.1)
    payload_t = st.slider(
        "짐 [kg]", 1.0, 2e7,
        8e6 if purpose == "cargo" else 50.0,
        format="%.0f")
    catamaran = st.toggle("쌍동선", value=False)
    with st.expander("고급"):
        hull_source = st.selectbox("선형 소스",
                                   ["auto", "shipd", "formula"])
    run = st.button("설계 실행", type="primary",
                    use_container_width=True)

if run:
    from src.core.types import GoalSpec
    from src.pipeline import run_pipeline
    goal = GoalSpec(target_speed_ms=float(speed),
                    payload_kg=float(payload_t), purpose=purpose)
    try:
        with st.status("설계 나선 실행 중…", expanded=True) as status:
            st.write("치수 추정 → 선형 생성 → 정역학 → 저항 → "
                     "추진 → 8중 게이트")
            with tempfile.TemporaryDirectory() as td:
                report = run_pipeline(goal, td,
                                      hull_source=hull_source,
                                      catamaran=catamaran)
                import trimesh
                mesh = trimesh.load(Path(td) / "hull.stl")
            status.update(label="완료", state="complete")
        st.session_state["report"] = report
        st.session_state["mesh3d"] = mesh_to_plotly(mesh)
    except Exception as e:
        st.error(f"설계 실패 — {type(e).__name__}: {e}")

if "report" in st.session_state:
    report = st.session_state["report"]
    verdict = report.get("passed")
    st.subheader("판정: " + ("✅ 합격" if verdict else "❌ 불합격"))
    cols = st.columns(6)
    icon = {"pass": "🟢", "fail": "🔴", "skip": "⚪", "off": "⚫"}
    for col, card in zip(cols * 2, gate_cards(report)):
        col.metric(f"{icon[card['state']]} {card['name']}",
                   {"pass": "합격", "fail": "불합격",
                    "skip": "스킵", "off": "꺼짐"}[card["state"]],
                   help=card["detail"] or None)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("3D 선형")
        d = st.session_state["mesh3d"]
        fig = go.Figure(go.Mesh3d(**d, color="steelblue",
                                  opacity=0.9))
        fig.update_layout(scene_aspectmode="data",
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("성적표")
        for c in scorecards(report):
            st.metric(c["label"], c["value"], help=c["help"])
    with st.expander("리포트 원문 (JSON)"):
        st.json({k: v for k, v in report.items()
                 if k not in ("mesh_file",)})
```

- [ ] **Step 4: 통과 확인** — 시험 전부 PASS + 수동: `PYTHONPATH="$PWD" /opt/anaconda3/bin/python -m streamlit run app/main.py --server.headless true` 구동, 브라우저 확인 (오너 눈 = 정식 검증)
- [ ] **Step 5: 커밋** — `git commit -m "feat: Streamlit 라이브 설계 페이지 — 3입력·신호등·성적표·3D"`

### Task 5: 회귀 + 4축 마감

- [ ] 전체 회귀 `/opt/anaconda3/bin/python -m pytest tests/ -q` (441 + 신규)
- [ ] README 실행법 추가 (`streamlit run app/main.py`), worklog·easy-manual·스펙 §7 상태 갱신, 메모리
- [ ] 커밋·푸시

## Self-Review

- 스펙 §3 커버: 입력 (용도 4종·슬라이더·쌍동 토글·고급 접기) ✓ 실행 (st.status 실황·session_state — st.cache_data는 TemporaryDirectory와 궁합 문제로 세션 보관으로 대체, 스펙 취지 = 재실행 방지 충족) ✓ 결과 4요소 (3D·신호등·성적표·실황 로그) ✓ 오류 st.error ✓
- §5 시험 전략: ui/ 순수 함수 3종 + 문법 스모크 ✓ (렌더 수동 — 스펙 명시)
- 타입 일관성: gate_cards 이름 목록 = 시험 키 ✓, mesh_to_plotly 키 = Mesh3d 인자 ✓
- 잔여: 갤러리 페이지 = 2단계 회차 (스펙 §4 — 이 플랜 범위 밖)
