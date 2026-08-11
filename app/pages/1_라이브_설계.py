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
