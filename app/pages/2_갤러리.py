"""갤러리 — 사전 계산 자산 (전선 탐사기·설계 지도·연대기·GIF)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from app.ui.pareto import load_verified_front

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="갤러리", page_icon="🖼️",
                   layout="wide")
st.title("🖼️ 갤러리 — 사전 계산 자산")

# ── 1. 파레토 전선 탐사기 ─────────────────────────────────────
st.header("파레토 전선 탐사기")
st.caption("NSGA 4차 진화 (5.1만 평가) → 전체 8중 재검 완전 검증 "
           "전선 — 수송단가 vs 속도. 점에 마우스를 올리면 상세.")
front = load_verified_front(ROOT / "outputs")
if front is None:
    st.info("전선 CSV 없음 — outputs/는 로컬 전용 자산입니다. "
            "`optimize_large` 실행 후 다시 열어주세요 (정직 안내).")
else:
    hover = [
        (f"Cb {r.cb:.3f} · LOA {r.loa:.0f} m<br>"
         f"roll RMS {r.sk_roll_rms:.2f}° · 선회 {r.mv_dt:.2f}L<br>"
         f"EEDI 여유 {r.eedi_margin_pct:+.1f}%")
        for r in front.itertuples()
    ]
    fig = go.Figure(go.Scatter(
        x=front["speed_ms"] * 1.9438,
        y=front["transport_usd_per_tnm"] * 1000.0,
        mode="markers", text=hover, hoverinfo="text",
        marker=dict(size=9, color=front["cb"],
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Cb"))))
    best = front.loc[front["transport_usd_per_tnm"].idxmin()]
    fig.add_trace(go.Scatter(
        x=[best["speed_ms"] * 1.9438],
        y=[best["transport_usd_per_tnm"] * 1000.0],
        mode="markers", name="추천 경제형",
        hoverinfo="skip",
        marker=dict(symbol="star", size=22, color="gold",
                    line=dict(color="red", width=2))))
    fig.update_layout(xaxis_title="속도 [kn]",
                      yaxis_title="수송단가 [$/kt·nm]",
                      height=450, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.success(
        f"추천 경제형: {best['speed_ms'] * 1.9438:.1f} kn · "
        f"Cb {best['cb']:.3f} · "
        f"단가 {best['transport_usd_per_tnm'] * 1000:.3f} $/kt·nm")
    st.metric("완전 검증 전선", f"{len(front)}척",
              help="fast 6중 + full 재검(스트립 내항·MMG 조종) "
                   "전부 통과")

# ── 2. 설계 지도 ─────────────────────────────────────────────
st.header("설계 지도 (속도 × 짐)")
map_p = ROOT / "outputs" / "design_map_12.json"
if not map_p.exists():
    st.info("설계 지도 자산 없음 — `design_map` 실행 후 표시 "
            "(정직 안내).")
else:
    pts = json.loads(map_p.read_text())
    xs = [p["speed_ms"] for p in pts]
    ys = [p["payload_kg"] / 1000.0 for p in pts]
    cost = [p.get("transport_usd_per_tnm") for p in pts]
    ok = [p.get("passed") for p in pts]
    fig2 = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        text=[f"{c*1000:.2f}" if c else "✗" for c in cost],
        textposition="top center",
        marker=dict(size=14,
                    color=["#2a9d2a" if o else "#cc3333"
                           for o in ok])))
    fig2.update_layout(xaxis_title="속도 [m/s]",
                       yaxis_title="짐 [t]",
                       height=400, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("초록 = 8중 합격 (숫자 = 단가 $/kt·nm) · 빨강 = "
               "불합격. 규모의 경제·EEDI 경계가 보입니다.")

# ── 3. 항해일지 연대기 ───────────────────────────────────────
st.header("항해일지 연대기")
TIMELINE = [
    ("~7/24", "원안 6단계 스펙 — 목적 지향 설계 PoC 출발", 0),
    ("7월 말", "소형 USV: 선형 생성·정역학·저항·시뮬레이션", 150),
    ("8/3", "캠페인 5개의 날 — 러더·다구획·특징 v3", 270),
    ("8/5", "ShipGen 실선 생성기 채택 (혈통 있는 배)", 292),
    ("8/6", "전 크기 개방 — 100 m 화물선까지 같은 CLI", 321),
    ("8/7~8", "내항성 — 파도의 수학 (Lewis·Ursell·Frank·RAO)", 350),
    ("8/9", "캠페인 5개: 구조·조종·경제·CII·설계지도 → 8중", 416),
    ("8/10", "화물창·쌍동선·RMS 원전 승급 — 잣대 검거", 436),
    ("8/11", "조종 프록시·NSGA v4 — 완전 검증 신챔피언", 441),
]
for date, event, n_tests in TIMELINE:
    c1, c2, c3 = st.columns([1, 6, 1])
    c1.markdown(f"**{date}**")
    c2.markdown(event)
    c3.markdown(f"`{n_tests}` 시험" if n_tests else "—")

# ── 4. 파도 타는 배 ──────────────────────────────────────────
st.header("파도 타는 배")
gif_p = ROOT / "outputs" / "wave_exp" / "wave_ride.gif"
if gif_p.exists():
    st.image(str(gif_p),
             caption="스트립 이론 실측 heave·pitch + 해석 파면 — "
                     "Gazebo 독립 물리와 교차 검증된 응답")
else:
    st.info("파랑 GIF 자산 없음 (outputs/ 로컬 전용 — 정직 안내).")
