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
    assert cards["부양·복원"]["state"] == "pass"
    assert cards["공간"]["state"] == "pass"


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


def test_app_pages_importable():
    """페이지 모듈 문법 검증 (streamlit 스크립트는 bare 실행
    구조라 ast 파싱만 — 렌더는 수동 브라우저 검증 몫)."""
    import ast
    from pathlib import Path
    for p in ["app/main.py", "app/pages/1_라이브_설계.py",
              "app/pages/2_갤러리.py"]:
        ast.parse(Path(p).read_text(), filename=p)


def test_pareto_front_loader_missing_dir(tmp_path):
    """전선 로더 — 자산 없으면 정직 None (앱은 안내문 표시)."""
    from app.ui.pareto import load_verified_front
    assert load_verified_front(tmp_path) is None


def test_pareto_front_loader_merges_and_filters(tmp_path):
    """evo(설계값) + recheck(판정) 병합 — 완전 검증만 남김."""
    import pandas as pd
    from app.ui.pareto import load_verified_front
    pd.DataFrame({
        "speed_ms": [5.0, 6.0], "loa": [99.0, 99.0],
        "cb": [0.48, 0.47], "transport_usd_per_tnm": [4e-4, 9e-4],
        "vector_json": ["[1]", "[2]"],
    }).to_csv(tmp_path / "pareto_large_v4_evo.csv", index=False)
    pd.DataFrame({
        "idx": [0, 1], "full_passed": [True, False],
        "sk_roll_rms": [0.3, 0.4], "mv_dt": [4.9, 5.2],
    }).to_csv(tmp_path / "pareto_v4_full_recheck.csv", index=False)
    df = load_verified_front(tmp_path)
    assert len(df) == 1                    # 완전 검증만
    assert df.iloc[0]["speed_ms"] == 5.0
    assert "vector_json" not in df.columns  # 라이선스 — 화면 비노출
