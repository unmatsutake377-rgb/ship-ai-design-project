"""리포트 dict → 8중 게이트 신호등 카드 (streamlit 비의존).

3태 + 미실행: pass(초록)·fail(빨강)·skip(회색 — 정직 스킵 관례
"데이터 없음 ≠ 불합격")·off(게이트 플래그로 꺼짐).
"""
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
                    str(entry.get("note", "")))
        if "feasible" in entry:
            return ("pass" if entry["feasible"] else "fail", "")
        return "pass", "성적표 전용 (합불 판정 없음 — 소형 전기 등가 등)"
    return "pass", ""


def gate_cards(report: dict) -> list[dict]:
    """8중 게이트 순서 고정 신호등 — 각 {name, state, detail}."""
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
