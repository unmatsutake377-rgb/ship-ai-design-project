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
    """있는 값만 카드로 (KeyError 금지 — 소형/대형 리포트 공용)."""
    cards = []
    for path, label, fmt, help_txt in _SPECS:
        v = _dig(report, path)
        if v is not None:
            cards.append({"label": label, "value": fmt.format(v),
                          "help": help_txt})
    return cards
