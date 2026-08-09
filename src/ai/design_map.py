"""설계 지도 — 8중 게이트를 탐색 축으로 (스펙 2026-08-09-design-map).

속도 × 짐 격자를 평가해 "합격 설계들의 지도"를 만들고, 합격 중
수송단가 최소를 추천한다 — 원안 "AI 주도 목적 지향"의 종합.

fast=True: 느린 게이트(내항·조종) 생략 — 탐색 전용. 최종 추천은
전체 게이트(fast=False)로 재검하는 2단 구도 (스펙 §2).
설계 불가(엔진 한계 등)는 정직 기록 — 지도의 정보다.
"""
from __future__ import annotations

import tempfile

from src.core.types import GoalSpec


def evaluate_design(payload_kg: float, speed_ms: float,
                    purpose: str = "cargo",
                    fast: bool = False) -> dict:
    """한 설계점 평가 → 핵심 지표. 실패는 error 필드로 정직 반환."""
    from src.pipeline import run_pipeline
    base = {"payload_kg": payload_kg, "speed_ms": speed_ms}
    try:
        with tempfile.TemporaryDirectory() as d:
            report = run_pipeline(
                GoalSpec(target_speed_ms=speed_ms,
                         payload_kg=payload_kg, purpose=purpose),
                d,
                seakeeping=not fast,
                maneuvering=not fast,
            )
    except Exception as e:
        return {**base, "passed": None, "error":
                f"{type(e).__name__}: {e}",
                "transport_usd_per_tnm": None,
                "eedi_margin_pct": None}
    ec = report.get("economics") or {}
    cii = ec.get("cii") or {}
    return {
        **base,
        "loa_m": report["dimensions"]["loa"],
        "passed": bool(report["passed"]),
        "gates": {
            "structure": _gate_state(report.get("structure")),
            "seakeeping": _gate_state(report.get("seakeeping")),
            "maneuvering": _gate_state(report.get("maneuvering")),
            "economics": _gate_state(ec),
        },
        "eedi_margin_pct": ec.get("margin_pct"),
        "cii_rating_2026": cii.get("rating_2026"),
        "fuel_cost_usd_per_year": ec.get("fuel_cost_usd_per_year"),
        "transport_usd_per_tnm": ec.get("transport_usd_per_tnm"),
        "fast": fast,
    }


def _gate_state(rep) -> str:
    if rep is None:
        return "생략"
    if rep.get("skipped"):
        return "스킵"
    return "합격" if rep.get("passed") else "불합격"


def design_map(payloads_kg, speeds_ms, fast: bool = True) -> list:
    """격자 순회 — 설계 지도."""
    out = []
    for p in payloads_kg:
        for v in speeds_ms:
            out.append(evaluate_design(p, v, fast=fast))
    return out


def recommend(results: list) -> dict | None:
    """전 게이트 합격 중 수송단가 최소 (동률 시 EEDI 여유 큰 쪽).

    합격 없음 = None (정직 — 요구 조합 재검토 신호)."""
    ok = [r for r in results if r.get("passed") is True
          and r.get("transport_usd_per_tnm") is not None]
    if not ok:
        return None
    return min(ok, key=lambda r: (r["transport_usd_per_tnm"],
                                  -(r.get("eedi_margin_pct") or 0.0)))
