"""쌍동선(카타마란) 기하·저항 (1단계, 스펙 2026-08-10).

기하: 데미헐 = 기존 generate_hull_mesh (전폭의 데미헐 비율로 폭
재배분), y = ±s/2 복제 병합. 병합 메쉬는 기존 정역학(evaluate)이
그대로 소화 — waterplane_properties가 폐곡선 합산이라 **평행축
효과 자동** (I_T = 2×(I_own + A·(s/2)²), 손계산 앵커 실증).

저항: 데미헐 단동 평가 × 2 — **선체 간 파 간섭 무시 (C급 정직
각주)**. USV 통상 간격(s/L 0.2~0.4)에서 간섭은 저속 Fn<0.3 기준
수 % 대역 (Molland 계열 문헌 확보 시 승급 백로그).

관례: 데미헐 폭 비율 0.25 (전폭의 1/4 — BlueBoat 계보 실측 대역),
간격비 separation_ratio = s/전폭.
"""
from __future__ import annotations

import trimesh

from src.ai.hull_generator import generate_hull_mesh
from src.core.types import MainDimensions

DEMIHULL_BEAM_FRAC = 0.25     # 데미헐 폭 / 전폭 (BlueBoat 계보)


def demihull_dims(dims: MainDimensions) -> MainDimensions:
    """전체 치수 → 데미헐 치수 (폭만 재배분)."""
    return MainDimensions(
        loa=dims.loa, beam=dims.beam * DEMIHULL_BEAM_FRAC,
        depth=dims.depth, draft_design=dims.draft_design,
        cb=dims.cb)


def generate_catamaran_mesh(dims: MainDimensions,
                            separation_ratio: float = 0.7,
                            cm: float | None = None
                            ) -> trimesh.Trimesh:
    """쌍동 메쉬 — 데미헐 2개 병합 (중심 간격 s = ratio × 전폭)."""
    demi = generate_hull_mesh(demihull_dims(dims)) if cm is None \
        else generate_hull_mesh(demihull_dims(dims), cm=cm)
    s = separation_ratio * dims.beam
    left = demi.copy()
    left.apply_translation([0.0, -s / 2.0, 0.0])
    right = demi.copy()
    right.apply_translation([0.0, +s / 2.0, 0.0])
    return trimesh.util.concatenate([left, right])


def catamaran_resistance(dims: MainDimensions,
                         separation_ratio: float,
                         draft: float, speed_ms: float) -> dict:
    """쌍동 저항 [N] — 데미헐 메쉬형 Michell × 2 (간섭 무시 C급)."""
    from src.physics.resistance import total_resistance_mesh
    demi = generate_hull_mesh(demihull_dims(dims))
    r = total_resistance_mesh(demi, dims.loa, draft, speed_ms)
    return {"demihull_n": r.total, "total_n": 2.0 * r.total,
            "rf_n": 2.0 * r.rf, "rw_n": 2.0 * r.rw,
            "note": "선체 간 파 간섭 무시 (C급) — Molland 문헌 확보"
                    " 시 승급 (통상 s/L 대역 수 % 오차)"}
