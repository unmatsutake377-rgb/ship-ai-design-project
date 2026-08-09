"""대형 화물창 용적 게이트 (스펙 2026-08-09-cargo-capacity).

8중 게이트의 space 축이 대형 분기에 비어 있던 구멍을 메운다 —
NSGA 캠페인 정직 각주("날씬 전선이 짐을 실을 수 있는지 미검")의
해소. 소형은 MaxBox(다구획) 기존 유지.

용적 사슬 (전부 메쉬 실측 절단 — immersed_volume 계보 재사용):
  화물창 = [전체 내부(z≤depth) − 이중저(z<h_db) − 기관실(선미 15%,
  이중저 위) − 연료 탱크] × 0.90 (구조 부재·통로 공제)

관례 상수 (C급 정직 표기 — 수집 후보):
- 이중저 h_db = B/15, 0.76~2.0 m 클램프 (SOLAS II-1 계보 —
  원문 유료, 공개 2차 문헌 관례값)
- 기관실 = 선미 15% 구간 (중량 블록 machinery 0.10~0.30 배치 정합)
- 연료 밀도 0.9 t/m³, grain 공제 0.90
- 요구 용적 = payload × 적재계수(stowage factor) 1.3 m³/t —
  일반화물 실선 대역 1.2~1.5 (C급 관례; 초안의 "소형 밀도 500
  재사용"은 과대 요구 오판이라 정정 — 500은 소형 장비 환산 전용)
"""
from __future__ import annotations

import trimesh

from src.physics.hydrostatics import immersed_volume

RHO_FUEL_T_M3 = 0.9
GRAIN_FACTOR = 0.90
STOWAGE_M3_PER_T = 1.3        # 일반화물 적재계수 (실선 1.2~1.5 C급)
ENGINE_ROOM_FRAC = 0.15


def double_bottom_height_m(beam: float) -> float:
    """이중저 높이 — B/15, 0.76~2.0 m 클램프 (SOLAS 계보 C급)."""
    return min(2.0, max(0.76, beam / 15.0))


def hold_volume_large(mesh: trimesh.Trimesh, depth: float,
                      loa: float, fuel_t: float) -> dict:
    """화물창 용적 [m³] — 메쉬 실측 절단 사슬."""
    zmin = float(mesh.bounds[0][2])
    beam = float(mesh.bounds[1][1] - mesh.bounds[0][1])
    gross = immersed_volume(mesh, zmin + depth)          # 전체 내부
    h_db = double_bottom_height_m(beam)
    v_db = immersed_volume(mesh, zmin + min(h_db, depth))

    # 기관실: 선미 15% 구간 (이중저 위) — x 절단 후 부피 실측
    xmin = float(mesh.bounds[0][0])
    x_cut = xmin + ENGINE_ROOM_FRAC * loa
    aft = mesh.slice_plane([x_cut, 0, 0], [-1, 0, 0], cap=True)
    if aft is not None and aft.volume > 0:
        aft_total = immersed_volume(aft, zmin + depth)
        aft_db = immersed_volume(aft, zmin + min(h_db, depth))
        v_engine = max(aft_total - aft_db, 0.0)
    else:
        v_engine = 0.0

    v_fuel = fuel_t / RHO_FUEL_T_M3
    hold = max(gross - v_db - v_engine - v_fuel, 0.0) * GRAIN_FACTOR
    return {"gross_m3": gross, "double_bottom_m3": v_db,
            "engine_room_m3": v_engine, "fuel_tank_m3": v_fuel,
            "hold_m3": hold,
            "note": "이중저 B/15·기관실 15%·grain 0.90 — C급 관례"}


def space_gate_large(mesh: trimesh.Trimesh, depth: float, loa: float,
                     fuel_t: float, payload_t: float) -> dict:
    """대형 space 게이트 — 화물창 용적 ≥ payload × 적재계수."""
    hv = hold_volume_large(mesh, depth, loa, fuel_t)
    required = payload_t * STOWAGE_M3_PER_T
    return {**hv, "required_m3": required,
            "margin_ratio": hv["hold_m3"] / max(required, 1e-9),
            "passed": bool(hv["hold_m3"] >= required)}
