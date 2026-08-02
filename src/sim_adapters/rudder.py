"""러더(방향타) 물리 모형 (스펙 2026-08-03 §1) — 조타·추력의 이혼.

배경: 전진 전용 추력기 쌍은 "조타 = 추력" 결합이 기하적 필연
(|차동| ≤ 전진합) — 자기지속 루프의 뿌리 (배분 재설계 2안 계측 기각).
러더는 물살 속 양력판으로 조타 모멘트를 추력과 독립으로 생산한다.

모형: 표준 평판 양력 (소각 선형 + 실속 포화)
  L_r = ½·ρ·V²·A_r·C_L,  C_L = 2π·k·δ (|δ| ≤ δ_stall에서),
  N_r = −L_r·x_r  (x_r < 0 선미 → 양의 δ가 양의 요 모멘트)
핵심 특성: V² 항 — **물살 없으면 무력** (저속 조타는 차동이 보조,
구성 A의 존재 이유).

상수는 전부 개략 (실물 서보·러더 스펙 수집 #17 확장 전까지 — 정직
표기). 러더 유속 = 선속 근사 (프로펠러 후류 상호작용 무시, 스펙 §5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

RHO = 1025.0
RUDDER_MAX_RAD = math.radians(35.0)    # 최대 타각 (관행)
RUDDER_STALL_RAD = math.radians(25.0)  # 실속 시작 — 이상 각도는 포화
RUDDER_K = 0.9                         # 종횡비 보정 개략
RUDDER_AREA_FRAC = 0.10                # 러더 현폭 / 선체 길이 (개략)


@dataclass(frozen=True)
class RudderModel:
    area: float          # 러더 면적 [m²]
    x_pos: float         # 선체 중앙 기준 위치 [m] (선미 = 음수)
    k: float = RUDDER_K

    @staticmethod
    def for_vessel(loa: float, draft: float) -> "RudderModel":
        """선체 치수 → 개략 러더 (면적 = 흘수 × 0.1L)."""
        return RudderModel(area=draft * RUDDER_AREA_FRAC * loa,
                           x_pos=-0.48 * loa)


def rudder_moment(rudder: RudderModel, u: float, delta: float) -> float:
    """타각 δ [rad] → 요 모멘트 [N·m]. 실속 밖은 포화."""
    d_eff = max(-RUDDER_STALL_RAD, min(RUDDER_STALL_RAD, delta))
    cl = 2.0 * math.pi * rudder.k * d_eff
    lift = 0.5 * RHO * u * u * rudder.area * cl
    return -lift * rudder.x_pos
