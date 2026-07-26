"""파이프라인 공용 데이터 타입.

모든 모듈은 이 타입으로만 대화한다 (spec §2.1).
ai ↛ physics ↛ ros2 직접 의존 금지.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GoalSpec:
    """사용자 운항 목적. 파이프라인의 유일한 입력."""

    target_speed_ms: float  # 목표 순항 속도 [m/s]
    payload_kg: float       # 적재량 [kg]
    purpose: str            # "survey" | "patrol" | "workboat"
    endurance_h: float = 4.0  # 항속시간 [h] — 배터리 크기 산정 근거


@dataclass(frozen=True)
class MainDimensions:
    """주요 치수. draft_design은 형상 정의용 설계 흘수 —
    실제 운항 흘수는 physics.hydrostatics가 중량에서 역산한다."""

    loa: float           # 전장 L [m]
    beam: float          # 최대폭 B [m]
    depth: float         # 형심 D [m]
    draft_design: float  # 설계 흘수 T [m]
    cb: float            # 방형계수 목표값


@dataclass
class ShipDesign:
    """설계 1건의 전체 상태. 단계가 진행되며 extras에 결과가 쌓인다."""

    goal: GoalSpec
    dims: MainDimensions
    mesh_path: str | None = None
    extras: dict = field(default_factory=dict)
