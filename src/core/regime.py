"""속도 체계 판정 (spec §2.1).

세 체계: 배수량형 / 반배수량형 / 활주형.
Phase A는 배수량형만 지원 — 나머지는 명시적 중단 (조용한 외삽 금지).
"""
from __future__ import annotations

import math
from enum import Enum, auto

G = 9.81  # 중력가속도 [m/s^2]

FN_DISPLACEMENT_MAX = 0.40  # 길이 Froude 수 상한 (배수량형)
FNV_PLANING_MIN = 3.0       # 용적 Froude 수 하한 (활주형)


class Regime(Enum):
    DISPLACEMENT = auto()       # 배수량형: Phase A 구현
    SEMI_DISPLACEMENT = auto()  # 반배수량형: 2차 사이클
    PLANING = auto()            # 활주형: Phase C


class UnsupportedRegimeError(NotImplementedError):
    def __init__(self, regime: Regime, message: str):
        self.regime = regime
        super().__init__(message)


_REGIME_KO = {
    Regime.SEMI_DISPLACEMENT: "반배수량형 (Fn 0.4~, 2차 사이클에서 지원 예정)",
    Regime.PLANING: "활주형 (Fn∇ ≥ 3, Phase C에서 지원 예정)",
}


def froude_length(speed_ms: float, loa: float) -> float:
    """길이 Froude 수 Fn = V / sqrt(g·L)."""
    return speed_ms / math.sqrt(G * loa)


def froude_volumetric(speed_ms: float, volume_m3: float) -> float:
    """용적 Froude 수 Fn∇ = V / sqrt(g·∇^(1/3)). 활주 판정용."""
    return speed_ms / math.sqrt(G * volume_m3 ** (1.0 / 3.0))


def classify(speed_ms: float, loa: float, volume_m3: float) -> Regime:
    fn = froude_length(speed_ms, loa)
    fnv = froude_volumetric(speed_ms, volume_m3)
    if fnv >= FNV_PLANING_MIN:
        return Regime.PLANING
    if fn < FN_DISPLACEMENT_MAX:
        return Regime.DISPLACEMENT
    return Regime.SEMI_DISPLACEMENT


def max_displacement_speed(loa: float) -> float:
    """이 길이의 선체가 배수량형으로 낼 수 있는 속도 상한 [m/s].

    hull speed: 자기가 만든 파도의 길이가 배 길이와 같아지는 속도(Fn≈0.4)
    부터 저항이 급증 — 긴 배일수록 빠를 수 있다.
    """
    return FN_DISPLACEMENT_MAX * math.sqrt(G * loa)


def min_loa_for_speed(speed_ms: float) -> float:
    """이 속도를 배수량형으로 내려면 필요한 최소 선체 길이 [m] (역함수)."""
    return (speed_ms / FN_DISPLACEMENT_MAX) ** 2 / G


def require_supported(regime: Regime) -> None:
    """Phase A 미구현 체계면 명시적으로 중단한다."""
    if regime is not Regime.DISPLACEMENT:
        raise UnsupportedRegimeError(
            regime,
            f"현재 버전은 배수량형(Fn < {FN_DISPLACEMENT_MAX})만 지원합니다. "
            f"요청된 체계: {_REGIME_KO[regime]}. "
            "목표 속도를 낮추거나 더 긴 선체를 허용해 주세요.",
        )
