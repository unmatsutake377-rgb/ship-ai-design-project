"""목적 → 주요 치수 추정 (spec §2.2, Step 1).

상선 경험식(Watson 등)은 20 m 이상 회귀식이라 USV 스케일에서 무효 —
공개 소형선/USV 제원에서 뽑은 용도별 비율 밴드를 사용한다.
(대표 참고 제원: 조사용 2~5 m급 쌍동/단동 USV, 항만 순찰정,
 소형 작업선의 L/B·B/T·Cb 통상 범위. PoC용 개략값.)

역산: W_est = payload / payload_fraction, ∇ = W_est/ρ,
      ∇ = Cb·L·B·T, B = L/r_LB, T = B/r_BT
      ⇒ L = (∇ · r_LB² · r_BT / Cb)^(1/3)
"""
from __future__ import annotations

from dataclasses import dataclass

from src.core.types import GoalSpec, MainDimensions

RHO_SEAWATER = 1025.0
DEPTH_OVER_DRAFT = 1.6  # 형심/설계흘수 (건현 여유)


class UnknownPurposeError(ValueError):
    pass


@dataclass(frozen=True)
class RatioBand:
    lb: float                # L/B
    bt: float                # B/T
    cb: float                # 방형계수 목표
    payload_fraction: float  # 적재량/전체 배수량


PURPOSE_BANDS: dict[str, RatioBand] = {
    # 조사용: 안정성 우선, 통통한 선형
    "survey": RatioBand(lb=3.0, bt=4.0, cb=0.50, payload_fraction=0.35),
    # 순찰용: 상대적으로 날씬
    "patrol": RatioBand(lb=3.5, bt=4.5, cb=0.45, payload_fraction=0.25),
    # 작업선: 적재 능력 우선
    "workboat": RatioBand(lb=2.8, bt=3.5, cb=0.55, payload_fraction=0.45),
}


def estimate_dimensions(goal: GoalSpec,
                        rho: float = RHO_SEAWATER) -> MainDimensions:
    band = PURPOSE_BANDS.get(goal.purpose)
    if band is None:
        raise UnknownPurposeError(
            f"'{goal.purpose}'는 지원 용도가 아닙니다. "
            f"지원 용도: {sorted(PURPOSE_BANDS)}"
        )
    total_mass = goal.payload_kg / band.payload_fraction
    volume = total_mass / rho
    loa = (volume * band.lb ** 2 * band.bt / band.cb) ** (1.0 / 3.0)
    beam = loa / band.lb
    draft = beam / band.bt
    return MainDimensions(
        loa=loa, beam=beam,
        depth=DEPTH_OVER_DRAFT * draft,
        draft_design=draft, cb=band.cb,
    )
