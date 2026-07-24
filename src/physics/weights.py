"""중량·무게중심(KG) 추정 (spec §2.3 — v1 검토의 최대 누락 보완).

개략 모델 (PoC 수준, 모든 가정을 assumptions에 기록):
- 구조: 선각 표면적 × 면밀도(소형 GRP 외판+보강 개략) × 의장계수
- 추진+배터리: 전체 중량의 고정 비율 f
  → W = 구조 + f·W + 적재  ⇒  W = (구조 + 적재) / (1 − f)
  (저항 기반 파워 추정으로의 교체는 M3.5 이후)
- KG: 성분별 VCG 가정(형심 D 비율)의 가중 평균
"""
from __future__ import annotations

from dataclasses import dataclass

AREAL_DENSITY_KG_M2 = 8.0   # GRP 소형정 외판+보강 개략 면밀도 [kg/m²]
OUTFIT_FACTOR = 1.4         # 의장·접합·도장 여유 계수
PROPULSION_FRACTION = 0.15  # 추진기+배터리 / 전체 중량 (소형 전동 USV 개략)

VCG_STRUCTURE_OVER_D = 0.55   # 선각+갑판 무게중심 높이 / 형심
VCG_PAYLOAD_OVER_D = 0.60     # 탑재 장비 (선내 거치 가정)
VCG_PROPULSION_OVER_D = 0.25  # 모터·배터리 (선저 근처)


@dataclass(frozen=True)
class WeightEstimate:
    total_mass: float
    structure_mass: float
    propulsion_mass: float
    payload_mass: float
    kg: float                # 무게중심 높이 (킬 기준) [m]
    assumptions: dict


def estimate_weights(hull_area_m2: float, depth: float,
                     payload_kg: float) -> WeightEstimate:
    structure = hull_area_m2 * AREAL_DENSITY_KG_M2 * OUTFIT_FACTOR
    total = (structure + payload_kg) / (1.0 - PROPULSION_FRACTION)
    propulsion = PROPULSION_FRACTION * total

    vcg_s = VCG_STRUCTURE_OVER_D * depth
    vcg_p = VCG_PAYLOAD_OVER_D * depth
    vcg_m = VCG_PROPULSION_OVER_D * depth
    kg = (structure * vcg_s + payload_kg * vcg_p + propulsion * vcg_m) / total

    return WeightEstimate(
        total_mass=total,
        structure_mass=structure,
        propulsion_mass=propulsion,
        payload_mass=payload_kg,
        kg=kg,
        assumptions={
            "areal_density": AREAL_DENSITY_KG_M2,
            "outfit_factor": OUTFIT_FACTOR,
            "propulsion_fraction": PROPULSION_FRACTION,
            "vcg_structure": vcg_s,
            "vcg_payload": vcg_p,
            "vcg_propulsion": vcg_m,
        },
    )
