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

# 세로 방향 배치 (선체 중앙 기준, +선수 방향) — 분포모델 (오너 Q4)
LCG_STRUCTURE_OVER_L = 0.0     # 선각은 선수미 대칭 (Wigley)
LCG_PAYLOAD_OVER_L = 0.0       # 중앙 탑재 가정
LCG_PROPULSION_OVER_L = -0.45  # 모터·배터리는 선미
KZZ_OVER_L = 0.25              # 구조 선회 회전반경/L (통상값)
TRIM_WARN_LCG_OVER_L = 0.01    # |LCG| 이 비율 초과 시 트림 경고
                               # (Wigley는 LCB=0 — LCG가 벗어나면 트림 발생)


@dataclass(frozen=True)
class WeightEstimate:
    total_mass: float
    structure_mass: float
    propulsion_mass: float
    payload_mass: float
    kg: float                # 무게중심 높이 (킬 기준) [m]
    lcg: float = 0.0         # 무게중심 세로 위치 (중앙 기준, +선수) [m]
    izz: float = 0.0         # 선회(요) 관성모멘트 [kg·m²]
    trim_warning: bool = False
    assumptions: dict = None


def estimate_weights(hull_area_m2: float, depth: float, payload_kg: float,
                     propulsion_mass_kg: float | None = None,
                     loa: float = 0.0) -> WeightEstimate:
    """중량·KG·LCG·Izz 추정 (분포모델 — 점질량 3개 + 구조 회전반경).

    propulsion_mass_kg 지정 시 실측(선택 모터+배터리 계산값)을 사용 —
    설계 나선 반복용. 미지정 시 고정비율 개략(초기 추정용).
    loa 미지정(0)이면 세로 분포 항은 0으로 퇴화 (구버전 호환).
    """
    structure = hull_area_m2 * AREAL_DENSITY_KG_M2 * OUTFIT_FACTOR
    if propulsion_mass_kg is None:
        total = (structure + payload_kg) / (1.0 - PROPULSION_FRACTION)
        propulsion = PROPULSION_FRACTION * total
    else:
        propulsion = propulsion_mass_kg
        total = structure + payload_kg + propulsion

    vcg_s = VCG_STRUCTURE_OVER_D * depth
    vcg_p = VCG_PAYLOAD_OVER_D * depth
    vcg_m = VCG_PROPULSION_OVER_D * depth
    kg = (structure * vcg_s + payload_kg * vcg_p + propulsion * vcg_m) / total

    x_s = LCG_STRUCTURE_OVER_L * loa
    x_p = LCG_PAYLOAD_OVER_L * loa
    x_m = LCG_PROPULSION_OVER_L * loa
    lcg = (structure * x_s + payload_kg * x_p + propulsion * x_m) / total
    izz = (structure * (KZZ_OVER_L * loa) ** 2
           + structure * x_s ** 2
           + payload_kg * x_p ** 2
           + propulsion * x_m ** 2)
    trim_warning = bool(loa > 0 and abs(lcg) > TRIM_WARN_LCG_OVER_L * loa)

    return WeightEstimate(
        total_mass=total,
        structure_mass=structure,
        propulsion_mass=propulsion,
        payload_mass=payload_kg,
        kg=kg,
        lcg=lcg,
        izz=izz,
        trim_warning=trim_warning,
        assumptions={
            "areal_density": AREAL_DENSITY_KG_M2,
            "outfit_factor": OUTFIT_FACTOR,
            "propulsion_fraction": PROPULSION_FRACTION,
            "vcg_structure": vcg_s,
            "vcg_payload": vcg_p,
            "vcg_propulsion": vcg_m,
            "lcg_propulsion": x_m,
            "kzz_over_l": KZZ_OVER_L,
        },
    )
