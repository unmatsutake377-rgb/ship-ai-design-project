"""러더(방향타) 물리 모형 (스펙 2026-08-03 §1) — 조타·추력의 이혼.

배경: 전진 전용 추력기 쌍은 "조타 = 추력" 결합이 기하적 필연
(|차동| ≤ 전진합) — 자기지속 루프의 뿌리 (배분 재설계 2안 계측 기각).
러더는 물살 속 양력판으로 조타 모멘트를 추력과 독립으로 생산한다.

모형 (3단계에서 실측 공식으로 승격 — 출처 data/rudder_servo_specs.csv):
  양력 기울기 = Mandel(1967) 저종횡비 경험식
    CLα = 1.8π·ΛE / (√(ΛE²+4) + 1.8)   [Liu & Hekkenberg 2017, eq.(7)]
    ΛE = kΛ·ΛG — 루트가 선저에 밀착하면 거울 효과로 kΛ≈2
  실속각 = 개수면 15~20° (중앙 17.5° 채택) — 우리 모형은 프로펠러
    후류를 무시(러더 유속=선속)하므로 개수면 값이 일관된 선택.
    후류 모형을 추가하면 30~40°로 지연됨 (같은 논문).
  면적 = DNV(1975) A = (L·T/100)(1+25(B/L)²)
  타각 속도 한계 = 서보 실측 (Power HD 20kg·cm 방수, 0.16s/60°
    = 375°/s 무부하). 실선 SOLAS 하한 2.3°/s는 USV엔 비적용 참고.

남은 개략 (정직): 러더 유속 = 선속 (프로펠러 후류 무시), 서보
속도는 무부하값 (수중 부하 감속 미계측), 스톡 위치·힌지 토크 미모형.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

RHO = 1025.0
RUDDER_MAX_RAD = math.radians(35.0)     # 최대 타각 (실선 관행 ±35°)
RUDDER_STALL_RAD = math.radians(17.5)   # 개수면 실속 15~20° 중앙
RUDDER_K_LAMBDA = 2.0                   # 유효/기하 종횡비 — 선체 밀착 거울 효과
RUDDER_AR_GEOMETRIC = 1.5               # 기하 종횡비 (항양선 1.5~3 하단)
SERVO_RATE_RAD_S = math.radians(60.0 / 0.16)  # 375°/s 무부하 (Power HD 20kg)


def lift_slope_mandel(ar_effective: float) -> float:
    """Mandel(1967) 저종횡비 양력 기울기 [1/rad]."""
    return (1.8 * math.pi * ar_effective
            / (math.sqrt(ar_effective ** 2 + 4.0) + 1.8))


def rudder_area_dnv(loa: float, beam: float, draft: float) -> float:
    """DNV(1975) 러더 면적 [m²]: (L·T/100)·(1+25(B/L)²)."""
    return (loa * draft / 100.0) * (1.0 + 25.0 * (beam / loa) ** 2)


@dataclass(frozen=True)
class RudderModel:
    area: float          # 러더 면적 [m²]
    x_pos: float         # 선체 중앙 기준 위치 [m] (선미 = 음수)
    ar_geometric: float = RUDDER_AR_GEOMETRIC

    @property
    def lift_slope(self) -> float:
        """CLα [1/rad] — Mandel, 유효 종횡비 = kΛ×기하."""
        return lift_slope_mandel(RUDDER_K_LAMBDA * self.ar_geometric)

    @staticmethod
    def for_vessel(loa: float, draft: float,
                   beam: float | None = None,
                   required_moment: float | None = None,
                   u_design: float | None = None) -> "RudderModel":
        """선체 치수 → 러더 사이징.

        면적 = max(DNV 최소, 요구 모멘트 역산) — DNV 공식은 규정
        *최소치*고, 방향 불안정 선체(통통배)는 실선도 러더·스케그를
        키운다. 역산: 설계 속도 u_design(최저 운용 속도)에서 실속각
        양력으로 required_moment를 낼 수 있는 면적.
        실측 (2026-08-03): 조사선 DNV의 2.6배 필요, 활주정은 DNV로
        3배 여유 — V² 항 때문에 느린 배일수록 큰 러더."""
        if beam is None:
            beam = loa / 2.5
        area = rudder_area_dnv(loa, beam, draft)
        x_pos = -0.48 * loa
        if required_moment is not None and u_design is not None:
            slope = lift_slope_mandel(RUDDER_K_LAMBDA * RUDDER_AR_GEOMETRIC)
            denom = (0.5 * RHO * u_design ** 2 * slope
                     * RUDDER_STALL_RAD * abs(x_pos))
            if denom > 0.0:
                area = max(area, required_moment / denom)
        return RudderModel(area=area, x_pos=x_pos)


def rudder_moment(rudder: RudderModel, u: float, delta: float) -> float:
    """타각 δ [rad] → 요 모멘트 [N·m]. 실속 밖은 포화."""
    d_eff = max(-RUDDER_STALL_RAD, min(RUDDER_STALL_RAD, delta))
    cl = rudder.lift_slope * d_eff
    lift = 0.5 * RHO * u * u * rudder.area * cl
    return -lift * rudder.x_pos
