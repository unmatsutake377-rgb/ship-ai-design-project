"""대형 강선 법칙 — Watson-Gilfillan 경하중량 + ICLL 건현 + IMO GM.

전 크기 개방 2단계 (스펙 2026-08-06-all-size §3). 출처:
- Watson, "Practical Ship Design" (1998) — 강선 구조·의장 회귀 (B급)
- ICLL 1966 Type B 표준 건현표 — 대표점 선형 보간 (B급)
- IMO IS Code 2008 — 초기 GM 최소 0.15 m (A급 규정값)

유효 대역: 강선 상선 L 60~350 m 근방 (Watson 회귀 모집단).
소형(수 m급) 회귀는 기존 weights.py가 담당 — 법칙별 유효 대역
명시가 관례 (크기 스위치가 아니라 데이터 유효성의 문제).
"""
from __future__ import annotations

from dataclasses import dataclass

# 기관 중량 개략 (C급 — 3단계 엔진 실물 카탈로그로 대체 예정):
# 중속 디젤 플랜트 통상 ~20 kg/kW (보기류 포함)
MACHINERY_KG_PER_KW = 20.0

# Watson 구조 계수 K (선종별 통상 대역, B급). cargo = 일반 화물선.
WATSON_K = {"cargo": 0.032, "container": 0.036, "tanker": 0.032}
OUTFIT_CO = {"cargo": 0.40, "container": 0.36, "tanker": 0.28}

# ICLL 1966 Type B 표준 건현 대표점 [m → mm] (B급 — 표 선형 보간)
_ICLL_TYPE_B = [(24, 200), (50, 443), (85, 1075), (100, 1271),
                (150, 2315), (200, 3264), (250, 3883), (300, 4630),
                (365, 5303)]

IMO_GM_MIN_M = 0.15   # IS Code 2008 Part A 2.2.4 — 초기 GM 최소


@dataclass(frozen=True)
class LightshipBreakdown:
    structure_t: float
    outfit_t: float
    machinery_t: float

    @property
    def total_t(self) -> float:
        return self.structure_t + self.outfit_t + self.machinery_t


def equipment_numeral(loa: float, beam: float, depth: float,
                      draft: float) -> float:
    """Watson E 수 (상부구조 생략 개략 — 선각 본체 항만)."""
    return loa * (beam + draft) + 0.85 * loa * (depth - draft)


def watson_lightship(loa: float, beam: float, depth: float, draft: float,
                     cb: float, mcr_kw: float,
                     ship_type: str = "cargo") -> LightshipBreakdown:
    """경하중량 3분해 [t] — 구조(Watson-Gilfillan)·의장·기관(개략).

    Cb' 보정: Watson 규약 — 0.8D 흘수 기준 방형계수로 환산."""
    e = equipment_numeral(loa, beam, depth, draft)
    cb_prime = cb + (1.0 - cb) * (0.8 * depth - draft) / (3.0 * draft)
    k = WATSON_K.get(ship_type, WATSON_K["cargo"])
    structure = k * e ** 1.36 * (1.0 + 0.5 * (cb_prime - 0.70))
    outfit = OUTFIT_CO.get(ship_type, OUTFIT_CO["cargo"]) * loa * beam
    machinery = MACHINERY_KG_PER_KW * mcr_kw / 1000.0
    return LightshipBreakdown(structure_t=structure, outfit_t=outfit,
                              machinery_t=machinery)


def icll_freeboard_m(loa: float) -> float:
    """ICLL Type B 최소 건현 [m] — 대표점 선형 보간, 범위 밖은 끝값."""
    pts = _ICLL_TYPE_B
    if loa <= pts[0][0]:
        return pts[0][1] / 1000.0
    for (l0, f0), (l1, f1) in zip(pts, pts[1:]):
        if loa <= l1:
            return (f0 + (f1 - f0) * (loa - l0) / (l1 - l0)) / 1000.0
    return pts[-1][1] / 1000.0


def large_gm_band(beam: float) -> tuple[float, float]:
    """대형선 GM/B 밴드 — 하한 = IMO 절대 0.15 m를 GM/B로 환산.

    소형 밴드(0.04~0.40)의 하한을 대형에 그대로 쓰면 B 30 m에서
    GM 1.2 m 요구 — IMO(0.15 m)보다 과도하게 엄격. 규정 하한 채택,
    상한 0.40은 유지 (과대 GM = 급격 횡요 — 크기 무관 원리)."""
    return (IMO_GM_MIN_M / beam, 0.40)
