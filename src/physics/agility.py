"""조종성(민첩성) 지표 — 오너 발견의 정량화 (2026-08-02).

배경: ELO 3R에서 오너의 판정 이유 "목표 방향으로 빠르게 틀어서 도착"
— 기존 3목적(저항·중량·안정)에 없는 축. 오너 결정: "충분하면 됨"
→ 최대화 목적이 아니라 **문턱 검사 + 표기** (MaxBox와 같은 패턴).
+ "사용자가 체감할 수 있게 영상으로" — duel_media의 선회 시연과 짝.

지표 유도 — python_sim의 운동방정식에서 정상 선회를 직접 풀었다
(1자유도 노모토는 횡활주 결합을 무시해 선회지름을 ~2배 과대예측함을
시뮬 교차 대조로 실측 — 통통 선체의 방향 불안정성은 선회엔 유리):
  정상 횡활주: v = −m_x·u·r / yv
  모멘트 평형: r_ss = M_max / (nr − nv·m_x·u/yv)
  최대 차동 모멘트: M_max = thrust_max·sep/2 (전진 전용 배분 규약)
  선회지름 D = 2·u / r_ss

감쇠 인자(yv, nv, nr)는 VesselModel 규약대로 **크기(양수)**로 받는다.

문턱: IMO 조종성 기준 선회지름 ≤ 5L — 대형선 기준의 외삽 + Clarke
계수 외삽의 이중 외삽 → 판정은 경고 표기만 (하드 필터 승격 보류).
"""
from __future__ import annotations

from dataclasses import dataclass

IMO_TURNING_DIAMETER_OVER_L = 5.0  # IMO 기준 (대형선 — 외삽 명시)


@dataclass(frozen=True)
class AgilityReport:
    nomoto_t: float          # 초기 반응 시간상수 [s] = Iz′/nr
    turn_rate_max: float     # 최대 정상 선회율 [rad/s] (결합 포함)
    turning_diameter: float  # 선회지름 [m]
    diameter_over_l: float
    within_imo: bool         # ≤ 5L (외삽 기준 — 경고용)
    coupled_unstable: bool   # 분모≤0: 정상 선회 한계 없음 (방향 불안정)


def agility_metrics(izz_total: float, m_x: float, yv: float, nv: float,
                    nr: float, thrust_max: float, thruster_sep: float,
                    speed: float, loa: float) -> AgilityReport:
    """동역학 계수 → 조종성 지표. 시뮬·제어기 무관한 선체 성질.

    izz_total: Izz + Nr_dot [kg·m²] / m_x: m + Xu̇ [kg]
    yv, nv, nr: 감쇠 크기 (양수 — VesselModel 규약)
    """
    if min(yv, nr) <= 0:
        raise ValueError(f"감쇠 크기는 양수여야 함: yv={yv}, nr={nr}")
    m_max = thrust_max * thruster_sep / 2.0   # 차동 모멘트 (시뮬 규약)
    t_nomoto = izz_total / nr
    denom = nr - nv * m_x * speed / yv        # 횡활주 결합 보정
    if denom <= 0:
        # 방향 불안정 속도역: 정상 선회 한계가 없음 — "무한히 민첩"이
        # 아니라 직진 유지가 어렵다는 뜻 (7/27 한계 사이클의 원인)
        return AgilityReport(nomoto_t=t_nomoto, turn_rate_max=float("inf"),
                             turning_diameter=0.0, diameter_over_l=0.0,
                             within_imo=True, coupled_unstable=True)
    r_ss = m_max / denom
    # 선회 원 지름은 대지속도 기준 — 옆미끄럼 v_ss까지 합성
    # (통통 선체는 v가 u의 상당 비율 — u만 쓰면 지름 과소)
    v_ss = m_x * speed * r_ss / yv
    ground_speed = (speed ** 2 + v_ss ** 2) ** 0.5
    diameter = 2.0 * ground_speed / r_ss
    return AgilityReport(
        nomoto_t=t_nomoto, turn_rate_max=r_ss,
        turning_diameter=diameter, diameter_over_l=diameter / loa,
        within_imo=diameter / loa <= IMO_TURNING_DIAMETER_OVER_L,
        coupled_unstable=False,
    )
