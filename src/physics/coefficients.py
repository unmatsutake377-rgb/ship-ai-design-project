"""Fossen 3자유도 계수 추정 (spec §2.3, M4a).

출처와 한계 (정직 원칙):
- 선형 미계수·부가질량류: Clarke et al. (1983) 회귀 — **대형 상선 시험 통계**.
  USV 스케일은 외삽이므로 extrapolation_warning=True를 항상 리포트.
- 전진 부가질량 Xu̇: 세장체에서 작음 — 질량의 고정 비율 개략 (명명 상수).
- 전진 감쇠 Xu: 자체 저항곡선의 수치미분 (외부 회귀 아님 — 우리 물리).

부호 규약: 모든 계수를 **크기(양수)**로 저장한다. Fossen 방정식 조립 시
(M4b) 관례에 맞는 부호(-)를 적용할 것.

무차원계 (SNAME prime): 길이 L, 속도 U 기준.
차원화: 질량류 ×½ρL³ (교차항 ×½ρL⁴, Nṙ ×½ρL⁵),
        감쇠류 ×½ρUL² (Yr·Nv ×½ρUL³, Nr ×½ρUL⁴).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import trimesh

from src.core.types import MainDimensions
from src.physics.resistance import RHO_SEAWATER, total_resistance

XU_DOT_MASS_FRACTION = 0.05  # 전진 부가질량/질량 (세장체 개략)
SURGE_FD_STEP = 0.05         # 저항 미분 중앙차분 스텝 (±5% U)


def clarke_nondim(loa: float, beam: float, draft: float,
                  cb: float) -> dict[str, float]:
    """Clarke(1983) 회귀 — 무차원 계수 크기. T는 실제(평형) 흘수."""
    t_l = draft / loa
    b_l = beam / loa
    b_t = beam / draft
    k = math.pi * t_l ** 2
    return {
        "Yv_dot_p": k * (1.0 + 0.16 * cb * b_t - 5.1 * b_l ** 2),
        "Yr_dot_p": k * abs(0.67 * b_l - 0.0033 * b_t ** 2),
        "Nv_dot_p": k * abs(1.1 * b_l - 0.041 * b_t),
        "Nr_dot_p": k * (1.0 / 12.0 + 0.017 * cb * b_t - 0.33 * b_l),
        "Yv_p": k * (1.0 + 0.40 * cb * b_t),
        "Yr_p": k * abs(-0.5 + 2.2 * b_l - 0.080 * b_t),
        "Nv_p": k * (0.5 + 2.4 * t_l),
        "Nr_p": k * (0.25 + 0.039 * b_t - 0.56 * b_l),
    }


@dataclass(frozen=True)
class CoefficientSet:
    nondim: dict
    # 차원값 [SI] — 전부 크기(양수)
    xu_dot: float   # 전진 부가질량 [kg]
    yv_dot: float   # 횡 부가질량 [kg]
    yr_dot: float   # 교차 부가항 [kg·m]
    nv_dot: float   # 교차 부가항 [kg·m]
    nr_dot: float   # 선회 부가관성 [kg·m²]
    yv: float       # 횡 감쇠 [N/(m/s)]
    yr: float       # [N/(rad/s)]
    nv: float       # [N·m/(m/s)]
    nr: float       # [N·m/(rad/s)]
    xu: float       # 전진 감쇠 = dR/du @ U [N/(m/s)]
    straight_line_stable: bool
    extrapolation_warning: bool


def estimate_coefficients(dims: MainDimensions, draft: float, mass: float,
                          lcg: float, speed: float, mesh: trimesh.Trimesh,
                          n_exp: float, m_exp: float,
                          rho: float = RHO_SEAWATER) -> CoefficientSet:
    """Fossen 3자유도 계수 세트 (M4b 시뮬레이션·Phase B 내보내기 입력)."""
    nd = clarke_nondim(dims.loa, dims.beam, draft, dims.cb)
    L, U = dims.loa, speed
    half_rho = 0.5 * rho

    # 전진 감쇠: 자체 저항곡선 중앙차분
    r_hi = total_resistance(mesh, dims, n_exp, m_exp, draft,
                            (1.0 + SURGE_FD_STEP) * U, rho).total
    r_lo = total_resistance(mesh, dims, n_exp, m_exp, draft,
                            (1.0 - SURGE_FD_STEP) * U, rho).total
    xu = (r_hi - r_lo) / (2.0 * SURGE_FD_STEP * U)

    # 직진 안정 판별 (Clarke): C = Nr'·Yv' − Nv'·(Yr' − m') > 0
    m_prime = mass / (half_rho * L ** 3)
    stability_index = (nd["Nr_p"] * nd["Yv_p"]
                       - nd["Nv_p"] * (nd["Yr_p"] - m_prime))

    return CoefficientSet(
        nondim=nd,
        xu_dot=XU_DOT_MASS_FRACTION * mass,
        yv_dot=nd["Yv_dot_p"] * half_rho * L ** 3,
        yr_dot=nd["Yr_dot_p"] * half_rho * L ** 4,
        nv_dot=nd["Nv_dot_p"] * half_rho * L ** 4,
        nr_dot=nd["Nr_dot_p"] * half_rho * L ** 5,
        yv=nd["Yv_p"] * half_rho * U * L ** 2,
        yr=nd["Yr_p"] * half_rho * U * L ** 3,
        nv=nd["Nv_p"] * half_rho * U * L ** 3,
        nr=nd["Nr_p"] * half_rho * U * L ** 4,
        xu=xu,
        straight_line_stable=bool(stability_index > 0),
        extrapolation_warning=True,
    )
