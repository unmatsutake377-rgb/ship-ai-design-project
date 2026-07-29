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

# 수직면(상하·횡동요·종동요) 부가질량 개략 계수 — 수상선 통상 범위 (B-3a).
# 정밀값은 스트립/패널법 영역 — PoC는 자릿수 정합 목적 (docstring 참조)
ZW_DOT_MASS_FACTOR = 1.0   # 상하 부가질량/질량 (폭넓은 선체 ~1)
KP_DOT_IXX_FACTOR = 0.2    # 횡동요 부가관성/Ixx
MQ_DOT_IYY_FACTOR = 1.0    # 종동요 부가관성/Iyy (상하 분포 유사)
VERTICAL_DAMPING_ZETA = 0.7  # 목표 감쇠비 — 강성에서 감쇠 역산용


def vertical_plane_estimates(mass: float, ixx: float, iyy: float,
                             awp: float, ixx_wp: float, gm: float,
                             disp_vol: float, loa: float,
                             rho: float = 1025.0,
                             zeta: float = VERTICAL_DAMPING_ZETA) -> dict:
    """수직면 3축(상하 z, 횡동요 φ, 종동요 θ) 부가질량·감쇠 개략 (B-3a).

    방법: 복원 강성은 정역학 실계산값에서 —
      C33 = ρ·g·Awp (상하), C44 = ρ·g·∇·GM (횡동요),
      C55 = ρ·g·I_L,  I_L ≈ Awp·L²/12 (상자형 수선면 근사 — 명시)
    감쇠는 목표 감쇠비 ζ로 역산: b = 2ζ·√((관성+부가)·C).
    실제 조파감쇠(radiation damping)의 정밀 계산은 스트립/패널법 영역 —
    여기서는 '정착이 물리적 시간 스케일로 일어나는' 자릿수 정합이 목적.
    반환값 전부 크기(양수) — 부호 조립은 사용처 관례.
    """
    g = 9.81
    z_added = ZW_DOT_MASS_FACTOR * mass
    k_added = KP_DOT_IXX_FACTOR * ixx
    m_added = MQ_DOT_IYY_FACTOR * iyy

    c33 = rho * g * awp
    c44 = rho * g * disp_vol * max(gm, 1e-6)
    i_l = awp * loa ** 2 / 12.0
    c55 = rho * g * i_l

    return {
        "z_added_mass": z_added,
        "k_added_inertia": k_added,
        "m_added_inertia": m_added,
        "z_damping": 2.0 * zeta * math.sqrt((mass + z_added) * c33),
        "k_damping": 2.0 * zeta * math.sqrt((ixx + k_added) * c44),
        "m_damping": 2.0 * zeta * math.sqrt((iyy + m_added) * c55),
    }


# 회귀 괄호항이 음수로 떨어지면 물리 위반(음의 관성·감쇠) — 선두항의
# 이 비율을 하한으로 클램프하고 out-of-range를 표시한다.
# 배경: Clarke는 B/L 0.1~0.2 상선 통계. 실선 USV(B/L ~0.5)는 범위 밖 —
# 실제로 B/L=0.5에서 Nṙ 괄호항이 음수가 되어 시뮬레이션이 발산했음 (M4b).
BRACKET_FLOOR_FRACTION = 0.10

# 각 계수의 (괄호항 계산식, 선두항) — 선두항×비율이 하한
_LEADING_TERMS = {
    "Yv_dot_p": 1.0,
    "Nr_dot_p": 1.0 / 12.0,
    "Yv_p": 1.0,
    "Nv_p": 0.5,
    "Nr_p": 0.25,
}


def clarke_nondim(loa: float, beam: float, draft: float,
                  cb: float) -> tuple[dict[str, float], list[str]]:
    """Clarke(1983) 회귀 — 무차원 계수 크기 + 클램프된 항 목록.

    T는 실제(평형) 흘수. 반환: (계수 dict, 하한 클램프가 발동한 키 목록).
    클램프 발동 = 이 선형이 회귀 유효범위 밖이라는 신호.
    """
    t_l = draft / loa
    b_l = beam / loa
    b_t = beam / draft
    k = math.pi * t_l ** 2
    brackets = {
        "Yv_dot_p": 1.0 + 0.16 * cb * b_t - 5.1 * b_l ** 2,
        "Yr_dot_p": abs(0.67 * b_l - 0.0033 * b_t ** 2),
        "Nv_dot_p": abs(1.1 * b_l - 0.041 * b_t),
        "Nr_dot_p": 1.0 / 12.0 + 0.017 * cb * b_t - 0.33 * b_l,
        "Yv_p": 1.0 + 0.40 * cb * b_t,
        "Yr_p": abs(-0.5 + 2.2 * b_l - 0.080 * b_t),
        "Nv_p": 0.5 + 2.4 * t_l,
        "Nr_p": 0.25 + 0.039 * b_t - 0.56 * b_l,
    }
    clamped: list[str] = []
    result: dict[str, float] = {}
    for key, bracket in brackets.items():
        floor = BRACKET_FLOOR_FRACTION * _LEADING_TERMS.get(key, 0.0)
        if bracket < floor:
            clamped.append(key)
            bracket = floor if floor > 0 else abs(bracket)
        result[key] = k * bracket
    return result, clamped


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
    clamped_terms: tuple = ()  # 하한 클램프 발동 항 — 회귀 범위 밖 신호


def estimate_coefficients(dims: MainDimensions, draft: float, mass: float,
                          lcg: float, speed: float, mesh: trimesh.Trimesh,
                          n_exp: float, m_exp: float,
                          rho: float = RHO_SEAWATER) -> CoefficientSet:
    """Fossen 3자유도 계수 세트 (M4b 시뮬레이션·Phase B 내보내기 입력)."""
    nd, clamped = clarke_nondim(dims.loa, dims.beam, draft, dims.cb)
    L, U = dims.loa, speed
    half_rho = 0.5 * rho

    # 전진 감쇠: 자체 저항곡선 중앙차분
    r_hi = total_resistance(mesh, dims, n_exp, m_exp, draft,
                            (1.0 + SURGE_FD_STEP) * U, rho).total
    r_lo = total_resistance(mesh, dims, n_exp, m_exp, draft,
                            (1.0 - SURGE_FD_STEP) * U, rho).total
    xu = (r_hi - r_lo) / (2.0 * SURGE_FD_STEP * U)

    # 직진 안정 판별 — 부호 포함 정식 (2026-07-27 정정: 크기값 계산은 오류).
    # SNAME 부호(감쇠 음수)로 C = Y'v·N'r − N'v·(Y'r − m') 전개하면
    # 크기 기준: C = Yv_p·Nr_p − Nv_p·(Yr_p + m').
    # 통통한 맨몸 선체는 대개 C<0 (방향 불안정) — 실선이 스케그를 다는 이유.
    m_prime = mass / (half_rho * L ** 3)
    stability_index = (nd["Yv_p"] * nd["Nr_p"]
                       - nd["Nv_p"] * (nd["Yr_p"] + m_prime))

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
        clamped_terms=tuple(clamped),
    )
