"""Savitsky 활주 평형·저항 (Phase C-2, 1964 반경험식).

물리 전환점: 배수량·반배수량은 부력이 배를 들었지만, 활주는 속도가
만드는 **동적 양력**이 든다 — 주행 트림 τ와 침수 길이비 λ가 속도의
함수가 되고, 저항 = 무게×tanτ + 바닥 마찰.

공식 (Savitsky 1964 — 프리즘 선체 β=const):
  C_L0 = τ^1.1 · (0.0120·λ^0.5 + 0.0055·λ^2.5/Cv²)      [τ in deg]
  C_Lβ = C_L0 − 0.0065·β·C_L0^0.60                       [β in deg]
  Cp   = lcp/(λ·b) = 0.75 − 1/(5.21·Cv²/λ² + 2.39)
  V₁   = V·√(1 − 0.0120·τ^1.1/(λ^0.5·cosτ))              (바닥 평균유속 근사)
  R    = Δ·tanτ + Df/cosτ,  Df = ½ρV₁²·(λ·b²/cosβ)·Cf(ITTC)

검증 상태 (정직): 경험식 전사 손계산 + 평형 자기일관 + 물리 경향으로
고정. 원논문 워크드 예제 정량 대조는 원문 확보 시 추가 예정.
유효범위: 0.60≤Cv, 2°≤τ≤15°, λ≤4 부근 (밖이면 결과에 경고 플래그).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

RHO_SEAWATER = 1025.0
NU_SEAWATER = 1.19e-6
G = 9.81

TAU_RANGE = (0.5, 15.0)   # 주행 트림 탐색 범위 [deg]
LAMBDA_RANGE = (0.05, 8.0)  # 침수 길이비 탐색 범위


class PlaningEquilibriumError(RuntimeError):
    """활주 평형 해 없음 — 속도·무게·LCG 조합이 활주 성립 밖."""


def cl_zero(tau_deg: float, lam: float, cv: float) -> float:
    """데드라이즈 0° 양력계수 (Savitsky 식 전사)."""
    return tau_deg ** 1.1 * (0.0120 * lam ** 0.5
                             + 0.0055 * lam ** 2.5 / cv ** 2)


def cl_beta(cl0: float, beta_deg: float) -> float:
    """데드라이즈 보정 양력계수."""
    return cl0 - 0.0065 * beta_deg * cl0 ** 0.60


def solve_cl0(cl_beta_target: float, beta_deg: float) -> float:
    """C_Lβ → C_L0 역산 (단조 증가 — 이분법)."""
    lo, hi = cl_beta_target, cl_beta_target * 3.0 + 0.1
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cl_beta(mid, beta_deg) < cl_beta_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def center_of_pressure_ratio(cv: float, lam: float) -> float:
    """압력중심 위치 / (λ·b) — 트랜섬에서 앞으로."""
    return 0.75 - 1.0 / (5.21 * cv ** 2 / lam ** 2 + 2.39)


def _lambda_from_lift(tau_deg: float, cl0_target: float, cv: float) -> float:
    """주어진 τ에서 양력식을 만족하는 λ (단조 증가 — 이분법)."""
    lo, hi = LAMBDA_RANGE
    if cl_zero(tau_deg, hi, cv) < cl0_target:
        raise PlaningEquilibriumError("λ 상한에서도 양력 부족")
    if cl_zero(tau_deg, lo, cv) > cl0_target:
        raise PlaningEquilibriumError("λ 하한에서도 양력 과다")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cl_zero(tau_deg, mid, cv) < cl0_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class PlaningState:
    trim_deg: float
    lam: float             # 평균 침수 길이 / 폭
    wetted_length: float   # λ·b [m]
    cv: float
    resistance_n: float
    friction_n: float
    induced_n: float       # Δ·tanτ (양력 유도 성분)
    out_of_validity: bool  # Savitsky 유효범위 밖 플래그


def solve_equilibrium(weight_n: float, speed: float, beam: float,
                      deadrise_deg: float, lcg_from_transom: float,
                      rho: float = RHO_SEAWATER,
                      nu: float = NU_SEAWATER) -> PlaningState:
    """활주 평형 (τ, λ) 이중 이분법 + 저항.

    바깥 루프: 모멘트 잔차 f(τ) = lcp(τ) − LCG.
    τ↑ → 같은 양력에 필요한 λ↓ → lcp↓ : f는 τ에 단조 감소.
    """
    from src.physics.resistance import ittc_cf

    cv = speed / math.sqrt(G * beam)
    clb_target = weight_n / (0.5 * rho * speed ** 2 * beam ** 2)
    cl0_target = solve_cl0(clb_target, deadrise_deg)

    def lcp_minus_lcg(tau):
        lam = _lambda_from_lift(tau, cl0_target, cv)
        return center_of_pressure_ratio(cv, lam) * lam * beam \
            - lcg_from_transom, lam

    # τ 격자에서 실행 가능(양력식 해 존재) 구간을 먼저 찾는다 — 경계
    # 양끝만 평가하면 중간에 해가 있어도 놓침 (07-30 실측 수정)
    taus = [TAU_RANGE[0] + k * (TAU_RANGE[1] - TAU_RANGE[0]) / 60
            for k in range(61)]
    feasible: list[tuple[float, float]] = []
    for t in taus:
        try:
            f, _ = lcp_minus_lcg(t)
            feasible.append((t, f))
        except PlaningEquilibriumError:
            continue
    if not feasible:
        raise PlaningEquilibriumError(
            "활주 평형 불성립 — τ 전 범위에서 양력식 해 없음 "
            "(속도·무게·폭 조합이 활주 성립 밖)")
    bracket = None
    for (t1, f1), (t2, f2) in zip(feasible, feasible[1:]):
        if f1 * f2 <= 0:
            bracket = (t1, f1, t2)
            break
    if bracket is None:
        raise PlaningEquilibriumError(
            "모멘트 균형 해 없음 — LCG가 활주 압력중심 도달범위 밖 "
            f"(잔차 부호 불변, LCG={lcg_from_transom:.2f} m)")
    lo, f_lo, hi = bracket
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid, lam = lcp_minus_lcg(mid)
        if f_mid * f_lo > 0:
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    tau = 0.5 * (lo + hi)
    _, lam = lcp_minus_lcg(tau)

    # 저항
    tau_rad = math.radians(tau)
    v1 = speed * math.sqrt(max(
        0.1, 1.0 - 0.0120 * tau ** 1.1 / (lam ** 0.5 * math.cos(tau_rad))))
    wetted = lam * beam ** 2 / math.cos(math.radians(deadrise_deg))
    re = v1 * lam * beam / nu
    df = 0.5 * rho * v1 ** 2 * wetted * ittc_cf(re)
    induced = weight_n * math.tan(tau_rad)
    total = induced + df / math.cos(tau_rad)

    out = not (0.60 <= cv and 2.0 <= tau <= 15.0 and lam <= 4.0)
    return PlaningState(trim_deg=tau, lam=lam, wetted_length=lam * beam,
                        cv=cv, resistance_n=total, friction_n=df,
                        induced_n=induced, out_of_validity=out)
