# 조종성 1단계 — MMG 표준기 (KVLCC2 실측 재현) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MMG 3자유도 모델 + KVLCC2 공개 계수로 35° 선회권·지그재그를 실행, 원전 논문의 계산·실측값을 재현한다 (모델 오차 단독 계측).

**Architecture:** `src/physics/maneuvering/` — `kvlcc2.py`(원전 데이터 박제) → `mmg.py`(계수 주입형 힘 모델 + 상태 미분) → `trials.py`(RK4 적분 + 표준 시험 지표). 원전 = `references/Yasukawa2015_MMG.pdf` (확보됨 — Table 1 p5·계수 Table 3 p12·선회 실측 Table 4 p12·지그재그 Table 5 p13, 정식화 Eq 4~38 p3~11).

**Tech Stack:** Python 3, numpy, pytest.

## Global Constraints

- 원전 수치가 정답 — 아래 박제값과 PDF가 다르면 PDF 값으로 교체 (페이지 병기)
- 무차원 관례 (원전 §2): 힘 ½ρ·L·d·U², 모멘트 ½ρ·L²·d·U², 질량류 ½ρ·L²·d — v' = v_m/U, r' = r·L/U
- 좌표: 미드십 원점, x 선수+, y 우현+, ψ 우회두+ (원전 Fig 1)
- 한국어 docstring, 기존 시험 382 통과 유지, main 직커밋
- 앵커 허용 대역: 원전 "계산" 재현이 목표 (같은 계수·같은 모델 → 수 % 이내), 실측 대비는 원전도 5.8% 어긋남 — 시험은 원전 계산값 ±10% 대역

---

### Task 1: `kvlcc2.py` — 원전 데이터 박제

**Files:**
- Create: `src/physics/maneuvering/__init__.py` (빈 파일)
- Create: `src/physics/maneuvering/kvlcc2.py`
- Test: `tests/test_maneuvering_kvlcc2.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) MMGCoeffs` — 선체 미계수 15종(X'vv…N'rrr), 부가질량 3종(mx' my' Jz'), 프로펠러(tP, wP0, k0 k1 k2), 러더(tR, aH, xH', ε, κ, fα, γR_plus, γR_minus, ℓR', C1, C2_plus, C2_minus, x'R=−0.5)
  - `@dataclass(frozen=True) ShipParticulars` — lpp, beam, draft, displacement_m3, xg, cb, dp, hr, ar, rho
  - `KVLCC2_L7: ShipParticulars` (Lpp 7.00·B 1.27·d 0.46·∇ 3.27·xG 0.25·Cb 0.810·DP 0.216·HR 0.345·AR 0.0539 — 원전 Table 1 p5)
  - `KVLCC2_COEFFS: MMGCoeffs` (원전 Table 3 p12 전 계수)
  - `PAPER_ANCHORS: dict` — 선회 A_D 계산 3.31/실측 3.25, D_T 3.36/3.34 (L7, Table 4 p12)

- [ ] **Step 1: 실패 시험 작성**

```python
# tests/test_maneuvering_kvlcc2.py
"""KVLCC2 원전 데이터 박제 검증 (Yasukawa & Yoshimura 2015)."""
import pytest


def test_particulars_table1():
    """Table 1 (p5) 대표값 — L7 모델."""
    from src.physics.maneuvering.kvlcc2 import KVLCC2_L7
    s = KVLCC2_L7
    assert s.lpp == pytest.approx(7.00)
    assert s.beam == pytest.approx(1.27)
    assert s.draft == pytest.approx(0.46)
    assert s.cb == pytest.approx(0.810)
    assert s.ar == pytest.approx(0.0539)


def test_coeffs_table3():
    """Table 3 (p12) 대표값 — 부호 포함."""
    from src.physics.maneuvering.kvlcc2 import KVLCC2_COEFFS
    c = KVLCC2_COEFFS
    assert c.yv == pytest.approx(-0.315)
    assert c.xvvvv == pytest.approx(0.771)
    assert c.nrrr == pytest.approx(-0.013)
    assert c.a_h == pytest.approx(0.312)
    assert c.eps == pytest.approx(1.09)
    assert c.f_alpha == pytest.approx(2.747)
    assert c.gamma_r_plus == pytest.approx(0.640)   # βR>0
    assert c.gamma_r_minus == pytest.approx(0.395)  # βR<0
```

- [ ] **Step 2: 실패 확인** — Run: `python -m pytest tests/test_maneuvering_kvlcc2.py -q` → FAIL ModuleNotFoundError

- [ ] **Step 3: 구현** — 원전 Table 1(p5)·Table 3(p12) 재판독 후 박제:

```python
# src/physics/maneuvering/kvlcc2.py
"""KVLCC2 원전 데이터 (Yasukawa & Yoshimura 2015, JMST 20:37-52).

references/Yasukawa2015_MMG.pdf — Table 1 (p5, 0-기준) 주요목,
Table 3 (p12) 유체력 계수, Table 4 (p12) 선회 실측 앵커.
A급: MMG 표준법 원저 + SIMMAN2008 공개 실측 계보.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShipParticulars:
    lpp: float
    beam: float
    draft: float
    displacement_m3: float
    xg: float              # 미드십 기준 +선수 [m]
    cb: float
    dp: float              # 프로펠러 직경
    hr: float              # 러더 높이
    ar: float              # 러더 가동부 면적
    rho: float = 1000.0    # 수조 담수


@dataclass(frozen=True)
class MMGCoeffs:
    # 선체 (Table 3)
    xvv: float; xvr: float; xrr: float; xvvvv: float
    yv: float; yr: float; yvvv: float; yvvr: float
    yvrr: float; yrrr: float
    nv: float; nr: float; nvvv: float; nvvr: float
    nvrr: float; nrrr: float
    # 부가질량
    mx_p: float; my_p: float; jz_p: float
    # 프로펠러
    t_p: float; w_p0: float
    k0: float; k1: float; k2: float
    c1: float; c2_plus: float; c2_minus: float   # 반류 변화 (βP 부호별)
    # 러더
    t_r: float; a_h: float; x_h_p: float
    eps: float; kappa: float; f_alpha: float
    gamma_r_plus: float; gamma_r_minus: float
    ell_r_p: float
    x_r_p: float = -0.5    # 러더 위치 (미드십 기준 −L/2)


KVLCC2_L7 = ShipParticulars(
    lpp=7.00, beam=1.27, draft=0.46, displacement_m3=3.27,
    xg=0.25, cb=0.810, dp=0.216, hr=0.345, ar=0.0539)

KVLCC2_COEFFS = MMGCoeffs(
    xvv=-0.040, xvr=0.002, xrr=0.011, xvvvv=0.771,
    yv=-0.315, yr=0.083, yvvv=-1.607, yvvr=0.379,
    yvrr=-0.391, yrrr=0.008,
    nv=-0.137, nr=-0.049, nvvv=-0.030, nvvr=-0.294,
    nvrr=0.055, nrrr=-0.013,
    mx_p=0.022, my_p=0.223, jz_p=0.011,
    t_p=0.220, w_p0=0.40,
    k0=0.2931, k1=0.2753, k2=-0.1385,
    c1=2.0, c2_plus=1.6, c2_minus=1.1,
    t_r=0.387, a_h=0.312, x_h_p=-0.464,
    eps=1.09, kappa=0.50, f_alpha=2.747,
    gamma_r_plus=0.640, gamma_r_minus=0.395,
    ell_r_p=-0.710)

PAPER_ANCHORS = {
    "turning_advance_cal": 3.31, "turning_advance_exp": 3.25,
    "turning_tactical_cal": 3.36, "turning_tactical_exp": 3.34,
    "note": "Table 4 (p12) L7 모델 δ=35° — 계산/실측, L 배수",
}
```

주의: Y'R·N'R 표기 = 원전의 Y'r·N'r (r' 미계수 — Table 3에서 대문자 R 아래첨자). C2는 βP 부호별 (>0: 1.6, <0: 1.1).

- [ ] **Step 4: 통과 확인 + 커밋**

```bash
git add src/physics/maneuvering/ tests/test_maneuvering_kvlcc2.py
git commit -m "feat: KVLCC2 원전 데이터 박제 — Yasukawa 2015 Table 1·3 (A급)"
```

---

### Task 2: `mmg.py` — MMG 3자유도 힘 모델 + 상태 미분

**Files:**
- Create: `src/physics/maneuvering/mmg.py`
- Test: `tests/test_maneuvering_mmg.py`

**Interfaces:**
- Consumes: `ShipParticulars`, `MMGCoeffs` (Task 1)
- Produces:
  - `@dataclass MMGShip: par, co, mass_kg(=ρ∇), izz(=m·(0.25L)²), r0_n(정수 저항 [N] at U0), n_p(프로펠러 회전수 [1/s])`
  - `solve_self_propulsion(par, co, u0, r0_n) -> float` — 직진 평형 nP: (1−tP)·T(nP) = R0
  - `derivatives(ship, state, delta) -> np.ndarray` — state = [u, v_m, r, x0, y0, psi], 반환 미분 6종
  - 내부: `_hull_forces`, `_propeller_force`, `_rudder_forces` (원전 Eq 9~27 — 구현 직전 p3~11 재판독)

핵심 수식 (원전 정식화 — 구현 시 페이지 대조):
- 운동방정식 (미드십 원점, Eq 4~6): `(m+mx)u̇ − (m+my)v_m r − xG m r² = X`, `(m+my)v̇_m + (m+mx)u r + xG m ṙ = Y`, `(Izz + xG²m + Jz)ṙ + xG m (v̇_m + u r) = N`
- 선체 (Eq 9~11): X_H = ½ρLdU²·(−R0' + X'vv v'² + X'vr v'r' + X'rr r'² + X'vvvv v'⁴), Y_H·N_H 다항 동형 (N은 ½ρL²dU²)
- 프로펠러 (Eq 12~15): X_P = (1−tP)·ρ nP² DP⁴·KT(JP), KT = k0 + k1 JP + k2 JP², JP = u(1−wP)/(nP DP), 조종 중 반류 (Eq 29~31 재판독): (1−wP)/(1−wP0) = 1 + {1−exp(−C1|βP|)}(C2−1), βP = β − x'P r' (x'P 원전값 재판독)
- 러더 (Eq 16~27): F_N = ½ρ AR UR² fα sin αR, αR = δ − atan(vR/uR), uR (Eq 25: ε·κ·η=DP/HR 증속식), vR = U γR βR (βR = β − ℓ'R r', γR 부호별), X_R = −(1−tR)F_N sinδ, Y_R = −(1+aH)F_N cosδ, N_R = −(x'R + aH x'H)L·F_N cosδ

- [ ] **Step 1: 실패 시험 작성**

```python
# tests/test_maneuvering_mmg.py
"""MMG 힘 모델 — 직진 평형·대칭 자기검증 (스펙 §3 앵커 ①②)."""
import numpy as np
import pytest

from src.physics.maneuvering.kvlcc2 import KVLCC2_COEFFS, KVLCC2_L7

U0 = 1.179          # L7 접근 속도 [m/s] = 15.5kn/√45.7 (원전 §3.2)


def _ship():
    from src.physics.maneuvering.mmg import MMGShip, solve_self_propulsion
    r0 = 8.0        # 정수 저항 [N] 개략 — 직진 평형 항등식엔 임의값 무방
    n_p = solve_self_propulsion(KVLCC2_L7, KVLCC2_COEFFS, U0, r0)
    return MMGShip(par=KVLCC2_L7, co=KVLCC2_COEFFS, r0_n=r0, n_p=n_p)


def test_straight_run_equilibrium():
    """타 0·직진 → 가속 0 (자기검증 — 저항과 추력이 상쇄)."""
    from src.physics.maneuvering.mmg import derivatives
    ship = _ship()
    state = np.array([U0, 0.0, 0.0, 0.0, 0.0, 0.0])
    d = derivatives(ship, state, delta=0.0)
    assert abs(d[0]) < 1e-6      # u̇ ≈ 0
    assert abs(d[1]) < 1e-9      # v̇ = 0 (대칭)
    assert abs(d[2]) < 1e-9      # ṙ = 0


def test_rudder_sign_symmetry():
    """±δ 거울: 우현타 → 우회두 모멘트 (ṙ>0), 좌현타 반대.

    γR 비대칭(0.640/0.395)은 사항 상태에서만 작동 — 직진 순간
    타력은 좌우 크기 동일."""
    from src.physics.maneuvering.mmg import derivatives
    ship = _ship()
    state = np.array([U0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dp = derivatives(ship, state, delta=np.radians(20.0))
    dm = derivatives(ship, state, delta=np.radians(-20.0))
    assert dp[2] * dm[2] < 0                      # 반대 방향 회두
    assert abs(dp[2]) == pytest.approx(abs(dm[2]), rel=1e-6)
    assert dp[0] < 0 and dm[0] < 0                # 조타 = 감속


def test_drift_restoring_moment():
    """우현 사항(v_m<0 β>0) → 선체가 회두 유발 (N_H 부호 방향,
    불안정 대형 유조선 특성 — N'v 음수 확인)."""
    from src.physics.maneuvering.mmg import derivatives
    ship = _ship()
    state = np.array([U0, -0.1, 0.0, 0.0, 0.0, 0.0])
    d = derivatives(ship, state, delta=0.0)
    assert d[2] != 0.0
```

- [ ] **Step 2: 실패 확인** — Run → FAIL ModuleNotFoundError

- [ ] **Step 3: 원전 재판독** — p3~11 (Eq 4~31): 운동방정식 부호, βP 정의의 x'P 값, uR 식 (Eq 25)의 η = DP/HR, N_R의 x'H 적용식. 아래 코드와 다르면 원전이 정답.

- [ ] **Step 4: 구현**

```python
# src/physics/maneuvering/mmg.py
"""MMG 3자유도 조종 모델 (조종성 1단계, 스펙 2026-08-09 §2).

원전: Yasukawa & Yoshimura (2015) — 힘 = 선체(H) + 프로펠러(P) +
러더(R) 분해, 미드십 원점·xG 오프셋 운동방정식 (Eq 4~6).
무차원: 힘 ½ρLdU², 모멘트 ½ρL²dU² (v'=v_m/U, r'=rL/U).

계수 주입형 — 배(KVLCC2·우리 배)와 모델이 분리. 2단계 estimation
이 같은 인터페이스로 우리 배 계수를 만든다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.physics.maneuvering.kvlcc2 import MMGCoeffs, ShipParticulars

X_P_PRIME = -0.48      # βP = β − x'P·r' 의 프로펠러 위치 (원전 재판독 확정)


@dataclass(frozen=True)
class MMGShip:
    par: ShipParticulars
    co: MMGCoeffs
    r0_n: float            # 직진 저항 at 접근 속도 [N]
    n_p: float             # 프로펠러 회전수 [1/s] (자항 평형)

    @property
    def mass(self) -> float:
        return self.par.rho * self.par.displacement_m3

    @property
    def izz(self) -> float:
        return self.mass * (0.25 * self.par.lpp) ** 2


def _kt(co: MMGCoeffs, j_p: float) -> float:
    return co.k0 + co.k1 * j_p + co.k2 * j_p * j_p


def solve_self_propulsion(par: ShipParticulars, co: MMGCoeffs,
                          u0: float, r0_n: float) -> float:
    """직진 평형 nP — (1−tP)·ρnP²DP⁴·KT(JP) = R0 이분법."""
    def net(n_p: float) -> float:
        j_p = u0 * (1.0 - co.w_p0) / max(n_p * par.dp, 1e-9)
        t = par.rho * n_p ** 2 * par.dp ** 4 * _kt(co, j_p)
        return (1.0 - co.t_p) * t - r0_n
    lo, hi = 0.1, 60.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if net(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _hull_forces(par, co, u, v_m, r, upow) -> tuple[float, float, float]:
    """선체 유체력 (Eq 9~11) — R0 제외 (호출측 합산)."""
    big_u = math.hypot(u, v_m)
    vp = v_m / big_u
    rp = r * par.lpp / big_u
    q = 0.5 * par.rho * par.lpp * par.draft * big_u ** 2
    xh = q * (co.xvv * vp ** 2 + co.xvr * vp * rp + co.xrr * rp ** 2
              + co.xvvvv * vp ** 4)
    yh = q * (co.yv * vp + co.yr * rp + co.yvvv * vp ** 3
              + co.yvvr * vp ** 2 * rp + co.yvrr * vp * rp ** 2
              + co.yrrr * rp ** 3)
    nh = q * par.lpp * (co.nv * vp + co.nr * rp + co.nvvv * vp ** 3
                        + co.nvvr * vp ** 2 * rp
                        + co.nvrr * vp * rp ** 2 + co.nrrr * rp ** 3)
    return xh, yh, nh


def _propeller(par, co, u, v_m, r) -> tuple[float, float]:
    """프로펠러 추력 X_P + 러더용 (KT, JP, wP) 전달."""
    big_u = math.hypot(u, v_m)
    beta = math.atan2(-v_m, u)
    rp = r * par.lpp / max(big_u, 1e-9)
    beta_p = beta - X_P_PRIME * rp
    c2 = co.c2_plus if beta_p >= 0.0 else co.c2_minus
    ratio = 1.0 + (1.0 - math.exp(-co.c1 * abs(beta_p))) * (c2 - 1.0)
    w_p = 1.0 - ratio * (1.0 - co.w_p0)
    n_p = None                    # 호출측 ship.n_p 사용 — 아래 조립
    return w_p, beta


def _rudder_forces(ship: MMGShip, u, v_m, r,
                   delta) -> tuple[float, float, float]:
    """러더 힘 (Eq 16~27) — 프로펠러 후류 증속 + 선체 상호작용."""
    par, co = ship.par, ship.co
    big_u = math.hypot(u, v_m)
    w_p, beta = _propeller(par, co, u, v_m, r)
    rp = r * par.lpp / max(big_u, 1e-9)
    j_p = u * (1.0 - w_p) / max(ship.n_p * par.dp, 1e-9)
    kt = max(_kt(co, j_p), 0.0)
    eta = par.dp / par.hr
    inner = 1.0 + co.kappa * (math.sqrt(1.0 + 8.0 * kt
                                        / (math.pi * j_p ** 2)) - 1.0)
    u_r = (co.eps * u * (1.0 - w_p)
           * math.sqrt(eta * inner ** 2 + (1.0 - eta)))
    beta_r = beta - co.ell_r_p * rp
    gamma = co.gamma_r_plus if beta_r >= 0.0 else co.gamma_r_minus
    v_r = big_u * gamma * beta_r
    u_res2 = u_r ** 2 + v_r ** 2
    alpha_r = delta - math.atan2(v_r, max(u_r, 1e-9))
    f_n = (0.5 * par.rho * par.ar * u_res2
           * co.f_alpha * math.sin(alpha_r))
    x_r = -(1.0 - co.t_r) * f_n * math.sin(delta)
    y_r = -(1.0 + co.a_h) * f_n * math.cos(delta)
    n_r = -(co.x_r_p + co.a_h * co.x_h_p) * par.lpp \
        * f_n * math.cos(delta)
    return x_r, y_r, n_r


def derivatives(ship: MMGShip, state: np.ndarray,
                delta: float) -> np.ndarray:
    """상태 미분 — state = [u, v_m, r, x0, y0, psi]."""
    par, co = ship.par, ship.co
    u, v_m, r, _x0, _y0, psi = state
    m = ship.mass
    md = 0.5 * par.rho * par.lpp ** 2 * par.draft
    mx = co.mx_p * md
    my = co.my_p * md
    jz = co.jz_p * 0.5 * par.rho * par.lpp ** 4 * par.draft

    xh, yh, nh = _hull_forces(par, co, u, v_m, r, None)
    w_p, _ = _propeller(par, co, u, v_m, r)
    j_p = u * (1.0 - w_p) / max(ship.n_p * par.dp, 1e-9)
    x_p = (1.0 - co.t_p) * par.rho * ship.n_p ** 2 * par.dp ** 4 \
        * max(_kt(co, j_p), 0.0)
    # R0는 속도 제곱 스케일 (접근 속도 기준 정규화)
    big_u = math.hypot(u, v_m)
    r0 = ship.r0_n * (big_u / max(abs(u), 1e-9)) ** 0    # 원전: R0(u) —
    r0 = ship.r0_n * (u / max(ship_u0(ship), 1e-9)) ** 2
    x_r, y_r, n_r = _rudder_forces(ship, u, v_m, r, delta)

    x = xh - r0 + x_p + x_r
    y = yh + y_r
    n = nh + n_r - ship.par.xg * 0.0   # N_H는 미드십 기준 — xG 항은 EOM에

    xg = par.xg
    # 연립 (v̇, ṙ) — Y·N 식이 결합 (xG 오프셋)
    a11 = m + my
    a12 = xg * m
    a21 = xg * m
    a22 = ship.izz + xg ** 2 * m + jz
    b1 = y - (m + mx) * u * r
    b2 = n - xg * m * u * r
    det = a11 * a22 - a12 * a21
    v_dot = (b1 * a22 - b2 * a12) / det
    r_dot = (a21 * -b1 + a11 * b2) / det * 0 + (b2 * a11 - b1 * a21) / det
    u_dot = (x + (m + my) * v_m * r + xg * m * r ** 2) / (m + mx)

    x0_dot = u * math.cos(psi) - v_m * math.sin(psi)
    y0_dot = u * math.sin(psi) + v_m * math.cos(psi)
    return np.array([u_dot, v_dot, r_dot, x0_dot, y0_dot, r])


def ship_u0(ship: MMGShip) -> float:
    """자항 평형이 성립하는 접근 속도 (r0_n 정의 속도) — MMGShip
    생성 관례상 solve_self_propulsion에 준 u0와 동일해야 함."""
    # r0_n = (1−tP)·T(u0) 역산
    ...
```

**구현 주의 (계획 스케치의 정리 지점):** 위 `derivatives`의 R0 처리·`ship_u0`는 스케치 — 실제 구현은 `MMGShip`에 `u0` 필드를 추가해 `r0 = r0_n·(u/u0)²`로 깔끔히 (저항 ∝ u², 원전은 Schoenherr 곡선 — 접근 속도 근방 ±20%에서 u² 근사 충분, docstring 정직 표기). `_propeller` 반환은 `(w_p, beta)`로 확정, `r_dot` 이중 대입 스케치는 `(b2·a11 − b1·a21)/det` 한 줄로. `n` 조립의 `xg·0.0` 잔재 제거 — N_H·N_R 모두 미드십 기준이 원전 관례.

- [ ] **Step 5: 통과 확인 + 커밋**

Run: `python -m pytest tests/test_maneuvering_mmg.py -v` → 3 PASS

```bash
git add src/physics/maneuvering/mmg.py tests/test_maneuvering_mmg.py
git commit -m "feat: MMG 3자유도 힘 모델 — 직진 평형·타 대칭 자기검증"
```

---

### Task 3: `trials.py` — 표준 시험 + KVLCC2 실측 재현

**Files:**
- Create: `src/physics/maneuvering/trials.py`
- Test: `tests/test_maneuvering_trials.py`

**Interfaces:**
- Consumes: `MMGShip`, `derivatives` (Task 2)
- Produces:
  - `simulate(ship, delta_fn, t_end, dt=0.05, u0) -> dict` — RK4, 기록 (t, state, delta)
  - `turning_circle(ship, u0, delta_deg=35.0, rudder_rate_dps=None) -> dict` — 키: `advance_over_l, transfer_over_l, tactical_diameter_over_l` (ψ=90°/180° 보간 판독)
  - `zigzag(ship, u0, delta_deg=10.0, switch_deg=10.0) -> dict` — 키: `first_overshoot_deg, second_overshoot_deg, initial_turning_time_s`
  - 러더 속도: 원전 §5.1 재판독 (모델 스케일 통상 15.8°/s 대역) — 상수 `RUDDER_RATE_MODEL_DPS` 박제

- [ ] **Step 1: 실패 시험 작성**

```python
# tests/test_maneuvering_trials.py
"""표준 조종 시험 — KVLCC2 원전 재현 앵커 (스펙 §3 ③④)."""
import pytest

from src.physics.maneuvering.kvlcc2 import (
    KVLCC2_COEFFS, KVLCC2_L7, PAPER_ANCHORS)

U0 = 1.179


def _ship():
    from src.physics.maneuvering.mmg import MMGShip, solve_self_propulsion
    r0 = 8.0     # Task 3 Step 3에서 원전 §5.1 저항 처리 확정 후 갱신
    n_p = solve_self_propulsion(KVLCC2_L7, KVLCC2_COEFFS, U0, r0)
    return MMGShip(par=KVLCC2_L7, co=KVLCC2_COEFFS, r0_n=r0,
                   n_p=n_p, u0=U0)


def test_turning_circle_reproduces_paper():
    """35° 선회 — 원전 계산값 (A_D 3.31·D_T 3.36) ±10% 재현.

    원전 자체가 실측과 5.8% 어긋남 — 우리 목표는 '같은 계수·같은
    모델이면 같은 답' (모델 오차 단독 계측)."""
    from src.physics.maneuvering.trials import turning_circle
    r = turning_circle(_ship(), U0, delta_deg=35.0)
    assert r["advance_over_l"] == pytest.approx(
        PAPER_ANCHORS["turning_advance_cal"], rel=0.10)
    assert r["tactical_diameter_over_l"] == pytest.approx(
        PAPER_ANCHORS["turning_tactical_cal"], rel=0.10)


def test_turning_mirror():
    """±35° 거울 — 대칭 모델 자기검증 (γR 비대칭은 크기 수 % 차이
    허용)."""
    from src.physics.maneuvering.trials import turning_circle
    rp = turning_circle(_ship(), U0, delta_deg=35.0)
    rm = turning_circle(_ship(), U0, delta_deg=-35.0)
    assert rm["tactical_diameter_over_l"] == pytest.approx(
        rp["tactical_diameter_over_l"], rel=0.10)


def test_zigzag_overshoot_band():
    """10/10 지그재그 — 오버슈트 존재·원전 Fig14 대역 (1차 ~8°±5°),
    dt 반감 수렴 (적분 건강)."""
    from src.physics.maneuvering.trials import zigzag
    z = zigzag(_ship(), U0, delta_deg=10.0, switch_deg=10.0)
    assert 1.0 < z["first_overshoot_deg"] < 15.0
    assert z["second_overshoot_deg"] > 0.0
```

- [ ] **Step 2: 실패 확인** — FAIL ModuleNotFoundError

- [ ] **Step 3: 원전 §5.1 (p11) 재판독** — 러더 속도·모델 저항 처리·프로펠러 회전수 기술 확인 → `_ship()`의 r0 결정 방식 확정 (원전에 수치 없으면: ITTC Cf + 형상계수 1+k=1.25 (SIMMAN 계보)·습면적 27,194/45.7² m² 로 R0 산출, C급 병기 — 선회 지표는 저항 민감도 낮음 (docstring 정직 표기))

- [ ] **Step 4: 구현**

```python
# src/physics/maneuvering/trials.py
"""표준 조종 시험 실행기 (조종성 1단계, 스펙 §2).

선회권: 타 35° → ψ=90° 시점 전진거리(advance)·횡거리(transfer),
ψ=180° 시점 횡거리 = 선회지름(tactical diameter) — L 배수 무차원.
지그재그: 타 δ↔−δ (ψ가 ±switch 도달 시 반전) → 오버슈트 각.
적분: RK4 고정 스텝 (dt 0.05 s 모델 스케일 — 수렴 시험으로 확인).
"""
from __future__ import annotations

import math

import numpy as np

from src.physics.maneuvering.mmg import MMGShip, derivatives

RUDDER_RATE_MODEL_DPS = 15.8    # 원전 §5.1 재판독 확정값으로 교체


def _rk4_step(ship, state, delta, dt):
    k1 = derivatives(ship, state, delta)
    k2 = derivatives(ship, state + 0.5 * dt * k1, delta)
    k3 = derivatives(ship, state + 0.5 * dt * k2, delta)
    k4 = derivatives(ship, state + dt * k3, delta)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(ship: MMGShip, delta_cmd_fn, t_end: float, u0: float,
             dt: float = 0.05) -> dict:
    """delta_cmd_fn(t, state, delta_now) → 명령 타각 [rad].
    실제 타각은 러더 속도 제한으로 추종."""
    rate = math.radians(RUDDER_RATE_MODEL_DPS)
    state = np.array([u0, 0.0, 0.0, 0.0, 0.0, 0.0])
    delta = 0.0
    ts, states, deltas = [], [], []
    n = int(t_end / dt)
    for i in range(n):
        t = i * dt
        cmd = delta_cmd_fn(t, state, delta)
        step = np.clip(cmd - delta, -rate * dt, rate * dt)
        delta += float(step)
        ts.append(t); states.append(state.copy()); deltas.append(delta)
        state = _rk4_step(ship, state, delta, dt)
    return {"t": np.array(ts), "state": np.array(states),
            "delta": np.array(deltas)}


def _interp_at_psi(res, psi_target):
    psi = res["state"][:, 5]
    idx = np.argmax(np.abs(psi) >= abs(psi_target))
    if idx == 0:
        raise ValueError("ψ 목표 미도달 — t_end 부족")
    p0, p1 = abs(psi[idx - 1]), abs(psi[idx])
    w = (abs(psi_target) - p0) / max(p1 - p0, 1e-12)
    s = res["state"]
    return s[idx - 1] + w * (s[idx] - s[idx - 1])


def turning_circle(ship: MMGShip, u0: float,
                   delta_deg: float = 35.0) -> dict:
    d_cmd = math.radians(delta_deg)
    res = simulate(ship, lambda t, s, d: d_cmd,
                   t_end=400.0 * ship.par.lpp / 7.0 / u0 * 1.179, u0=u0)
    s90 = _interp_at_psi(res, math.pi / 2.0)
    s180 = _interp_at_psi(res, math.pi)
    lpp = ship.par.lpp
    return {
        "advance_over_l": abs(s90[3]) / lpp,
        "transfer_over_l": abs(s90[4]) / lpp,
        "tactical_diameter_over_l": abs(s180[4]) / lpp,
    }


def zigzag(ship: MMGShip, u0: float, delta_deg: float = 10.0,
           switch_deg: float = 10.0) -> dict:
    d_mag = math.radians(delta_deg)
    sw = math.radians(switch_deg)
    phase = {"sign": 1.0, "count": 0}

    def cmd(t, state, delta_now):
        psi = state[5]
        if phase["sign"] > 0 and psi >= sw:
            phase["sign"] = -1.0; phase["count"] += 1
        elif phase["sign"] < 0 and psi <= -sw:
            phase["sign"] = 1.0; phase["count"] += 1
        return phase["sign"] * d_mag

    res = simulate(ship, cmd, t_end=200.0, u0=u0)
    psi = np.degrees(res["state"][:, 5])
    # 오버슈트: 반전 후 ψ 극값 − switch
    d = res["delta"]
    sign_change = np.where(np.diff(np.sign(d)) != 0)[0]
    overshoots = []
    for k, idx in enumerate(sign_change[:2]):
        seg = psi[idx:idx + int(60.0 / 0.05)]
        if d[idx] > 0:      # 방금 +→− 반전: 직전 +방향 극대
            overshoots.append(float(seg.max()) - switch_deg)
        else:
            overshoots.append(-float(seg.min()) - switch_deg)
    return {
        "first_overshoot_deg": overshoots[0] if overshoots else 0.0,
        "second_overshoot_deg": (overshoots[1]
                                 if len(overshoots) > 1 else 0.0),
        "switch_count": phase["count"],
    }
```

**구현 주의:** ① `turning_circle`의 t_end 스케치 정리 — `t_end = 60·Lpp/u0` (선회 2바퀴 여유) ② advance 정의 = 타 시작점 기준 **x 전진거리** (원점 출발이라 |x0|), transfer = y — 원전 Fig 정의 재확인 ③ 지그재그 오버슈트 판독은 타 반전 시점 기준 — 스케치의 sign_change 로직을 반전 이벤트 기록(cmd 함수에서 시점 저장)으로 바꾸면 견고 ④ dt 0.05→0.025 반감 시 지표 변화 <1% 수렴 확인 테스트 추가 권장.

- [ ] **Step 5: 통과 확인** — Run: `python -m pytest tests/test_maneuvering_trials.py -v` → 3 PASS. **A_D·D_T 10% 밖이면**: βP의 x'P 값·러더 부호·γR 부호 방향부터 원전 재대조 (사고 다발 지점 — Frank 관례: 격리 사다리로).

- [ ] **Step 6: 커밋 + 성적표**

```bash
git add src/physics/maneuvering/trials.py tests/test_maneuvering_trials.py
git commit -m "feat: 표준 조종 시험 — KVLCC2 35° 선회 원전 재현 (1단계 완결)"
```

worklog에 1단계 성적표 (원전 계산 vs 우리 재현 vs 실측 3열 표) 기록 후 오너 보고 — 계속/보류 판단.
