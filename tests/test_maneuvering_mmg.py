"""MMG 힘 모델 — 직진 평형·대칭 자기검증 (스펙 §3 앵커 ①②)."""
import numpy as np
import pytest

from src.physics.maneuvering.kvlcc2 import KVLCC2_COEFFS, KVLCC2_L7

U0 = 1.179          # L7 접근 속도 [m/s] = 15.5kn/√45.7 (원전 §3.2)


def _ship():
    from src.physics.maneuvering.mmg import MMGShip, solve_self_propulsion
    r0 = 8.0        # 직진 평형 항등식엔 임의값 무방 (Task 3에서 확정)
    n_p = solve_self_propulsion(KVLCC2_L7, KVLCC2_COEFFS, U0, r0)
    return MMGShip(par=KVLCC2_L7, co=KVLCC2_COEFFS, r0_n=r0,
                   n_p=n_p, u0=U0)


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
    """±δ 거울: 우현타 → 회두 반대 방향·크기 동일.

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


def test_drift_produces_yaw_moment():
    """사항(v_m≠0) → 선체 회두 모멘트 발생 (N'v 작동 확인)."""
    from src.physics.maneuvering.mmg import derivatives
    ship = _ship()
    state = np.array([U0, -0.1, 0.0, 0.0, 0.0, 0.0])
    d = derivatives(ship, state, delta=0.0)
    assert d[2] != 0.0
