"""표준 조종 시험 실행기 (조종성 1단계, 스펙 §2).

선회권: 타 35° → ψ=90° 시점 전진거리(advance)·횡거리(transfer),
ψ=180° 시점 횡거리 = 선회지름(tactical diameter) — L 배수 무차원.
지그재그: 타 δ↔−δ (ψ가 ±switch 도달 시 반전) → 오버슈트 각.
적분: RK4 고정 스텝. 타속 제한: 원전 p12 — 실선 1.76°/s
(모델은 Froude 환산 ×√λ), 프로펠러 회전수는 접근 속도 값 고정.
"""
from __future__ import annotations

import math

import numpy as np

from src.physics.maneuvering.mmg import MMGShip, derivatives

RUDDER_RATE_FULLSCALE_DPS = 1.76    # 원전 p12


def _rk4_step(ship, state, delta, dt):
    k1 = derivatives(ship, state, delta)
    k2 = derivatives(ship, state + 0.5 * dt * k1, delta)
    k3 = derivatives(ship, state + 0.5 * dt * k2, delta)
    k4 = derivatives(ship, state + dt * k3, delta)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(ship: MMGShip, delta_cmd_fn, t_end: float, u0: float,
             rudder_rate_dps: float, dt: float = 0.05) -> dict:
    """delta_cmd_fn(t, state) → 명령 타각 [rad]. 실제 타각은
    타속 제한으로 추종."""
    rate = math.radians(rudder_rate_dps)
    state = np.array([u0, 0.0, 0.0, 0.0, 0.0, 0.0])
    delta = 0.0
    n = int(t_end / dt)
    ts = np.empty(n)
    states = np.empty((n, 6))
    deltas = np.empty(n)
    for i in range(n):
        t = i * dt
        cmd = delta_cmd_fn(t, state)
        step = min(max(cmd - delta, -rate * dt), rate * dt)
        delta += step
        ts[i] = t
        states[i] = state
        deltas[i] = delta
        state = _rk4_step(ship, state, delta, dt)
    return {"t": ts, "state": states, "delta": deltas, "dt": dt}


def _state_at_psi(res, psi_target: float) -> np.ndarray:
    """|ψ|가 목표에 처음 닿는 시점의 상태 (선형 보간)."""
    psi = np.abs(res["state"][:, 5])
    idx_arr = np.where(psi >= abs(psi_target))[0]
    if len(idx_arr) == 0 or idx_arr[0] == 0:
        raise ValueError("ψ 목표 미도달 — t_end 부족")
    i = idx_arr[0]
    w = (abs(psi_target) - psi[i - 1]) / max(psi[i] - psi[i - 1], 1e-12)
    s = res["state"]
    return s[i - 1] + w * (s[i] - s[i - 1])


def turning_circle(ship: MMGShip, u0: float, delta_deg: float = 35.0,
                   rudder_rate_dps: float = 12.0,
                   dt: float = 0.05) -> dict:
    """35° 선회권 — advance·transfer·tactical diameter (L 배수)."""
    d_cmd = math.radians(delta_deg)
    t_end = 60.0 * ship.par.lpp / u0        # 선회 2바퀴 여유
    res = simulate(ship, lambda t, s: d_cmd, t_end, u0,
                   rudder_rate_dps, dt=dt)
    s90 = _state_at_psi(res, math.pi / 2.0)
    s180 = _state_at_psi(res, math.pi)
    lpp = ship.par.lpp
    return {
        "advance_over_l": abs(s90[3]) / lpp,
        "transfer_over_l": abs(s90[4]) / lpp,
        "tactical_diameter_over_l": abs(s180[4]) / lpp,
    }


def zigzag(ship: MMGShip, u0: float, delta_deg: float = 10.0,
           switch_deg: float = 10.0, rudder_rate_dps: float = 12.0,
           dt: float = 0.05) -> dict:
    """지그재그 — 타 반전 이벤트 기록 → 반전 후 ψ 극값으로
    오버슈트 판독."""
    d_mag = math.radians(delta_deg)
    sw = math.radians(switch_deg)
    events: list[int] = []            # 반전 시점 스텝 인덱스
    phase = {"sign": 1.0, "i": 0}

    def cmd(t, state):
        psi = state[5]
        if phase["sign"] > 0 and psi >= sw:
            phase["sign"] = -1.0
            events.append(phase["i"])
        elif phase["sign"] < 0 and psi <= -sw:
            phase["sign"] = 1.0
            events.append(phase["i"])
        phase["i"] += 1
        return phase["sign"] * d_mag

    t_end = 60.0 * ship.par.lpp / u0
    res = simulate(ship, cmd, t_end, u0, rudder_rate_dps, dt=dt)
    psi_deg = np.degrees(res["state"][:, 5])
    horizon = int(20.0 * ship.par.lpp / u0 / dt)   # 극값 탐색 창
    overshoots = []
    for k, idx in enumerate(events[:2]):
        seg = psi_deg[idx:idx + horizon]
        if len(seg) < 2:
            break
        if k % 2 == 0:      # +방향 진행 중 반전 → ψ 극대
            overshoots.append(float(seg.max()) - switch_deg)
        else:               # −방향 → ψ 극소
            overshoots.append(-float(seg.min()) - switch_deg)
    return {
        "first_overshoot_deg": overshoots[0] if overshoots else 0.0,
        "second_overshoot_deg": (overshoots[1]
                                 if len(overshoots) > 1 else 0.0),
        "switch_count": len(events),
    }
