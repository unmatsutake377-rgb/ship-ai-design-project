"""조종성 지표 — 손계산 정답지 + 시뮬 교차 대조."""
import numpy as np
import pytest

from src.physics.agility import agility_metrics


def test_hand_calc():
    """r_ss = (F·sep/2)/(nr − nv·m_x·u/yv), D = 2u/r_ss — 손계산.

    값: F=40, sep=0.8 → M=16. 분모 = 50 − 5·100·1/100 = 45.
    r_ss = 16/45 ≈ 0.3556. v_ss = 100·1·0.3556/100 = 0.3556.
    대지속도 = √(1+0.3556²) ≈ 1.0613. D = 2·1.0613/0.3556 ≈ 5.970."""
    a = agility_metrics(izz_total=200.0, m_x=100.0, yv=100.0, nv=5.0,
                        nr=50.0, thrust_max=40.0, thruster_sep=0.8,
                        speed=1.0, loa=3.0)
    assert a.nomoto_t == pytest.approx(200.0 / 50.0)
    assert a.turn_rate_max == pytest.approx(16.0 / 45.0)
    assert a.turning_diameter == pytest.approx(5.9698, rel=1e-3)
    assert a.within_imo is True
    assert a.coupled_unstable is False


def test_coupling_tightens_turn():
    """횡활주 결합(nv>0)이 선회를 조임 — nv=0 대비 지름 감소."""
    base = dict(izz_total=200.0, m_x=100.0, yv=100.0, nr=50.0,
                thrust_max=40.0, thruster_sep=0.8, speed=1.0, loa=3.0)
    d0 = agility_metrics(nv=0.0, **base).turning_diameter
    d1 = agility_metrics(nv=5.0, **base).turning_diameter
    assert d1 < d0


def test_unstable_regime_flagged():
    """분모≤0 (고속·강결합) → 방향 불안정 플래그."""
    a = agility_metrics(izz_total=200.0, m_x=100.0, yv=100.0, nv=60.0,
                        nr=50.0, thrust_max=40.0, thruster_sep=0.8,
                        speed=1.0, loa=3.0)
    assert a.coupled_unstable is True


def test_rejects_nonpositive_damping():
    with pytest.raises(ValueError):
        agility_metrics(200.0, 100.0, -1.0, 5.0, 50.0, 40.0, 0.8, 1.0, 3.0)


def test_formula_matches_simulation():
    """공식 선회지름 vs 시뮬 궤적 실측 지름 — 같은 방정식이니 ±10%.

    지표가 종이 위 숫자가 아니라 시뮬 배의 실거동과 맞는지 확인."""
    import pandas as pd

    from src.core.types import GoalSpec
    from src.hitl.duel_media import report_for_dims
    from src.optimize import dims_from_vector
    from src.sim_adapters.python_sim import step, vessel_from_report

    df = pd.read_csv("outputs/pareto/pareto.csv")
    row = df.loc[df.total_mass_kg.idxmin()]
    dims = dims_from_vector(np.array([row.loa, row.lb, row.bt, row.cb]))
    v = vessel_from_report(report_for_dims(dims, GoalSpec(1.2, 100.0,
                                                          "survey")))
    state = np.zeros(6)
    for _ in range(400):                     # 직진 도달
        state = step(v, state, v.thrust_max, v.thrust_max, 0.05)
    xs, ys = [], []
    for _ in range(6000):                    # 전타 고정 (좌 0 / 우 최대)
        state = step(v, state, v.thrust_max, 0.0, 0.05)
        xs.append(state[0])
        ys.append(state[1])
    xs, ys = np.array(xs[-3000:]), np.array(ys[-3000:])  # 정착 후 원
    d_sim = 0.5 * ((xs.max() - xs.min()) + (ys.max() - ys.min()))

    u_turn = float(state[3])                 # 선회 중 실속도 (감속 반영)
    a = agility_metrics(v.i_z, v.m_x, v.yv, v.nv, v.nr,
                        v.thrust_max, v.thruster_sep,
                        speed=u_turn, loa=dims.loa)
    assert d_sim == pytest.approx(a.turning_diameter, rel=0.10)
