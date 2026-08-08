"""횡요 고유주기 — IMO 규정식 손계산 앵커."""
import pytest

from src.physics.seakeeping.criteria import imo_roll_c, roll_natural_period


def test_imo_c_hand_calc():
    """c 손계산: B 16.2·T 5.9·L 98.9 →
    0.373 + 0.023·2.746 − 0.043·0.989 = 0.394."""
    c = imo_roll_c(16.2, 5.9, 98.9)
    assert c == pytest.approx(0.373 + 0.023 * (16.2 / 5.9)
                              - 0.043 * 0.989, rel=1e-9)
    assert c == pytest.approx(0.394, abs=0.001)


def test_roll_period_hand_calc_cargo():
    """100 m 화물선 (GM 0.73 m): T = 2·0.394·16.2/√0.73 ≈ 14.9 s —
    상선 통상 대역 (10~25 s) 정합."""
    t = roll_natural_period(16.2, 5.9, 98.9, gm=0.73)
    assert t == pytest.approx(2 * 0.394 * 16.2 / 0.73 ** 0.5, rel=1e-3)
    assert 10.0 < t < 25.0


def test_stiff_ship_short_period():
    """성질: GM 크면 주기 짧음 (뻣뻣 — 급횡요) + GM≤0은 발산 정직."""
    soft = roll_natural_period(1.0, 0.3, 3.0, gm=0.05)
    stiff = roll_natural_period(1.0, 0.3, 3.0, gm=0.40)
    assert stiff < soft
    assert roll_natural_period(1.0, 0.3, 3.0, gm=0.0) == float("inf")


def test_seakeeping_gate_e2e():
    """5번째 게이트 관통: Wigley 조사선 — 연안 해상에서 성적·합불."""
    from src.ai.hull_generator import generate_hull_mesh
    from src.core.types import MainDimensions
    from src.physics.seakeeping.criteria import seakeeping_gate

    dims = MainDimensions(loa=3.0, beam=0.9, depth=0.5, draft_design=0.3,
                          cb=0.45)
    mesh = generate_hull_mesh(dims, cm=0.85)
    m = 1025.0 * 0.45 * 3.0 * 0.9 * 0.3
    g = seakeeping_gate(mesh, 0.3, m, m * 0.5625, beam=0.9, lwl=3.0,
                        gm=0.15, purpose="survey")
    assert set(g["checks"]) == {"roll", "pitch", "heave"}
    assert isinstance(g["passed"], bool)
    assert g["sig_heave_m"] > 0 and g["roll_period_s"] > 0
