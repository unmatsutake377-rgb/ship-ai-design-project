"""스펙트럼·불규칙 응답 검증 — 항등식·공진 해석값 앵커."""
import math

import pytest

from src.physics.seakeeping.waves import (
    ittc_spectrum,
    roll_rao,
    significant_amplitude,
    significant_roll_deg,
    spectral_moment,
)


def test_spectrum_m0_identity():
    """항등식: m0 = Hs²/16 — 유의파고 정의 정합 (공식 계수의 심판)."""
    for hs, tz in ((2.0, 7.0), (4.0, 9.0), (1.0, 5.0)):
        m0 = spectral_moment(hs, tz, 0)
        assert m0 == pytest.approx(hs * hs / 16.0, rel=0.02), (hs, tz)


def test_roll_resonance_analytic():
    """공진 해석값: RAO(ωn) = 1/(2κ)."""
    t_roll = 8.0
    wn = 2 * math.pi / t_roll
    assert roll_rao(wn, t_roll, kappa=0.06) == pytest.approx(
        1.0 / (2 * 0.06), rel=1e-9)
    # 저주파 극한: 파면 경사 추종 (RAO→1)
    assert roll_rao(0.05 * wn, t_roll) == pytest.approx(1.0, abs=0.01)


def test_significant_response_linear_in_wave_height():
    """선형계: 유의 응답은 파고에 비례."""
    rao = lambda o: 1.0 / (1.0 + o * o)
    r2 = significant_amplitude(2.0, 7.0, rao)
    r4 = significant_amplitude(4.0, 7.0, rao)
    assert r4 == pytest.approx(2.0 * r2, rel=1e-6)


def test_significant_roll_sane_band():
    """100 m 화물선 (T_roll 14.9 s), 해상 Hs 3 m·Tz 8 s — 유의 roll
    수 도(°) 대역 (상선 통상: 공진 이탈 조건에서 한 자릿수)."""
    r = significant_roll_deg(3.0, 8.0, 14.9)
    assert 0.5 < r < 15.0


def test_seakeeping_report_end_to_end():
    """관통: Wigley 3 m 배, 해상 Hs 0.5 m·Tz 3 s — 성적 3종 산출,
    유의 heave < Hs (감쇠 계 — 파고보다 클 수 없는 대역) + 양수."""
    from src.ai.hull_generator import generate_hull_mesh
    from src.core.types import MainDimensions
    from src.physics.seakeeping.waves import seakeeping_report

    dims = MainDimensions(loa=3.0, beam=0.9, depth=0.5, draft_design=0.3,
                          cb=0.45)
    mesh = generate_hull_mesh(dims, cm=0.85)
    m = 1025.0 * 0.45 * 3.0 * 0.9 * 0.3
    rep = seakeeping_report(mesh, 0.3, m, m * 0.5625, beam=0.9, lwl=3.0,
                            gm=0.15, hs=0.5, tz=3.0)
    assert 0.0 < rep["sig_heave_m"] < 0.5 * 1.2
    assert 0.0 < rep["sig_pitch_deg"] < 30.0
    assert 0.0 < rep["sig_roll_deg"] < 40.0
    assert rep["roll_period_s"] > 0
