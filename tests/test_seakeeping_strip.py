"""스트립 이론 RAO 검증 — 해석 극한 앵커 (내항성 2단계)."""
import math

import pytest

from src.ai.hull_generator import generate_hull_mesh
from src.core.types import MainDimensions
from src.physics.seakeeping.strip import heave_pitch_rao


@pytest.fixture(scope="module")
def wigley_rao():
    dims = MainDimensions(loa=3.0, beam=0.9, depth=0.5, draft_design=0.3,
                          cb=0.45)
    mesh = generate_hull_mesh(dims, cm=0.85)
    vol = 0.45 * 3.0 * 0.9 * 0.3
    m = 1025.0 * vol
    iyy = m * (0.25 * 3.0) ** 2
    lam_ls = (8.0, 2.0, 1.2, 0.5)
    oms = [math.sqrt(9.81 * 2 * math.pi / (ll * 3.0)) for ll in lam_ls]
    raos = heave_pitch_rao(mesh, 0.3, m, iyy, oms, n_stations=9,
                           contour_n=10)
    return dict(zip(lam_ls, raos))


def test_long_wave_limit_follows_wave(wigley_rao):
    """장파 극한 (λ/L=8): 배가 파면을 그대로 탐 — heave→1·pitch→1
    (해석 확정 앵커: 정적 평형 추종)."""
    r = wigley_rao[8.0]
    assert r.heave_rao == pytest.approx(1.0, abs=0.05)
    assert r.pitch_rao == pytest.approx(1.0, abs=0.08)


def test_short_wave_limit_vanishes(wigley_rao):
    """단파 극한 (λ/L=0.5): 파가 배보다 짧으면 상쇄 — RAO→0."""
    r = wigley_rao[0.5]
    assert r.heave_rao < 0.15
    assert r.pitch_rao < 0.10


def test_monotone_transition_and_pitch_peak(wigley_rao):
    """중간 대역: heave 단조 감소 전이 + pitch 피크(>1) 존재
    (λ/L 1.2 근방 — 문헌 Wigley 계보 대역)."""
    assert wigley_rao[8.0].heave_rao > wigley_rao[2.0].heave_rao \
        > wigley_rao[1.2].heave_rao > wigley_rao[0.5].heave_rao
    assert wigley_rao[1.2].pitch_rao > 1.1
