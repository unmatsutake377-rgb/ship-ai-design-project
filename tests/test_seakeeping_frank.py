"""Frank close-fit 검증 — 반원 Ursell 이중검증 (스펙 §9 구도)."""
import math

import numpy as np
import pytest

from src.physics.seakeeping.frank import heave_coefficients_frank
from src.physics.seakeeping.ursell import heave_coefficients


def _semicircle(n=32):
    angs = np.linspace(math.pi, 2 * math.pi, n + 1)
    return [(math.cos(a), math.sin(a)) for a in angs]


def test_frank_matches_ursell_semicircle():
    """반원 대조 (독립 정식화 이중검증): 저·중주파에서 a33 4%·
    b33 2% 이내 (32 세그먼트). 고주파 수렴은 seg 96에서 1.5%
    (worklog 08-09 실측 — 파장↓ → 세그먼트↑ 필요, 물리 정합)."""
    for xi, tol_b in ((0.25, 0.02), (1.0, 0.02)):
        om = math.sqrt(xi * 9.81)
        f = heave_coefficients_frank(_semicircle(), om)
        u = heave_coefficients(1.0, om)
        assert f.added_mass == pytest.approx(u.added_mass, rel=0.04), xi
        assert f.damping == pytest.approx(u.damping, rel=tol_b), xi


def test_frank_rectangular_section_properties():
    """임의 단면 능력 (Lewis 밖 검증): 직사각형 단면 — 부가질량·
    감쇠 양수 + 같은 폭 반원보다 부가질량 큼 (풍만 단면)."""
    b2, t = 1.0, 1.0
    n = 12
    pts = ([(-b2, 0.0)] + [(-b2, -t * (i + 1) / n) for i in range(n)]
           + [(-b2 + 2 * b2 * (i + 1) / (2 * n), -t)
              for i in range(2 * n)]
           + [(b2, -t + t * (i + 1) / n) for i in range(n)])
    om = math.sqrt(1.0 * 9.81)
    f = heave_coefficients_frank(pts, om)
    u = heave_coefficients(1.0, om)     # 같은 반폭 반원
    assert f.added_mass > 0 and f.damping > 0
    assert f.added_mass > u.added_mass
