import numpy as np
import pytest
import trimesh

from src.physics.resistance import (
    FORM_FACTOR,
    NU_SEAWATER,
    frictional_resistance,
    ittc_cf,
    reynolds,
    wetted_surface,
)


def test_ittc_cf_exact():
    # Re=1e8: Cf = 0.075/(log10(1e8)-2)^2 = 0.075/36
    assert ittc_cf(1e8) == pytest.approx(0.075 / 36.0, rel=1e-12)


def test_reynolds():
    assert reynolds(1.5, 4.0) == pytest.approx(1.5 * 4.0 / NU_SEAWATER, rel=1e-12)


def test_frictional_positive_and_scales():
    """Rf ~ V^1.8~2.0 스케일링 (ITTC 마찰 물리 불변량)."""
    s, L, S = 1.5, 4.0, 3.0
    r1 = frictional_resistance(s, L, S)
    r2 = frictional_resistance(2 * s, L, S)
    assert r1 > 0
    ratio = r2 / r1
    assert 2 ** 1.7 < ratio < 2 ** 2.0


def test_wetted_surface_box():
    """바지선 해석해: S = L·B + 2(L+B)·T (바닥 + 측면, 캡 제외)."""
    L, B, D, t = 4.0, 1.2, 0.6, 0.3
    box = trimesh.creation.box(extents=[L, B, D])
    box.apply_translation([0, 0, D / 2])
    expected = L * B + 2 * (L + B) * t
    assert wetted_surface(box, t) == pytest.approx(expected, rel=1e-3)
