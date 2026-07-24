import numpy as np
import pytest
import trimesh

from src.physics.resistance import (
    FORM_FACTOR,
    NU_SEAWATER,
    frictional_resistance,
    ittc_cf,
    michell_wave_resistance,
    reynolds,
    wetted_surface,
)

# 표준 Wigley: n=2, m=2 (Cb≈0.444), L/B=10, B/T=1.6 — 문헌 벤치마크 선형
WIGLEY = dict(loa=4.0, beam=0.4, draft_design=0.25, n=2.0, m=2.0, draft=0.25)


def _fn_to_speed(fn, loa=4.0):
    return fn * (9.81 * loa) ** 0.5


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


def test_michell_positive():
    rw = michell_wave_resistance(**WIGLEY, speed=_fn_to_speed(0.316))
    assert rw > 0


def test_michell_convergence():
    """격자 2배 세분 시 변화 < 2% — 수치적분 수렴 확인."""
    v = _fn_to_speed(0.316)
    r1 = michell_wave_resistance(**WIGLEY, speed=v)
    r2 = michell_wave_resistance(**WIGLEY, speed=v, n_u=240, n_x=320, n_z=160)
    assert abs(r2 - r1) / r2 < 0.02


def test_michell_low_froude_vanishes():
    """조파저항은 저속에서 급감 — Fn 0.1은 Fn 0.316의 1/5 미만."""
    r_low = michell_wave_resistance(**WIGLEY, speed=_fn_to_speed(0.10))
    r_ref = michell_wave_resistance(**WIGLEY, speed=_fn_to_speed(0.316))
    assert r_low < 0.2 * r_ref


def test_michell_magnitude_band():
    """표준 Wigley Fn=0.316: 자릿수 검증 밴드 (문헌 Cw ~1e-3 대역).

    ½ρV²S 기준 Cw로 환산해 [4e-4, 4e-3] 밴드 — 계수/단위 오류(×10)를 잡는
    안전망. 정밀 문헌값 대조는 후속 정제 (Michell은 실험 대비 과대예측 경향).
    """
    v = _fn_to_speed(0.316)
    rw = michell_wave_resistance(**WIGLEY, speed=v)
    s_approx = 4.0 * (0.4 + 2 * 0.25) * 0.75  # 개략 침수면적 [m²]
    cw = rw / (0.5 * 1025.0 * v ** 2 * s_approx)
    assert 4e-4 < cw < 4e-3
