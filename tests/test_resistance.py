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


def test_michell_ultra_low_froude_is_zero():
    """극저속(Fn<0.1) 가드 — 수치 폭주 지뢰의 회귀 시험 (2026-08-02).

    Michell 적분은 Fn→0에서 피적분 함수가 무한 진동해 격자가 못 따라감
    — Fn 0.02에서 107 N짜리 쓰레기값이 나와 시뮬 배가 '가짜 평형'에
    갇혔던 실측 사고. 물리적으로 이 영역 조파저항은 무시 가능 → 0."""
    r = michell_wave_resistance(**WIGLEY, speed=_fn_to_speed(0.02))
    assert r == 0.0
    # 가드 경계 위는 정상 양수
    assert michell_wave_resistance(**WIGLEY, speed=_fn_to_speed(0.12)) > 0


def test_resistance_curve_monotone_for_sim():
    """시뮬용 저항곡선(0~2 m/s)이 단조 증가 — 가짜 평형 재발 방지."""
    speeds = np.linspace(0.05, 2.0, 12)
    rs = [michell_wave_resistance(**WIGLEY, speed=float(s)) +
          frictional_resistance(float(s), 4.0, 3.0) for s in speeds]
    assert all(b >= a for a, b in zip(rs, rs[1:]))


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


def test_mesh_michell_matches_analytic_wigley():
    """일반화 검증의 핵심: 메쉬판 Michell을 해석판과 같은 Wigley에서 대조.

    해석판은 Wigley 문헌 벤치마크를 통과한 기준 — 메쉬판이 이를 5% 내로
    재현하면 임의 형상(Ship-D)에도 신뢰 근거 확보.
    """
    from src.ai.hull_generator import generate_hull_mesh
    from src.core.types import MainDimensions
    from src.physics.resistance import michell_wave_resistance_mesh

    dims = MainDimensions(loa=4.0, beam=1.0, depth=0.40, draft_design=0.25,
                          cb=0.50)
    mesh = generate_hull_mesh(dims, n_stations=121, n_below=29)
    from src.ai.hull_generator import solve_exponents

    n, m = solve_exponents(dims.cb)
    for fn in (0.20, 0.316):
        v = _fn_to_speed(fn)
        analytic = michell_wave_resistance(4.0, 1.0, 0.25, n, m,
                                           draft=0.20, speed=v)
        from_mesh = michell_wave_resistance_mesh(mesh, draft=0.20, speed=v)
        assert from_mesh == pytest.approx(analytic, rel=0.05), f"Fn={fn}"


def test_total_resistance_report():
    from src.ai.hull_generator import generate_hull_mesh, solve_exponents
    from src.core.types import MainDimensions
    from src.physics.resistance import total_resistance

    dims = MainDimensions(loa=4.0, beam=1.3, depth=0.48, draft_design=0.30,
                          cb=0.50)
    mesh = generate_hull_mesh(dims)
    n, m = solve_exponents(dims.cb)
    rep = total_resistance(mesh, dims, n, m, draft=0.20, speed=1.5)
    assert rep.total == pytest.approx(rep.rf + rep.rw, rel=1e-9)
    assert rep.effective_power == pytest.approx(rep.total * 1.5, rel=1e-9)
    assert rep.rf > 0 and rep.rw > 0
    assert rep.wetted_area > 0
