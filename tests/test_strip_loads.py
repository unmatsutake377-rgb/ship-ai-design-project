"""스트립 동적 굽힘 RAO — 폐합 항등식 + 준정적 교차 (스펙 §2·§3)."""
import numpy as np
import pytest
import trimesh

G = 9.81
RHO = 1025.0


def _barge_setup(loa=80.0, beam=10.0, t=2.0):
    # station_geometry는 점 판독 — 상자 원시 꼭짓점은 희소라
    # 세분화로 조밀화 (실선 메쉬는 원래 조밀)
    mesh = trimesh.creation.box(
        extents=[loa, beam, 6.0]).subdivide_to_size(max_edge=1.0)
    mass = RHO * loa * beam * t
    iyy = mass * loa ** 2 / 12.0     # 균일 μ와 정합 (모멘트 폐합)
    wl_z = -3.0 + t
    blocks = [(mass, -loa / 2.0, loa / 2.0)]
    return mesh, mass, iyy, wl_z, blocks


def test_bending_balance_identity():
    """자유 운동 해에서 V 양끝 잔차 = 기계 정밀도 — 운동방정식
    총평형의 내부 하중 버전 (상시 자기검증). 힘 폐합은 EOM과 같은
    적분 가중치라 해석적으로 정확히 0; 모멘트는 μ 블록-스테이션
    절단(margin 2%) 오차만 남음 (실측 ~6%)."""
    from src.physics.structure.strip_loads import wave_bending_rao
    mesh, mass, iyy, wl_z, blocks = _barge_setup()
    omega = float(np.sqrt(2.0 * np.pi * G / 80.0))    # λ = L
    out = wave_bending_rao(mesh, wl_z, mass, iyy, blocks, [omega])
    assert out[0]["balance_v"] < 1e-9
    assert out[0]["balance_m"] < 0.10


def test_restrained_vs_quasi_static_band():
    """구속 모드(운동 0) 스트립 굽힘 vs 준정적 표준파 — λ=L 대역
    교차검증 (실측 비 ≈ 0.65).

    두 방법은 같은 양의 다른 물리: 준정적 = 정수압 쐐기 (깊이 감쇠
    없음), 스트립 FK = Smith 효과 e^{−kT} + 동적 관성 항 → 준정적이
    보수적 상한. 완전 일치는 kT→0 극한에서만 — 대역 판정이 정직."""
    from src.physics.structure.strip_loads import wave_bending_rao
    from src.physics.structure.wave_loads import quasi_static_wave_moment
    mesh, mass, iyy, wl_z, blocks = _barge_setup()
    amp = 0.5
    omega = float(np.sqrt(2.0 * np.pi * G / 80.0))
    strip = wave_bending_rao(mesh, wl_z, mass, iyy, blocks, [omega],
                             restrained=True)
    qs = quasi_static_wave_moment(mesh, wl_z, blocks, wave_amp=amp,
                                  wavelength=80.0, n=201)
    ratio = (strip[0]["m_mid_per_amp_nm"] * amp
             / abs(qs["m_wave_mid_nm"]))
    assert 0.5 < ratio < 1.0


def test_three_way_cross_check_100m_cargo(tmp_path):
    """100m 화물선 — IACS vs 준정적 vs 스트립 같은 자릿수 (성적표).

    표준파 a = L/40 (H = L/20 고전 관례). IACS는 극치 통계
    (10⁻⁸ 확률) 내장이라 규칙파 한 발과 수배 차이가 정상 —
    자릿수(0.2~5배) 합의만 판정, 값은 성적표 기록."""
    from src.core.types import GoalSpec
    from src.physics.structure.loads import standard_weight_blocks
    from src.physics.structure.strip_loads import wave_bending_rao
    from src.physics.structure.wave_loads import (
        iacs_wave_bending_knm,
        quasi_static_wave_moment,
    )
    from src.pipeline import run_pipeline

    goal = GoalSpec(target_speed_ms=7.0, payload_kg=5_000_000.0,
                    purpose="cargo")
    report = run_pipeline(goal, tmp_path, seakeeping=False)
    mesh = trimesh.load(tmp_path / report["mesh_file"])
    d = report["dimensions"]
    lg = report["large"]
    loa, beam = d["loa"], d["beam"]
    zmin = float(mesh.bounds[0][2])
    wl_z = zmin + lg["draft"]
    ls = lg["lightship_t"]
    comp = {"structure": ls["structure"] * 1e3,
            "outfit": ls["outfit"] * 1e3,
            "machinery": ls["machinery"] * 1e3,
            "fuel": lg["fuel_t"] * 1e3,
            "payload": lg["payload_t"] * 1e3}
    xmin = float(mesh.bounds[0][0])
    blocks = standard_weight_blocks(comp, xmin, loa)

    amp = loa / 40.0
    hog, _sag = iacs_wave_bending_knm(loa, beam, d["cb"])
    qs = quasi_static_wave_moment(mesh, wl_z, blocks, wave_amp=amp,
                                  wavelength=loa)
    mass = sum(comp.values())
    iyy = mass * (0.25 * loa) ** 2
    omega = float(np.sqrt(2.0 * np.pi * G / loa))
    st = wave_bending_rao(mesh, wl_z, mass, iyy, blocks, [omega],
                          restrained=True)
    m_strip = st[0]["m_mid_per_amp_nm"] * amp

    m_iacs = hog * 1e3                       # kN·m → N·m
    m_qs = abs(qs["m_wave_mid_nm"])
    print(f"\n[3중 교차] IACS {m_iacs:.3e} / 준정적 {m_qs:.3e} "
          f"/ 스트립(구속) {m_strip:.3e} N·m")
    assert 0.2 < m_qs / m_iacs < 5.0
    assert 0.2 < m_strip / m_iacs < 5.0


def test_free_less_than_restrained():
    """자유 운동은 파면을 타서 하중 경감 — |M_free| < |M_restrained|
    (장파 방향 성질)."""
    from src.physics.structure.strip_loads import wave_bending_rao
    mesh, mass, iyy, wl_z, blocks = _barge_setup()
    omega = float(np.sqrt(2.0 * np.pi * G / 160.0))   # λ = 2L 장파측
    free = wave_bending_rao(mesh, wl_z, mass, iyy, blocks, [omega])
    rest = wave_bending_rao(mesh, wl_z, mass, iyy, blocks, [omega],
                            restrained=True)
    assert free[0]["m_mid_per_amp_nm"] < rest[0]["m_mid_per_amp_nm"]
