"""미드십 단면계수 조립 — 상자 손계산 앵커 (스펙 §3)."""
import pytest


def test_symmetric_box_hand_calc():
    """대칭 상자 (선저=갑판 t, 선측 t): 중립축 = D/2 정확,
    I 손계산 대조.

    B=10, D=6, t=10mm 전둘레: 선저·갑판 A=0.1 m² 각각 z=0,6 →
    I_판 = 2·0.1·3² = 1.8, 선측 2장 I = 2·(0.01·6³/12) = 0.36,
    합 2.16 m⁴. Z = I/3 = 0.72 m³."""
    from src.physics.structure.midship import assemble_midship
    sec = assemble_midship(10.0, 6.0, 10.0, 10.0, 10.0,
                           n_bottom_long=0, n_deck_long=0,
                           long_area_cm2=0.0)
    assert sec.neutral_axis_m == pytest.approx(3.0, rel=1e-6)
    assert sec.inertia_m4 == pytest.approx(2.16, rel=0.01)
    assert sec.z_deck_m3 == pytest.approx(0.72, rel=0.01)
    assert sec.z_keel_m3 == pytest.approx(0.72, rel=0.01)


def test_asymmetric_thickness_shifts_neutral_axis():
    """선저를 두껍게 → 중립축 하강, Z_deck < Z_keel 역전 방향."""
    from src.physics.structure.midship import assemble_midship
    sec = assemble_midship(10.0, 6.0, 20.0, 10.0, 10.0,
                           n_bottom_long=0, n_deck_long=0,
                           long_area_cm2=0.0)
    assert sec.neutral_axis_m < 3.0
    assert sec.z_deck_m3 < sec.z_keel_m3


def test_longitudinals_add_inertia():
    """종늑골 추가 → I 증가 (부재는 공짜가 아니다)."""
    from src.physics.structure.midship import assemble_midship
    bare = assemble_midship(10.0, 6.0, 10.0, 10.0, 10.0, 0, 0, 0.0)
    stiff = assemble_midship(10.0, 6.0, 10.0, 10.0, 10.0,
                             n_bottom_long=10, n_deck_long=10,
                             long_area_cm2=20.0)
    assert stiff.inertia_m4 > bare.inertia_m4
