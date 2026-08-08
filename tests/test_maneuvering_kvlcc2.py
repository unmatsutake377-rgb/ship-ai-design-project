"""KVLCC2 원전 데이터 박제 검증 (Yasukawa & Yoshimura 2015)."""
import pytest


def test_particulars_table1():
    """Table 1 (p5) 대표값 — L7 모델."""
    from src.physics.maneuvering.kvlcc2 import KVLCC2_L7
    s = KVLCC2_L7
    assert s.lpp == pytest.approx(7.00)
    assert s.beam == pytest.approx(1.27)
    assert s.draft == pytest.approx(0.46)
    assert s.cb == pytest.approx(0.810)
    assert s.ar == pytest.approx(0.0539)


def test_coeffs_table3():
    """Table 3 (p12) 대표값 — 부호 포함."""
    from src.physics.maneuvering.kvlcc2 import KVLCC2_COEFFS
    c = KVLCC2_COEFFS
    assert c.yv == pytest.approx(-0.315)
    assert c.xvvvv == pytest.approx(0.771)
    assert c.nrrr == pytest.approx(-0.013)
    assert c.a_h == pytest.approx(0.312)
    assert c.eps == pytest.approx(1.09)
    assert c.f_alpha == pytest.approx(2.747)
    assert c.gamma_r_plus == pytest.approx(0.640)   # βR>0
    assert c.gamma_r_minus == pytest.approx(0.395)  # βR<0
