"""재료 3종 물성 — f1 계수·용접부 함정 (스펙 2026-08-09 §2)."""
import pytest


def test_material_catalog_values():
    """물성 앵커: 연강 235/f1=1.0 (UR S11 k=1 정합), AH36 355,
    알루 5083 용접부 강도 < 모재 (함정 지점 — 용접하면 항복 저하)."""
    from src.physics.structure.materials import MATERIALS
    ms = MATERIALS["mild_steel"]
    assert ms.yield_nmm2 == pytest.approx(235.0)
    assert ms.f1 == pytest.approx(1.0)
    ah = MATERIALS["ah36"]
    assert ah.yield_nmm2 == pytest.approx(355.0)
    assert ah.f1 > 1.0                     # 고장력 = 얇게 허용
    al = MATERIALS["al5083"]
    assert al.yield_nmm2 < 200.0           # 용접부 기준 (모재 아님)
    assert "용접" in al.note
    frp = MATERIALS["frp_eglass"]
    assert frp.grade == "B"                # KS ISO 12215-5 표 C.9 정본
    assert frp.design_stress_plate_nmm2 == pytest.approx(69.75, abs=0.1)


def test_select_material_by_size():
    """대형 = 강, 소형 = 알루 (실선 관행)."""
    from src.physics.structure.materials import select_material
    assert select_material(100.0, "cargo").name == "mild_steel"
    assert select_material(3.0, "survey").name == "al5083"
