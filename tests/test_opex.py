"""운항 경제 — 손계산 앵커 + 감속 성질 (경제성 2단계, 스펙 §2)."""
import pytest


def test_annual_fuel_hand_calc():
    """연료 손계산: 2877 kW × 178.8 g/kWh × 6000 h = 3,086 t/년."""
    from src.physics.economics.opex import annual_fuel
    r = annual_fuel(2877.0, 178.8, hours_per_year=6000.0)
    expected = 2877.0 * 178.8 * 6000.0 / 1e6   # g → t
    assert r["fuel_t_per_year"] == pytest.approx(expected, rel=1e-9)
    assert 3000.0 < expected < 3200.0


def test_fuel_opex_transport_unit_cost():
    """수송 단가 손계산 — 연간 비용 / (DWT × 연간 해리)."""
    from src.physics.economics.opex import fuel_opex
    r = fuel_opex(2877.0, 178.8, 7.0, 8000.0,
                  bunker_usd_per_t=600.0, hours_per_year=6000.0)
    dist_nm = 7.0 * (3600.0 / 1852.0) * 6000.0
    cost = 2877.0 * 178.8 * 6000.0 / 1e6 * 600.0
    assert r["distance_nm_per_year"] == pytest.approx(dist_nm, rel=1e-9)
    assert r["fuel_cost_usd_per_year"] == pytest.approx(cost, rel=1e-9)
    assert r["transport_usd_per_tnm"] == pytest.approx(
        cost / (8000.0 * dist_nm), rel=1e-9)


def test_slower_is_cheaper_per_tnm():
    """감속 → 톤·해리당 연료비 단조 감소 (P∝V³ ⇒ 단가∝V²)."""
    from src.physics.economics.opex import fuel_opex
    costs = []
    for v in (7.0, 6.0, 5.0):
        p = 2877.0 * (v / 7.0) ** 3
        costs.append(fuel_opex(p, 178.8, v, 8000.0)
                     ["transport_usd_per_tnm"])
    assert costs[0] > costs[1] > costs[2]
    # V² 성질: (6/7)² 비 재현
    assert costs[1] / costs[0] == pytest.approx((6.0 / 7.0) ** 2,
                                                rel=1e-6)


def test_electric_transport_hand_calc():
    """소형 전기 등가: 100 W·1.5 m/s(5.4 km/h)·짐 100 kg →
    18.52 Wh/km → 0.185 Wh/(kg·km)."""
    from src.physics.economics.opex import electric_transport
    r = electric_transport(100.0, 1.5, 100.0, elec_usd_per_kwh=0.15)
    wh_per_km = 100.0 / (1.5 * 3.6)
    assert r["wh_per_kg_km"] == pytest.approx(wh_per_km / 100.0,
                                              rel=1e-9)
    assert r["usd_per_kg_km"] == pytest.approx(
        wh_per_km / 100.0 * 0.15 / 1000.0, rel=1e-9)
