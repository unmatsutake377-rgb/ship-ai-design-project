"""대형 나선 관통 (4단계) — 100 m 화물선 첫 설계."""
import pytest

from src.ai.dimension_estimator import estimate_dimensions
from src.ai.hull_generator import cm_for_purpose, generate_hull_mesh, lcb_for_purpose
from src.core.types import GoalSpec
from src.pipeline_large import design_spiral_large


@pytest.fixture(scope="module")
def cargo_design():
    goal = GoalSpec(target_speed_ms=7.0, payload_kg=5_000_000.0,
                    purpose="cargo", endurance_h=240.0)
    dims = estimate_dimensions(goal)
    mesh = generate_hull_mesh(dims, cm=cm_for_purpose("cargo"),
                              lcb_frac=lcb_for_purpose("cargo"))
    return design_spiral_large(mesh, dims, goal), dims


def test_cargo_100m_spiral_converges_and_passes(cargo_design):
    r, dims = cargo_design
    assert r["iterations"] < 30
    assert r["passed"], (r["hydro_checks"], r["freeboard_ok"])
    assert r["propeller"]["cavitation_ok"]


def test_cargo_100m_magnitudes(cargo_design):
    """자릿수 실선 정합: DWT 5000t급 100 m — 경하 천 톤대,
    저항 수백 kN 이하, 엔진 MW급, 프로펠러 rpm 수십~이백."""
    r, dims = cargo_design
    assert 1_000 < r["lightship_t"]["total"] < 6_000
    assert 50_000 < r["resistance"]["total"] < 500_000
    assert 1_000 <= r["engine"]["mcr_kw"] <= 25_000
    assert 30 < r["propeller"]["rpm"] < 250
    assert r["resistance"]["froude"] < 0.30       # 배수량 대역


def test_cargo_gm_uses_imo_band(cargo_design):
    """GM 밴드 하한 = IMO 0.15 m 환산 (소형 0.04 아님)."""
    r, dims = cargo_design
    assert r["gm_band"][0] == pytest.approx(0.15 / dims.beam)
