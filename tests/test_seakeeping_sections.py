"""메쉬 → Lewis 단면 추출 검증 — Wigley 해석 성질 앵커."""
import pytest

from src.ai.hull_generator import generate_hull_mesh
from src.core.types import MainDimensions
from src.physics.seakeeping.sections import extract_stations, station_geometry


@pytest.fixture(scope="module")
def wigley():
    dims = MainDimensions(loa=3.0, beam=0.9, depth=0.5, draft_design=0.3,
                          cb=0.45)
    return generate_hull_mesh(dims, cm=0.85), dims


def test_midship_sigma_matches_cm(wigley):
    """중앙 단면 σ = Cm (해석 성질 m/(m+1)=Cm) — 실측 재현."""
    mesh, dims = wigley
    b, t, sigma = station_geometry(mesh, 0.0, dims.draft_design)
    assert b == pytest.approx(dims.beam, rel=0.01)
    assert t == pytest.approx(dims.draft_design, rel=0.02)
    assert sigma == pytest.approx(0.85, abs=0.02)


def test_stations_cover_hull_and_taper(wigley):
    """스테이션 목록: 다수 유효 + 폭이 중앙 최대·선수미로 갈수록 감소."""
    mesh, dims = wigley
    st = extract_stations(mesh, dims.draft_design, n_stations=21)
    assert len(st) >= 15
    beams = [s.beam for _, s in st]
    mid = max(beams)
    assert mid == pytest.approx(dims.beam, rel=0.02)
    assert beams[0] < 0.5 * mid and beams[-1] < 0.5 * mid


def test_added_mass_distribution_positive(wigley):
    """전 스테이션 무한주파수 부가질량 > 0 + 중앙이 최대."""
    from src.physics.seakeeping.lewis import added_mass_heave_inf

    mesh, dims = wigley
    st = extract_stations(mesh, dims.draft_design)
    ams = [added_mass_heave_inf(s) for _, s in st]
    assert all(a > 0 for a in ams)
    assert max(ams) == ams[len(ams) // 2] or \
        abs(ams.index(max(ams)) - len(ams) // 2) <= 2
