import pytest

from src.ai.dimension_estimator import (
    PURPOSE_BANDS,
    UnknownPurposeError,
    estimate_dimensions,
)
from src.ai.hull_generator import CB_ENVELOPE
from src.core.types import GoalSpec

RHO = 1025.0


def test_survey_dimensions_sane():
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=100.0, purpose="survey")
    dims = estimate_dimensions(goal)
    assert 2.0 < dims.loa < 6.0          # 소형 USV 범위
    assert dims.beam < dims.loa
    assert dims.draft_design < dims.beam
    assert dims.depth > dims.draft_design


def test_volume_closure():
    """Cb·L·B·T가 목표 배수용적과 일치해야 함 (역산 폐합)."""
    goal = GoalSpec(1.5, 100.0, "survey")
    band = PURPOSE_BANDS["survey"]
    dims = estimate_dimensions(goal)
    target_vol = (100.0 / band.payload_fraction) / RHO
    actual_vol = dims.cb * dims.loa * dims.beam * dims.draft_design
    assert actual_vol == pytest.approx(target_vol, rel=1e-6)


def test_ratios_respected():
    goal = GoalSpec(1.5, 150.0, "patrol")
    band = PURPOSE_BANDS["patrol"]
    dims = estimate_dimensions(goal)
    assert dims.loa / dims.beam == pytest.approx(band.lb, rel=1e-6)
    assert dims.beam / dims.draft_design == pytest.approx(band.bt, rel=1e-6)


def test_all_band_cbs_within_generator_envelope():
    """치수 추정기가 내는 Cb는 반드시 생성기 도달범위 안 (spec §2.1)."""
    for band in PURPOSE_BANDS.values():
        assert CB_ENVELOPE[0] <= band.cb <= CB_ENVELOPE[1]


def test_unknown_purpose_raises():
    with pytest.raises(UnknownPurposeError, match="지원 용도"):
        estimate_dimensions(GoalSpec(1.5, 100.0, "racing"))
