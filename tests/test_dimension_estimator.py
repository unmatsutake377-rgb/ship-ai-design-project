import pytest

from src.ai.dimension_estimator import (
    PURPOSE_BANDS,
    PayloadInfeasibleError,
    UnknownPurposeError,
    band_report,
    estimate_dimensions,
    payload_capacity,
)
from src.ai.hull_generator import CB_ENVELOPE
from src.core.types import GoalSpec

RHO = 1025.0


def test_survey_dimensions_sane():
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=100.0, purpose="survey")
    dims = estimate_dimensions(goal)
    assert 1.5 < dims.loa < 6.0          # 소형 USV 범위
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
    """치수 추정기가 내는 Cb는 반드시 생성기 도달범위 안 (spec §2.1).

    2단계 갱신 (2026-08-06): 도달성은 용도별 Cm 기준 — cargo Cb 0.75는
    기본 Cm 0.78로는 범위 밖이지만 Cm 0.98(상선 단면)로 생성돼 유효."""
    from src.ai.hull_generator import CP_RANGE, cm_for_purpose

    for purpose, band in PURPOSE_BANDS.items():
        cp = band.cb / cm_for_purpose(purpose)
        assert CP_RANGE[0] <= cp <= CP_RANGE[1], purpose


def test_unknown_purpose_raises():
    with pytest.raises(UnknownPurposeError, match="지원 용도"):
        estimate_dimensions(GoalSpec(1.5, 100.0, "racing"))


# ---------- 데이터 기반 개편 (2026-07-26) ----------

def test_survey_band_is_data_driven():
    """survey는 실선 단동 3척 이상 근거 — 개략값이 아니어야 함."""
    band = PURPOSE_BANDS["survey"]
    assert band.source == "data"
    assert band.n_samples >= 3
    # 실선 통계: 단동 조사용 USV L/B는 2 근방 (개략값 3.0과 뚜렷이 다름)
    assert 1.7 < band.lb < 2.5


def test_patrol_band_now_data_driven():
    """#17 수집(07-31: L30 정정 + M75·Inspector 90 추가)으로 patrol
    단동 3척 확보 → 치수 밴드가 fallback에서 데이터 기반으로 전환."""
    band = PURPOSE_BANDS["patrol"]
    assert band.source == "data"
    assert band.n_samples >= 3


def test_workboat_band_falls_back_with_label():
    """데이터 부족 카테고리는 fallback — 반드시 표시가 남아야 함.
    (workboat은 단동 표본 부족 — 수집 진행형 #17)"""
    band = PURPOSE_BANDS["workboat"]
    assert band.source == "fallback"


def test_user_loa_override():
    goal = GoalSpec(1.5, 10.0, "survey")
    dims = estimate_dimensions(goal, loa=1.6)
    assert dims.loa == 1.6
    band = PURPOSE_BANDS["survey"]
    assert dims.beam == pytest.approx(1.6 / band.lb, rel=1e-9)


def test_payload_infeasible_raises_with_min_length():
    """작은 배에 큰 짐 → 명시적 거절 + 필요한 최소 길이 제시."""
    goal = GoalSpec(1.5, 500.0, "survey")
    with pytest.raises(PayloadInfeasibleError, match="최소 L="):
        estimate_dimensions(goal, loa=1.2)


def test_payload_capacity_consistent_with_estimator():
    """역산 폐합: 자동 산정된 loa의 적재 한계 = 요청 적재량."""
    goal = GoalSpec(1.5, 100.0, "survey")
    dims = estimate_dimensions(goal)
    band = PURPOSE_BANDS["survey"]
    assert payload_capacity(dims.loa, band) == pytest.approx(100.0, rel=1e-6)


def test_band_report_fields():
    rep = band_report("survey")
    for key in ("source", "n_samples", "loa_range", "lb", "bt",
                "cb_assumed", "payload_fraction"):
        assert key in rep
