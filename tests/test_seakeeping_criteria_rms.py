"""내항 기준 RMS 승급 — NORDFORSK 원전 정의 (2026-08-10 회차).

원전: NORDFORSK 1987 (KTH Ocean Eng. 297 (2024) 116785 Table 4
재수록 — references/KTH_seakeeping_criteria.pdf): 기준은 **RMS**,
merchant roll 6°·fast small craft 4°. pitch는 NORDFORSK 무기재
(KTH 명시: Tasaki 계보 출처 불명) — 합불 제외·경고 강등."""
import pytest


def test_limits_are_rms_defined():
    """한계가 RMS 정의로 박제 — cargo 6·survey 4 (원전 표)."""
    from src.physics.seakeeping.criteria import SEAKEEPING_LIMITS
    assert SEAKEEPING_LIMITS["cargo"]["roll_rms_deg"] == pytest.approx(6.0)
    assert SEAKEEPING_LIMITS["survey"]["roll_rms_deg"] == pytest.approx(4.0)
    # pitch는 합불 키가 아니라 경고 키
    assert "pitch_rms_warn_deg" in SEAKEEPING_LIMITS["cargo"]


def test_gate_converts_significant_to_rms():
    """판정 환산 — 유의값/2 = RMS vs 한계 (정의 오독 2배 과엄 수리).

    유의 roll 10° = RMS 5° < cargo 한계 6 → 이제 합격 (구판은
    유의 10 vs 6 불합격 — 오독)."""
    from src.physics.seakeeping.criteria import judge_seakeeping
    rep = {"sig_roll_deg": 10.0, "sig_pitch_deg": 2.0,
           "sig_heave_m": 1.0}
    v = judge_seakeeping(rep, purpose="cargo", hs=3.5)
    assert v["checks"]["roll"] is True
    assert v["passed"] is True


def test_pitch_is_warning_not_gate():
    """pitch 초과 = 경고 표기만 (합불 불변) — 계보 불명 정직."""
    from src.physics.seakeeping.criteria import judge_seakeeping
    rep = {"sig_roll_deg": 2.0, "sig_pitch_deg": 9.9,
           "sig_heave_m": 0.5}
    v = judge_seakeeping(rep, purpose="cargo", hs=3.5)
    assert v["passed"] is True
    assert v["pitch_warning"] is True
    assert "출처" in v["pitch_note"] or "계보" in v["pitch_note"]


def test_small_usv_still_fails():
    """회귀 앵커 — 2m 조사선 유의 roll 28.5° = RMS 14.3 vs 4 →
    불합격 유지 (완화가 기존 검출을 뒤집지 않음)."""
    from src.physics.seakeeping.criteria import judge_seakeeping
    rep = {"sig_roll_deg": 28.5, "sig_pitch_deg": 1.0,
           "sig_heave_m": 0.2}
    v = judge_seakeeping(rep, purpose="survey", hs=0.5)
    assert v["checks"]["roll"] is False
    assert v["passed"] is False
