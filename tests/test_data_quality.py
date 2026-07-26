import pandas as pd
import pytest

from data.quality import (
    GRADE_A_MIN,
    GRADE_B_MIN,
    assign_grade,
    physics_flags,
    quality_score,
    validate_dataset,
)


def full_row(**overrides) -> dict:
    """모든 선택 필드가 채워진 A급 출처 행 (기준 케이스)."""
    row = {
        "name": "TestBoat", "maker": "TestCo", "category": "survey",
        "loa_m": 2.0, "beam_m": 1.0, "draft_m": 0.3,
        "weight_light_kg": 60.0, "weight_full_kg": 90.0, "payload_kg": 20.0,
        "speed_max_ms": 2.5, "speed_cruise_ms": 1.5,
        "source_url": "https://maker.example/spec.pdf", "source_grade": "A",
        "cross_verified": True, "notes": "제조사 스펙시트",
    }
    row.update(overrides)
    return row


# ---------- 점수 루브릭 ----------

def test_full_a_row_scores_100():
    score, breakdown = quality_score(full_row())
    assert score == 100
    assert breakdown["source"] == 40
    assert breakdown["cross"] == 15
    assert breakdown["completeness"] == 45


def test_source_grades():
    assert quality_score(full_row(source_grade="B"))[0] == 85
    assert quality_score(full_row(source_grade="C"))[0] == 70


def test_missing_optional_fields_reduce_score():
    row = full_row(draft_m=None, payload_kg=None, speed_cruise_ms=None,
                   weight_light_kg=None, notes=None)
    score, breakdown = quality_score(row)
    assert breakdown["completeness"] == 0
    assert score == 55  # 40 + 15 + 0


def test_no_cross_verification():
    assert quality_score(full_row(cross_verified=False))[0] == 85


# ---------- 물리 관문 ----------

def test_physics_clean_row_no_flags():
    assert physics_flags(full_row()) == []


def test_flag_impossible_weight():
    """치수 대비 물리적으로 뜰 수 없는 무게 → 플래그."""
    # 2.0×1.0×0.3 m, Cb 상한 0.9 → 최대 배수량 ≈ 553 kg
    flags = physics_flags(full_row(weight_full_kg=800.0))
    assert any("무게" in f for f in flags)


def test_flag_absurd_speed():
    """Fn > 3 주장 → 단위 오류 의심 (소형 활주정 물리 상한 밖) → 플래그."""
    # L=2.0: Fn 3.0 → v ≈ 13.3 m/s
    flags = physics_flags(full_row(speed_max_ms=15.0))
    assert any("속도" in f for f in flags)


def test_fast_planing_boat_not_flagged():
    """실재하는 활주 소형정 속도(Fn 1~3)는 오류 아님 — 통과해야 함."""
    # SL20 실례: L=1.05, 8kn=4.12 m/s → Fn 1.28
    flags = physics_flags(full_row(loa_m=1.05, beam_m=0.55, draft_m=0.15,
                                   weight_full_kg=27.0, speed_max_ms=4.12))
    assert not any("속도" in f for f in flags)


def test_flag_absurd_ratio():
    flags = physics_flags(full_row(beam_m=0.15))  # L/B ≈ 13
    assert any("비율" in f for f in flags)


def test_missing_draft_skips_weight_check():
    """흘수 결측이면 무게 검사 불가 — 플래그 없이 건너뜀 (결측≠오류)."""
    flags = physics_flags(full_row(draft_m=None, weight_full_kg=800.0))
    assert not any("무게" in f for f in flags)


# ---------- 등급 ----------

def test_grade_boundaries():
    assert assign_grade(GRADE_A_MIN, []) == "A"
    assert assign_grade(GRADE_A_MIN - 1, []) == "B"
    assert assign_grade(GRADE_B_MIN, []) == "B"
    assert assign_grade(GRADE_B_MIN - 1, []) == "C"


def test_physics_flag_forces_quarantine():
    """물리 관문: 점수 만점이어도 플래그 있으면 격리."""
    assert assign_grade(100, ["무게 오류"]) == "QUARANTINE"


# ---------- 데이터셋 일괄 검증 ----------

def test_validate_dataset_adds_columns():
    # 행2: C출처 + 교차검증 없음 + 완전정보 = 10+0+45 = 55점 → B
    # 행3: C출처 + 교차검증 없음 + 선택필드 전부 결측 = 10점 → C
    df = pd.DataFrame([
        full_row(),
        full_row(name="B2", source_grade="C", cross_verified=False),
        full_row(name="C3", source_grade="C", cross_verified=False,
                 draft_m=None, payload_kg=None, speed_cruise_ms=None,
                 weight_light_kg=None, notes=None),
    ])
    out = validate_dataset(df)
    assert list(out["grade"]) == ["A", "B", "C"]
    assert out.loc[0, "quality_score"] == 100
    assert "flags" in out.columns


def test_validate_dataset_requires_mandatory_fields():
    row = full_row()
    row["loa_m"] = None  # 필수 필드 결측 → 입장 불가
    with pytest.raises(ValueError, match="필수"):
        validate_dataset(pd.DataFrame([row]))
