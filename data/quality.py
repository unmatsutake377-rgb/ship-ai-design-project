"""실선 데이터 품질 점수·등급·물리 관문 (오너 설계, 2026-07-26).

원칙:
- 점수(0~100) = 출처 품질(40) + 교차 검증(15) + 정보 완전성(45)
- 등급은 절대 커트라인 (상대평가 금지 — 데이터 추가 시 기존 등급 불변)
- 물리 검사는 점수와 무관한 관문: 걸리면 무조건 QUARANTINE
  (출처가 좋아도 물리적으로 불가능한 스펙은 오타/조건누락)
- 결측 ≠ 오류: 선택 필드 결측은 감점만, 물리 검사는 해당 항목 건너뜀
"""
from __future__ import annotations

import math

import pandas as pd

from src.core.regime import froude_length

RHO_SEAWATER = 1025.0

# ---------- 점수 루브릭 ----------
SOURCE_POINTS = {"A": 40, "B": 25, "C": 10}  # 제조사/논문·조달/대리점·기사
CROSS_POINTS = 15
OPTIONAL_FIELD_POINTS = {
    "draft_m": 10,
    "payload_kg": 10,
    "speed_cruise_ms": 10,
    "weight_light_kg": 10,  # 경하/만재 구분 제공 여부
    "notes": 5,
}
REQUIRED_FIELDS = ["name", "category", "loa_m", "beam_m",
                   "weight_full_kg", "speed_max_ms", "source_url",
                   "source_grade"]

# ---------- 등급 커트라인 (절대) ----------
GRADE_A_MIN = 75   # 통계 가중치 1.0
GRADE_B_MIN = 50   # 통계 가중치 0.5, 미만은 C(보관만)

# ---------- 물리 관문 상수 ----------
CB_MAX = 0.90            # 부양 검사용 방형계수 상한 (상자에 가까운 극한)
# 소형 활주정은 Fn 1~3이 물리적으로 실재 (SL20 8kn = Fn 1.28 등).
# 이 관문의 목적은 "빠른 배 배제"가 아니라 단위 오류(kn↔m/s, km/h) 검출 —
# Fn > 3은 소형정 물리 상한 밖이라 오기 의심. 체계 구분은 통계 단계에서
# cruise Fn으로 별도 필터링한다.
FN_MAX_PLAUSIBLE = 3.0
LB_RANGE = (1.5, 10.0)        # L/B 상식 범위 (단동)
LB_RANGE_CAT = (1.2, 10.0)    # 쌍동은 광폭이 실물 (BlueBoat 1.29,
                              # Otter 1.85 — 08-07 수집 실측 캘리브레이션)
BT_RANGE = (1.5, 12.0)   # B/T 상식 범위


def _present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def quality_score(row: dict) -> tuple[int, dict]:
    """행 하나의 품질 점수와 항목별 내역."""
    source = SOURCE_POINTS.get(str(row.get("source_grade", "")).upper(), 0)
    cross = CROSS_POINTS if bool(row.get("cross_verified")) else 0
    completeness = sum(points for field, points in OPTIONAL_FIELD_POINTS.items()
                       if _present(row.get(field)))
    total = source + cross + completeness
    return total, {"source": source, "cross": cross,
                   "completeness": completeness}


def physics_flags(row: dict) -> list[str]:
    """물리 자기일관성 검사. 빈 목록 = 통과. 결측 항목은 건너뜀."""
    flags: list[str] = []
    loa = row.get("loa_m")
    beam = row.get("beam_m")
    draft = row.get("draft_m")
    weight = row.get("weight_full_kg")
    speed = row.get("speed_max_ms")

    # ① 부양 가능성: 만재 중량 ≤ ρ·L·B·T·Cb_max
    if all(_present(v) for v in (loa, beam, draft, weight)):
        max_displacement = RHO_SEAWATER * loa * beam * draft * CB_MAX
        if weight > max_displacement:
            flags.append(
                f"무게 불가능: 만재 {weight:.0f} kg > 배수량 상한 "
                f"{max_displacement:.0f} kg (L·B·T·Cb={CB_MAX})"
            )

    # ② 속도 타당성: 공표 최대속도의 Froude 수
    if all(_present(v) for v in (loa, speed)):
        fn = froude_length(speed, loa)
        if fn > FN_MAX_PLAUSIBLE:
            flags.append(
                f"속도 의심: Fn={fn:.2f} > {FN_MAX_PLAUSIBLE} "
                "(활주형이거나 오기 — 재확인 필요)"
            )

    # ③ 비율 상식 범위
    if all(_present(v) for v in (loa, beam)):
        lb = loa / beam
        if not LB_RANGE[0] <= lb <= LB_RANGE[1]:
            lb_rng = (LB_RANGE_CAT if str(row.get("hull_type", ""))
                      == "catamaran" else LB_RANGE)
            if not lb_rng[0] <= lb <= lb_rng[1]:
                flags.append(
                    f"비율 이상: L/B={lb:.1f} (상식 범위 {lb_rng})")
    if all(_present(v) for v in (beam, draft)):
        bt = beam / draft
        if not BT_RANGE[0] <= bt <= BT_RANGE[1]:
            flags.append(f"비율 이상: B/T={bt:.1f} (상식 범위 {BT_RANGE})")

    return flags


def assign_grade(score: int, flags: list[str]) -> str:
    """물리 관문 먼저, 그다음 절대 커트라인."""
    if flags:
        return "QUARANTINE"
    if score >= GRADE_A_MIN:
        return "A"
    if score >= GRADE_B_MIN:
        return "B"
    return "C"


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """전체 데이터셋 검증: 필수 필드 확인 → 점수·플래그·등급 컬럼 추가."""
    for field in REQUIRED_FIELDS:
        if field not in df.columns:
            raise ValueError(f"필수 컬럼 누락: {field}")
        missing = df[field].isna() | df[field].astype(str).str.strip().eq("")
        if missing.any():
            bad = df.loc[missing, "name"].tolist() if "name" in df else list(
                df.index[missing])
            raise ValueError(f"필수 필드 '{field}' 결측: {bad}")

    out = df.copy()
    scores, grades, all_flags = [], [], []
    for _, row in out.iterrows():
        r = row.to_dict()
        score, _ = quality_score(r)
        flags = physics_flags(r)
        scores.append(score)
        grades.append(assign_grade(score, flags))
        all_flags.append("; ".join(flags))
    out["quality_score"] = scores
    out["grade"] = grades
    out["flags"] = all_flags
    return out
