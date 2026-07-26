"""목적 → 주요 치수 추정 (spec §2.2 Step 1, 2026-07-26 데이터 기반 개편).

오너 설계 반영:
- 카테고리별 실선 통계(단동선만, data/stats.py)가 1순위 근거.
- 데이터 부족 카테고리는 개략값 fallback — 반드시 source="fallback"으로 표시
  (조용히 섞지 않음).
- 사용자가 loa를 직접 선택 가능 (크기 개입). 미지정 시 적재량 역산.
- 적재량-크기 상호 제약: 선택한 크기로 못 싣는 적재량이면 명시적 거절 +
  필요한 최소 길이 제시.

역산: W_est = payload / payload_fraction, ∇ = W_est/ρ,
      ∇ = Cb·L·B·T, B = L/r_LB, T = B/r_BT
      ⇒ L = (∇ · r_LB² · r_BT / Cb)^(1/3)
"""
from __future__ import annotations

from data.stats import DataBand, category_bands
from src.core.types import GoalSpec, MainDimensions

RHO_SEAWATER = 1025.0
DEPTH_OVER_DRAFT = 1.6  # 형심/설계흘수 (건현 여유)

# Cb는 제조사 비공개라 데이터에서 못 뽑음 — 카테고리별 상수 가정 (spec §2.2)
CB_BY_CATEGORY = {"survey": 0.50, "patrol": 0.45, "workboat": 0.55}


class UnknownPurposeError(ValueError):
    pass


class PayloadInfeasibleError(ValueError):
    """선택한 크기로 요청 적재량을 실을 수 없음."""


# 개략값 fallback (실선 데이터 확보 전 카테고리용) — 출처: 통상 범위 개략
_FALLBACK_BANDS: dict[str, DataBand] = {
    "survey": DataBand(lb=3.0, bt=4.0, cb=0.50, payload_fraction=0.35,
                       loa_range=(1.0, 8.0), n_samples=0, source="fallback"),
    "patrol": DataBand(lb=3.5, bt=4.5, cb=0.45, payload_fraction=0.25,
                       loa_range=(1.0, 8.0), n_samples=0, source="fallback"),
    "workboat": DataBand(lb=2.8, bt=3.5, cb=0.55, payload_fraction=0.45,
                         loa_range=(1.0, 8.0), n_samples=0, source="fallback"),
}


def _resolve_bands() -> dict[str, DataBand]:
    """데이터 밴드 우선, 없으면 fallback. 카테고리 추가 = CSV 행 + CB 상수 한 줄."""
    resolved = dict(_FALLBACK_BANDS)
    try:
        resolved.update(category_bands(cb_by_category=CB_BY_CATEGORY))
    except FileNotFoundError:
        pass  # 데이터 파일 없는 환경(초기 클론)에서도 동작
    return resolved


PURPOSE_BANDS: dict[str, DataBand] = _resolve_bands()


def payload_capacity(loa: float, band: DataBand,
                     rho: float = RHO_SEAWATER) -> float:
    """이 길이·비율에서 실을 수 있는 최대 적재량 [kg]."""
    beam = loa / band.lb
    draft = beam / band.bt
    displacement = rho * band.cb * loa * beam * draft
    return band.payload_fraction * displacement


def estimate_dimensions(goal: GoalSpec, loa: float | None = None,
                        rho: float = RHO_SEAWATER) -> MainDimensions:
    """치수 추정. loa 지정 시 사용자 크기 우선 + 적재량 타당성 검사."""
    band = PURPOSE_BANDS.get(goal.purpose)
    if band is None:
        raise UnknownPurposeError(
            f"'{goal.purpose}'는 지원 용도가 아닙니다. "
            f"지원 용도: {sorted(PURPOSE_BANDS)}"
        )

    if loa is None:
        total_mass = goal.payload_kg / band.payload_fraction
        volume = total_mass / rho
        loa = (volume * band.lb ** 2 * band.bt / band.cb) ** (1.0 / 3.0)
    else:
        capacity = payload_capacity(loa, band, rho)
        if goal.payload_kg > capacity:
            # 필요한 최소 길이 역산 (적재량-크기 상호 제약)
            total_mass = goal.payload_kg / band.payload_fraction
            volume = total_mass / rho
            loa_min = (volume * band.lb ** 2 * band.bt / band.cb) ** (1.0 / 3.0)
            raise PayloadInfeasibleError(
                f"L={loa:.2f} m ({goal.purpose})의 적재 한계는 "
                f"{capacity:.0f} kg — 요청 {goal.payload_kg:.0f} kg을 실으려면 "
                f"최소 L={loa_min:.2f} m가 필요합니다."
            )

    beam = loa / band.lb
    draft = beam / band.bt
    return MainDimensions(
        loa=loa, beam=beam,
        depth=DEPTH_OVER_DRAFT * draft,
        draft_design=draft, cb=band.cb,
    )


def band_report(purpose: str) -> dict:
    """리포트용 근거 요약: 데이터 기반인지, 몇 척 근거인지, 크기 범위."""
    band = PURPOSE_BANDS[purpose]
    return {
        "source": band.source,
        "n_samples": band.n_samples,
        "loa_range": list(band.loa_range),
        "lb": band.lb,
        "bt": band.bt,
        "cb_assumed": band.cb,
        "payload_fraction": band.payload_fraction,
    }
