"""정수 종강도 하중 곡선 (구조 강도 1단계, 스펙 2026-08-09 §2).

배 = 보(beam). 중량 w(x)와 부력 b(x)의 길이 방향 어긋남이 전단력
V(x)·굽힘 모멘트 M(x)를 만든다.

부호 관례 (프로젝트 공통): q = w − b, V = ∫q dx, M = ∫V dx,
**M > 0 = 호깅** (IACS hog 양수 정합). 중앙 화물 몰림 → 새깅(음수).

중량 분포 = 성분별 균일 블록 (C급 개략 — 정밀 분포는 백로그):
구조·의장 = 전장 균일, 기관·연료 = 선미 10~30% 구간,
화물(payload) = 중앙 25~85% 구간.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RHO_SEAWATER = 1025.0
G_ACC = 9.81

# 성분별 (선미 기준 시작 분율, 끝 분율) — 상선 통상 배치 (C급)
_BLOCK_FRACS = {
    "structure": (0.0, 1.0),
    "outfit": (0.0, 1.0),
    "machinery": (0.10, 0.30),
    "fuel": (0.10, 0.30),
    "payload": (0.25, 0.85),
}

WeightBlock = tuple[float, float, float]      # (mass_kg, x0, x1)


def standard_weight_blocks(component_masses_kg: dict[str, float],
                           xmin: float, loa: float) -> list[WeightBlock]:
    """성분 질량 → 통상 배치 균일 블록 목록. 미등록 성분 = 전장 균일."""
    out = []
    for name, mass in component_masses_kg.items():
        if mass <= 0.0:
            continue
        f0, f1 = _BLOCK_FRACS.get(name, (0.0, 1.0))
        out.append((float(mass), xmin + f0 * loa, xmin + f1 * loa))
    return out


def weight_linear_density(xs: np.ndarray,
                          blocks: list[WeightBlock]) -> np.ndarray:
    """블록 합성 w(x) [N/m] — 격자 적분이 총중량과 정확히 폐합하게
    정규화 (격자-블록 경계 불일치 오차 제거)."""
    w = np.zeros_like(xs, dtype=float)
    for mass, x0, x1 in blocks:
        span = max(x1 - x0, 1e-9)
        w += np.where((xs >= x0 - 1e-12) & (xs <= x1 + 1e-12),
                      mass * G_ACC / span, 0.0)
    total = sum(m for m, _, _ in blocks) * G_ACC
    integ = float(np.trapezoid(w, xs))
    if integ > 0.0:
        w *= total / integ
    return w


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """누적 사다리꼴 적분 — V·M 조립 공용."""
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate([[0.0], np.cumsum(seg)])
