"""Lewis 단면 등각사상 (내항성 1단계, 스펙 2026-08-07 §2).

선체 단면을 2-파라미터 (H = B/2T, σ = A/BT) Lewis form으로 근사:
  z = M·(ζ + a1/ζ + a3/ζ³)  — 단위원 → 단면 곡선

계수 역산 (표준 풀이, Journée SEAWAY 계보):
  c1 = 3 + 4σ/π + (1 − 4σ/π)·((H−1)/(H+1))²
  a3 = (−c1 + 3 + √(9 − 2c1)) / c1
  a1 = (1 + a3)·(H − 1)/(H + 1)

자기 검증 성질 (시험 앵커): 반원 (H=1, σ=π/4) → a1 = a3 = 0,
무한주파수 heave 부가질량 = ρπR²/2 (고전 해석값).

유효 대역: Lewis form이 표현 가능한 (H, σ) 조합 — 범위 밖(극단
풍만·역곡률)은 정직 거절. 주파수 의존 계수(Tasai)는 다음 조각.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

RHO_SEAWATER = 1025.0


class LewisRangeError(ValueError):
    """Lewis 2-파라미터 사상이 표현 못 하는 단면 (정직 거절)."""


@dataclass(frozen=True)
class LewisSection:
    beam: float       # 수선 폭 B [m]
    draft: float      # 단면 흘수 T [m]
    sigma: float      # 단면적계수 A/(B·T)
    a1: float
    a3: float
    scale: float      # M [m]

    @property
    def h_ratio(self) -> float:
        return self.beam / (2.0 * self.draft)


def fit_lewis(beam: float, draft: float, sigma: float) -> LewisSection:
    """(B, T, σ) → Lewis 계수. 범위 밖은 LewisRangeError."""
    if beam <= 0 or draft <= 0:
        raise LewisRangeError("치수는 양수여야 합니다")
    if not 0.30 <= sigma <= 1.0:
        raise LewisRangeError(
            f"σ={sigma:.3f}는 Lewis 유효 대역(0.30~1.0) 밖")
    h = beam / (2.0 * draft)
    lam = (h - 1.0) / (h + 1.0)
    c1 = 3.0 + 4.0 * sigma / math.pi + (1.0 - 4.0 * sigma / math.pi) \
        * lam ** 2
    disc = 9.0 - 2.0 * c1
    if disc < 0.0:
        raise LewisRangeError(
            f"(H={h:.2f}, σ={sigma:.2f}) 조합은 Lewis 사상 범위 밖")
    a3 = (-c1 + 3.0 + math.sqrt(disc)) / c1
    a1 = (1.0 + a3) * lam
    # 척도: B/2 = M(1+a1+a3)
    m = (beam / 2.0) / (1.0 + a1 + a3)
    return LewisSection(beam=beam, draft=draft, sigma=sigma,
                        a1=a1, a3=a3, scale=m)


def section_points(sec: LewisSection, n: int = 40):
    """사상 곡선 (y, z) 점열 — 기하 재현 검증·시각화용.

    θ ∈ [0, π/2]: ζ = e^{iθ} 하반원 → 단면 (y≥0, 킬→수선)."""
    pts = []
    for i in range(n + 1):
        th = math.pi / 2.0 * i / n
        y = sec.scale * ((1.0 + sec.a1) * math.sin(th)
                         - sec.a3 * math.sin(3.0 * th))
        z = sec.scale * ((1.0 - sec.a1) * math.cos(th)
                         + sec.a3 * math.cos(3.0 * th))
        pts.append((y, sec.draft - z))   # z=0 킬 기준으로 변환
    return pts


def added_mass_heave_inf(sec: LewisSection,
                         rho: float = RHO_SEAWATER) -> float:
    """무한주파수 2D heave 부가질량 [kg/m] — Lewis form 해석식.

    a33(∞) = (ρπ/2)·M²·((1+a1)² + 3a3²)
    반원 검증: a1=a3=0, M=R → ρπR²/2 (고전값)."""
    return (rho * math.pi / 2.0) * sec.scale ** 2 \
        * ((1.0 + sec.a1) ** 2 + 3.0 * sec.a3 ** 2)
