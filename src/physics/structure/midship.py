"""미드십 단면계수 조립 (구조 강도 2단계, 스펙 §2).

모델: 상자보 — 선저판(z=0)·갑판(z=D)·선측 2장(수직) + 종늑골
(면적 점부재, 판 근방 배치). 평행축 정리로 중립축·I → 단면계수
Z_deck = I/(D−zn), Z_keel = I/zn (중립축에서 먼 쪽이 먼저 아픔).

한계 정직 표기: 이중저·해치 개구·전단 지연 없음 — 개구 큰 배
(컨테이너선)는 과대평가. 상선 폐단면 근사.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MidshipSection:
    neutral_axis_m: float    # 킬 기준 중립축 높이
    inertia_m4: float
    z_deck_m3: float
    z_keel_m3: float
    area_m2: float


def assemble_midship(beam: float, depth: float,
                     t_bottom_mm: float, t_side_mm: float,
                     t_deck_mm: float,
                     n_bottom_long: int, n_deck_long: int,
                     long_area_cm2: float) -> MidshipSection:
    """판 3종 + 종늑골 → 중립축·I·단면계수.

    종늑골: 선저군 z=0.05D, 갑판군 z=0.95D 점면적 (부착판 근방)."""
    tb, ts, td = (t_bottom_mm * 1e-3, t_side_mm * 1e-3,
                  t_deck_mm * 1e-3)
    a_long = long_area_cm2 * 1e-4
    # (면적, z중심, 자체 I)
    elems = [
        (beam * tb, 0.0, beam * tb ** 3 / 12.0),          # 선저
        (beam * td, depth, beam * td ** 3 / 12.0),        # 갑판
        (2.0 * ts * depth, depth / 2.0,
         2.0 * ts * depth ** 3 / 12.0),                   # 선측 2장
        (n_bottom_long * a_long, 0.05 * depth, 0.0),      # 선저 종늑골
        (n_deck_long * a_long, 0.95 * depth, 0.0),        # 갑판 종늑골
    ]
    area = sum(a for a, _, _ in elems)
    zn = sum(a * z for a, z, _ in elems) / max(area, 1e-12)
    inertia = sum(i0 + a * (z - zn) ** 2 for a, z, i0 in elems)
    return MidshipSection(
        neutral_axis_m=zn, inertia_m4=inertia,
        z_deck_m3=inertia / max(depth - zn, 1e-9),
        z_keel_m3=inertia / max(zn, 1e-9),
        area_m2=area)
