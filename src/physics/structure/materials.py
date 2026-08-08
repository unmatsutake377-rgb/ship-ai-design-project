"""구조 재료 3종 물성 (구조 강도 2단계, 스펙 §2).

f1 = DNV 재료 계수 (Pt.3 Ch.1 Sec.2 B203, p17 원전 대조:
NV-NS 1.00 · NV-27 1.08 · NV-32 1.28 · NV-36 1.39 · NV-40 1.47).
고장력강은 허용 응력 상향 → 판 얇게. 알루미늄·FRP는 원전 규칙
미확보 — 등가 f1 = σy/235 환산 C급 (스펙 §3 보강 2 정직 표기).

함정 지점: 알루 5083은 **용접부(HAZ) 강도**가 설계 기준 — 모재
항복(215~228)이 아니라 용접 후 ~125 N/mm² (5083-H116 용접부 통상
대역). 강은 용접해도 모재 강도 유지 — 알루만의 함정.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    yield_nmm2: float      # 설계 기준 항복 (알루 = 용접부)
    f1: float              # DNV 재료 계수 (연강 1.0)
    density_kgm3: float
    e_nmm2: float          # 탄성계수
    grade: str             # 출처 등급 A/B/C
    note: str


MATERIALS: dict[str, Material] = {
    "mild_steel": Material(
        "mild_steel", 235.0, 1.0, 7850.0, 2.06e5, "A",
        "DNV NV-NS 연강 (Pt.3 Ch.1 Sec.2 B203) — UR S11 k=1 정합"),
    "ah36": Material(
        "ah36", 355.0, 1.39, 7850.0, 2.06e5, "A",
        "DNV NV-36 고장력강 — f1=1.39 (p17 원전 대조)"),
    "al5083": Material(
        "al5083", 125.0, 125.0 / 235.0, 2660.0, 0.69e5, "C",
        "알루 5083-H116 용접부(HAZ) 기준 — HSLC 원전 미확보, "
        "등가 f1 환산 C급"),
    "frp_eglass": Material(
        "frp_eglass", 100.0, 100.0 / 235.0, 1700.0, 0.15e5, "C",
        "E-glass 적층 설계 허용 대역 (극한 ~200/안전율 2) — "
        "ISO 12215-5 미확보 C급"),
}


def select_material(loa: float, purpose: str) -> Material:
    """크기·용도 → 기본 재료 (사용자 재지정 가능).

    실선 관행: 대형 상선 = 강, 소형 USV/작업정 = 알루미늄."""
    if loa >= 60.0:
        return MATERIALS["mild_steel"]
    return MATERIALS["al5083"]
