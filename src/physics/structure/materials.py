"""구조 재료 물성 (구조 강도 2단계 + 알루 정본화 2026-08-12).

강: DNV 재료 계수 f1 (Pt.3 Ch.1 Sec.2 B203, p17 원전 대조:
NV-NS 1.00 · NV-32 1.28 · NV-36 1.39). 대형 상선 종강도는 IACS
UR S11 프레임 (허용 = 175·f1) — 강은 design_stress_nmm2 None.

알루미늄: **KS V ISO 12215-5:2019 표 B.2 정본** (오너 KS 무료
열람 캡처 판독 2026-08-12). 비열처리 알루(5xxx) σ_d = 0.7·σ_yw,
τ_d = 0.58·σ_d. 표값은 **용접** 기준 (HAZ 강도). 5083:
σ_uw 275 · σ_yw(용접 항복) 125 · **σ_d(설계응력) 88** · τ_d 51 ·
E 70,000. **정정 이력**: 이전 C급은 용접 항복 125를 허용응력처럼
써 1.4배 과대(과소설계) — σ_d 88로 정정 (yield 125는 좌굴 σ_F
전용 유지). σ_d 있는 재료는 게이트가 그것을 허용응력으로 사용.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    yield_nmm2: float                    # 항복 (알루 = 용접 σ_yw)
    f1: float                            # DNV 재료 계수 (연강 1.0)
    density_kgm3: float
    e_nmm2: float                        # 탄성계수
    grade: str                           # 출처 등급 A/B/C
    note: str
    design_stress_nmm2: float | None = None  # ISO σ_d (없으면 IACS)
    tau_d_nmm2: float | None = None      # ISO 설계 전단 τ_d
    sigma_uw_nmm2: float | None = None   # 용접 극한 σ_uw


MATERIALS: dict[str, Material] = {
    "mild_steel": Material(
        "mild_steel", 235.0, 1.0, 7850.0, 2.06e5, "A",
        "DNV NV-NS 연강 (Pt.3 Ch.1 Sec.2 B203) — UR S11 k=1 정합"),
    "ah36": Material(
        "ah36", 355.0, 1.39, 7850.0, 2.06e5, "A",
        "DNV NV-36 고장력강 — f1=1.39 (p17 원전 대조)"),
    "al5083": Material(
        "al5083", 125.0, 125.0 / 235.0, 2660.0, 7.0e4, "B",
        "알루 5083 용접부 — KS V ISO 12215-5 표 B.2 정본 "
        "(σ_yw 125·σ_d 88·τ_d 51·σ_uw 275). σ_d=0.7σ_yw 비열처리",
        design_stress_nmm2=88.0, tau_d_nmm2=51.0,
        sigma_uw_nmm2=275.0),
    "frp_eglass": Material(
        "frp_eglass", 100.0, 100.0 / 235.0, 1700.0, 0.15e5, "C",
        "E-glass 적층 설계 허용 대역 (극한 ~200/안전율 2) — "
        "ISO 12215-5 복합재 표 미확보 C급"),
}


def select_material(loa: float, purpose: str) -> Material:
    """크기·용도 → 기본 재료 (사용자 재지정 가능).

    실선 관행: 대형 상선 = 강, 소형 USV/작업정 = 알루미늄."""
    if loa >= 60.0:
        return MATERIALS["mild_steel"]
    return MATERIALS["al5083"]
