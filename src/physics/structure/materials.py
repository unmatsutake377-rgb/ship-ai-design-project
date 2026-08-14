"""구조 재료 물성 (구조 강도 2단계 + ISO 표 17 정본화 2026-08-12).

강: DNV 재료 계수 f1 (Pt.3 Ch.1 Sec.2 B203, p17 원전 대조).
대형 상선 종강도는 IACS UR S11 프레임 (허용 = 175·f1) — ISO
12215-5는 24 m 미만 소형 표준이라 적용 밖 (girder_frame="IACS").

비강재(알루·FRP): **KS V ISO 12215-5:2019 표 17 정본**
(오너 KS 무료 열람 캡처 판독 2026-08-12) — 판재와 보강재 설계
응력이 **다름**:
- 알루: 판재 σ_d = min(0.6·σ_uw, 0.9·σ_yw) / 보강재 0.7·σ_yw
- 강:   판재 σ_d = min(0.6·σ_u, 0.9·σ_y)  / 보강재 0.8·σ_y
- FRP:  판재 0.5·σ_uf·k_BB·k_AM / 보강재 0.5·σ_ut·k_BB·k_AM
  (k_BB=1 표준 건조품질·k_AM=0.9 간소화법 — 고정 가정, 상세법
  이나 저품질 건조는 미지원 C급)
FRP 물성은 표 C.9 (E-glass/폴리에스터 손적층 CSM, ψ=0.30).

**배선 현황 (정직)**: 종강도 허용 = 판재 σ_d만 배선. 보강재 σ_d·
τ_d는 필드로 보유하되 스캔틀링(DNV 계보 0.63·l²·s·p/f1)이 아직
쓰지 않음 — ISO 보강재식 교체는 소형 구조 캠페인 백로그.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    yield_nmm2: float                    # 항복 (알루=용접 σ_yw)
    f1: float                            # DNV 재료 계수 (연강 1.0)
    density_kgm3: float
    e_nmm2: float                        # 탄성계수
    grade: str                           # 출처 등급 A/B/C
    note: str
    design_stress_nmm2: float | None = None   # σ_d 보강재 (표 17)
    tau_d_nmm2: float | None = None           # τ_d (표 17)
    sigma_uw_nmm2: float | None = None        # 용접/인장 극한
    design_stress_plate_nmm2: float | None = None  # σ_d 판재 (표17)
    girder_frame: str = "ISO"            # 종강도 프레임 IACS|ISO
    sigma_compression_nmm2: float | None = None  # 좌굴 σ_F용 압축
    corrosion_tk_mm: float = 1.5         # 부식 여유 (FRP 0)


MATERIALS: dict[str, Material] = {
    "mild_steel": Material(
        "mild_steel", 235.0, 1.0, 7850.0, 2.06e5, "A",
        "DNV NV-NS 연강 (Pt.3 Ch.1 Sec.2 B203) — UR S11 k=1. "
        "ISO 표 17 값 병기(판재 211.5·보강재 188, σ_u 400 표 B.2) "
        "이나 대형 상선 종강도는 IACS 175·f1 유지 (ISO 적용 밖)",
        design_stress_nmm2=188.0, tau_d_nmm2=109.0,
        design_stress_plate_nmm2=211.5, girder_frame="IACS"),
    "ah36": Material(
        "ah36", 355.0, 1.39, 7850.0, 2.06e5, "A",
        "DNV NV-36 고장력강 — f1=1.39 (p17 원전 대조). ISO 표 17 "
        "병기(판재 294·보강재 284) — 종강도는 IACS 프레임",
        design_stress_nmm2=284.0, tau_d_nmm2=165.0,
        design_stress_plate_nmm2=294.0, girder_frame="IACS"),
    "al5083": Material(
        "al5083", 125.0, 125.0 / 235.0, 2660.0, 7.0e4, "B",
        "알루 5083 용접부 — KS V ISO 12215-5 표 B.2·17 정본. "
        "σ_yw 125·σ_uw 275 → 판재 σ_d=min(0.6·275, 0.9·125)=112.5·"
        "보강재 0.7·125=87.5·τ_d=0.58·87.5=50.75. f1은 DNV 정의역"
        "(강) 밖 외삽 C급 — 보수측(두께↑)이라 유지",
        design_stress_nmm2=87.5, tau_d_nmm2=50.75,
        sigma_uw_nmm2=275.0, design_stress_plate_nmm2=112.5,
        sigma_compression_nmm2=125.0),
    "frp_eglass": Material(
        "frp_eglass", 155.0, 155.0 / 235.0, 1430.0, 8267.0, "B",
        "E-glass/폴리에스터 손적층 CSM (ψ=0.30) — KS V ISO "
        "12215-5 표 C.9 정본: E 8267·G 2943·σ_ut 112·σ_uc 141·"
        "σ_uf 155·τ_u 59. 표 17(k_BB 1·k_AM 0.9): 판재 0.5·155·0.9"
        "=69.75·보강재 0.5·112·0.9=50.4·τ_d 0.5·59·0.9=26.55. "
        "밀도 1430=혼합칙 ψ0.30 (유리 2560·수지 1200). 좌굴 σ_F는 "
        "σ_uc 141 (굽힘 σ_uf 아님 — 압축 좌굴 물리). 부식 여유 0. "
        "f1은 DNV 정의역 밖 마법비 C급 (스캔틀링 경로 — 캠페인 이월)",
        design_stress_nmm2=50.4, tau_d_nmm2=26.55,
        design_stress_plate_nmm2=69.75,
        sigma_compression_nmm2=141.0, corrosion_tk_mm=0.0),
}


def select_material(loa: float, purpose: str) -> Material:
    """크기·용도 → 기본 재료 (사용자 재지정 가능).

    실선 관행: 대형 상선 = 강, 소형 USV/작업정 = 알루미늄."""
    if loa >= 60.0:
        return MATERIALS["mild_steel"]
    return MATERIALS["al5083"]
