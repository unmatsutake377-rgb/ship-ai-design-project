"""대형 추진 1차 — 소요 제동동력 + 엔진 선정 (3단계 골격).

경로: 저항 R·속도 V → 유효동력 PE → 준추진효율 ηD → 제동동력 PB
→ 엔진 카탈로그 선정 (여유율).

ηD 대역 (B급, Holtrop 1982 계보·상선 통상): 0.60~0.70 — 1차는
보수 하한 0.60. **Wageningen B-시리즈 정밀 설계(직경·피치·회전수·
캐비테이션)는 후속** — Kt·Kq 다항 계수는 공개 문헌 검증 후 도입
(기억 계수 하드코딩 금지 — 환각 계수가 최악, 수집 관례).

엔진 카탈로그: data/engine_catalog.csv — 실물 공개 스펙만 (출처
URL·등급). 초기 표본은 C급(미검증 개략) — 오너 수집·웹 검증으로
승급 전까지 리포트에 등급 표기.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

ETA_D_DEFAULT = 0.60   # 준추진효율 보수 하한 (B급 — B-시리즈로 대체 예정)
ENGINE_MARGIN = 1.15   # 해상 여유 (sea margin 15% 통상)

CATALOG = Path(__file__).resolve().parents[2] / "data/engine_catalog.csv"


class NoSuitableEngineError(ValueError):
    """카탈로그에 소요 동력을 채울 엔진 없음 — 정직 거절."""


@dataclass(frozen=True)
class EnginePick:
    name: str
    maker: str
    mcr_kw: float
    mass_t: float
    sfoc_g_per_kwh: float
    source_grade: str
    pb_required_kw: float
    load_fraction: float


def brake_power_kw(resistance_n: float, speed_ms: float,
                   eta_d: float = ETA_D_DEFAULT) -> float:
    """제동동력 PB = R·V/ηD [kW]."""
    return resistance_n * speed_ms / eta_d / 1000.0


def select_engine(pb_kw: float, margin: float = ENGINE_MARGIN,
                  catalog: Path = CATALOG) -> EnginePick:
    """PB×여유율을 채우는 최소 MCR 엔진 — 없으면 정직 거절."""
    need = pb_kw * margin
    rows = []
    with open(catalog, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    ok = [r for r in rows if float(r["mcr_kw"]) >= need]
    if not ok:
        best = max(float(r["mcr_kw"]) for r in rows) if rows else 0.0
        raise NoSuitableEngineError(
            f"소요 제동동력 {pb_kw:.0f} kW × 여유 {margin} = {need:.0f} kW를 "
            f"채울 엔진이 카탈로그에 없습니다 (최대 {best:.0f} kW).")
    r = min(ok, key=lambda r: float(r["mcr_kw"]))
    return EnginePick(name=r["name"], maker=r["maker"],
                      mcr_kw=float(r["mcr_kw"]), mass_t=float(r["mass_t"]),
                      sfoc_g_per_kwh=float(r["sfoc_g_per_kwh"]),
                      source_grade=r["source_grade"],
                      pb_required_kw=pb_kw,
                      load_fraction=need / float(r["mcr_kw"]))


def design_propulsion(resistance_n: float, speed_ms: float,
                      draft: float, z: int = 4, ear: float = 0.55,
                      cb: float = 0.70):
    """프로펠러 실설계 + 엔진 선정 통합 (3단계 2차 완성).

    ηD를 개략(0.60)이 아니라 B-시리즈 설계점에서 실산출 —
    직경은 흘수 제한 D = 0.70·T (상선 관례). 캐비테이션이 EAR 부족을
    말하면 Keller 최소값으로 한 번 재설계 (정직 반복).
    반환: (PropellerDesign, EnginePick)."""
    from src.physics.propeller import design_propeller

    d_max = 0.70 * draft
    prop = design_propeller(resistance_n, speed_ms, diameter_max=d_max,
                            z=z, ear=ear, cb=cb)
    if not prop.cavitation_ok:
        prop = design_propeller(resistance_n, speed_ms,
                                diameter_max=d_max, z=z,
                                ear=min(1.05, prop.ear_min_keller), cb=cb)
    engine = select_engine(prop.brake_power_kw)
    return prop, engine
