"""저항 추정 (spec §2.3, M3.5).

마찰: ITTC-57 상관선 + 형상계수 (1+k).
조파: Michell 박선이론 적분.
Holtrop-Mennen 금지 — 20~300 m 상선 회귀식, USV 스케일 무효 (spec §2.3).
"""
from __future__ import annotations

import math

import numpy as np
import trimesh

from src.physics.hydrostatics import immersed_mesh, waterplane_properties

RHO_SEAWATER = 1025.0
NU_SEAWATER = 1.19e-6   # 해수 15°C 동점성계수 [m²/s]
G = 9.81
FORM_FACTOR = 0.10      # 형상계수 k (세장 선형 개략, 점성압력저항 보정)


def reynolds(speed: float, length: float, nu: float = NU_SEAWATER) -> float:
    return speed * length / nu


def ittc_cf(re: float) -> float:
    """ITTC-57 모형선-실선 상관선: Cf = 0.075 / (log10(Re) - 2)²."""
    return 0.075 / (math.log10(re) - 2.0) ** 2


def wetted_surface(mesh: trimesh.Trimesh, draft: float) -> float:
    """침수 표면적. slice가 만든 수선면 캡(들)은 물에 닿지 않으므로 제외."""
    below = immersed_mesh(mesh, draft)
    aw, _ = waterplane_properties(mesh, draft)
    return float(below.area) - aw


def frictional_resistance(speed: float, length: float, wetted_area: float,
                          rho: float = RHO_SEAWATER,
                          form_factor: float = FORM_FACTOR) -> float:
    """Rf = ½·ρ·V²·S·Cf·(1+k)."""
    cf = ittc_cf(reynolds(speed, length))
    return 0.5 * rho * speed ** 2 * wetted_area * cf * (1.0 + form_factor)
