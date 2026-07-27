"""Ship-D 실데이터 로더 + 45파라미터 형상 재구성 (spec §7 M5a).

Ship-D (Bagazinski & Ahmed, MIT 2023): 합성 파라메트릭 선형 30,000척.
github.com/noahbagz/ShipD — 라이선스 파일 없음 → 그들 코드·데이터는
data/shipd/ (git 제외)에만 두고 우리 저장소로 재배포하지 않는다.
사용 결정 (2026-07-26): 형상만 전이, 성능 라벨은 우리 물리로 재계산.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

SHIPD_DIR = Path(__file__).parent / "shipd"


def available() -> bool:
    """Ship-D 로컬 사본 존재 여부 (fresh clone에는 없음 — 테스트 skip 기준)."""
    return (SHIPD_DIR / "InputVectors_30k.npy").exists() \
        and (SHIPD_DIR / "HullParameterization.py").exists()


def load_vectors() -> tuple[np.ndarray, list[str]]:
    """설계 벡터 (30000, 45)와 파라미터 실명 45개."""
    if not available():
        raise FileNotFoundError(
            "Ship-D 미설치 — git clone https://github.com/noahbagz/ShipD "
            "data/shipd (오너 승인 2026-07-27)"
        )
    vectors = np.load(SHIPD_DIR / "InputVectors_30k.npy")
    labels = [str(s) for s in
              np.load(SHIPD_DIR / "X_LABELS.npy", allow_pickle=True)]
    return vectors, labels


def _hull_parameterization():
    """Ship-D의 형상 엔진 import (경로 주입 — 코드 복사 금지)."""
    if str(SHIPD_DIR) not in sys.path:
        sys.path.insert(0, str(SHIPD_DIR))
    from HullParameterization import Hull_Parameterization
    return Hull_Parameterization


def reconstruct_mesh(vector: np.ndarray, num_wl: int = 40,
                     points_per_wl: int = 200) -> trimesh.Trimesh:
    """45파라미터 벡터 → trimesh 메쉬 (Ship-D 원저자 코드로 생성)."""
    HP = _hull_parameterization()
    hull = HP(np.asarray(vector, dtype=np.float64))
    with tempfile.TemporaryDirectory() as tmp:
        namepath = str(Path(tmp) / "hull")
        hull.gen_stl(NUM_WL=num_wl, PointsPerWL=points_per_wl,
                     namepath=namepath)
        return trimesh.load(namepath + ".stl")
