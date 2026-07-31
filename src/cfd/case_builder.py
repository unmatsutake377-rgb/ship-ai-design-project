"""OpenFOAM 케이스 생성기 (스펙 §2·§3) — 붕어빵 틀.

파이프라인 산출물(hull.stl + report.json)을 받아 템플릿의 {{구멍}}을
메꾸고 STL을 배치한다. 좌표 규약: 케이스 안에서는 수면이 z=0
(선체를 −draft 평행이동). 반쪽 도메인 (y≥0, y=0 대칭면).

단상(simple) 모드의 이중모형(double-body) 트릭: 흘수선 위를 잘라내고
수면 자리에 미끄럼 벽을 두면 파도 없는 세계 — 마찰·점성만 깨끗하게.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import trimesh

TEMPLATE_ROOT = Path(__file__).parent / "templates"
RHO_SEAWATER = 1025.0   # src/physics/hydrostatics.py와 동일
NU_SEAWATER = 1.19e-6   # src/pipeline.py 레이놀즈 계산과 동일

# 배경 격자 해상도 (정직하게 거친 격자 — 스펙 §5 한계 1)
N_CELLS = {"simple": (96, 32, 24), "inter": (96, 32, 36)}


def domain_box(loa: float, mode: str, grid_factor: float = 1.0) -> dict:
    """스펙 §3 도메인: 선수 앞 1L · 선미 뒤 3L · 옆 1.5L · 아래 1L.

    locationInMesh(snappy가 '여기가 유체다' 확인하는 점)는 하류
    구석 근처 — 도메인 안이면서 선체에서 확실히 먼 곳.

    grid_factor: 격자 수렴 연구용 배율 — 상자는 그대로, 각 방향 셀
    수만 ×factor (1.5면 칸 수 ~3.4배). 답이 격자에 안 변할 때까지
    쪼개보는 것이 수렴 연구."""
    zmax = 0.0 if mode == "simple" else 0.5 * loa
    nx, ny, nz = (round(n * grid_factor) for n in N_CELLS[mode])
    box = {
        "XMIN": -1.5 * loa, "XMAX": 3.5 * loa,
        "YMIN": 0.0, "YMAX": 1.5 * loa,
        "ZMIN": -1.0 * loa, "ZMAX": zmax,
        "NX": nx, "NY": ny, "NZ": nz,
    }
    box["LOC_X"] = box["XMAX"] - 0.1 * loa
    box["LOC_Y"] = box["YMAX"] - 0.1 * loa
    box["LOC_Z"] = box["ZMIN"] + 0.1 * loa
    # inter 2단 블록용 (침수 버그 수리): 물층 1L / 공기층 0.5L,
    # 셀 높이 같게 2:1 배분
    box["NZ_WATER"] = int(nz * 2 / 3)
    box["NZ_AIR"] = nz - box["NZ_WATER"]
    return box


def prepare_hull(mesh: trimesh.Trimesh, draft: float, mode: str
                 ) -> trimesh.Trimesh:
    """수면을 z=0으로: −draft 평행이동. simple이면 물속 부분만 절단+캡."""
    out = mesh.copy()
    out.apply_translation([0.0, 0.0, -draft])
    if mode == "simple":
        out = trimesh.intersections.slice_mesh_plane(
            out, plane_normal=[0, 0, -1], plane_origin=[0, 0, 0],
            cap=True)  # 법선이 남기는 쪽을 가리킴: -z쪽(물속) 유지
        out.merge_vertices()
    return out


def render_template(text: str, values: dict) -> str:
    for key, val in values.items():
        text = text.replace("{{" + key + "}}", str(val))
    leftover = re.findall(r"\{\{(\w+)\}\}", text)
    if leftover:
        raise ValueError(f"안 메꿔진 구멍: {leftover}")
    return text


def _case_values(report: dict, mode: str, grid_factor: float = 1.0) -> dict:
    """report.json → 템플릿 값 사전."""
    loa = report["dimensions"]["loa"]
    speed = report["goal"]["target_speed_ms"]
    values = {"SPEED": speed, "NU": NU_SEAWATER, "RHO": RHO_SEAWATER,
              "LOA": loa, **domain_box(loa, mode, grid_factor)}
    # 난류 초기값: 난류강도 I=5%, 길이척도 l=0.07L (관용 개략)
    k = 1.5 * (0.05 * speed) ** 2
    values["K_INIT"] = f"{k:.6g}"
    values["OMEGA_INIT"] = f"{np.sqrt(k) / (0.09 ** 0.25 * 0.07 * loa):.6g}"
    # interFoam 물리 시간: 배 길이 15척분 흘려보내기
    values["END_TIME"] = f"{15.0 * loa / speed:.1f}"
    return values


def build_case(report_dir: Path, out_dir: Path, mode: str,
               grid_factor: float = 1.0) -> Path:
    """산출물 폴더 → 완성된 OpenFOAM 케이스 폴더. 반환: 케이스 경로."""
    report_dir, out_dir = Path(report_dir), Path(out_dir)
    report = json.loads((report_dir / "report.json").read_text())
    values = _case_values(report, mode, grid_factor)

    case = out_dir
    if case.exists():
        shutil.rmtree(case)
    template_dir = TEMPLATE_ROOT / mode
    for src in sorted(template_dir.rglob("*")):
        if not src.is_file():
            continue
        dst = case / src.relative_to(template_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(render_template(src.read_text(), values))

    mesh = trimesh.load(report_dir / report["mesh_file"])
    hull = prepare_hull(mesh, report["hydrostatics"]["draft"], mode)
    tri_dir = case / "constant" / "triSurface"
    tri_dir.mkdir(parents=True, exist_ok=True)
    hull.export(tri_dir / "hull.stl")
    return case
