"""케이스 생성기 코어 — 기하 준비와 치환 엔진 (OpenFOAM 불필요)."""
import pytest
import trimesh

from src.cfd.case_builder import domain_box, prepare_hull, render_template


def test_domain_box_proportional_to_loa():
    """스펙 §3: 앞1L·뒤3L·옆1.5L·아래1L. 선체 x∈[-L/2,+L/2]."""
    L = 4.0
    b = domain_box(L, mode="simple")
    assert b["XMIN"] == pytest.approx(-0.5 * L - 1.0 * L)
    assert b["XMAX"] == pytest.approx(+0.5 * L + 3.0 * L)
    assert b["YMIN"] == 0.0                       # 대칭면
    assert b["YMAX"] == pytest.approx(1.5 * L)
    assert b["ZMIN"] == pytest.approx(-1.0 * L)
    assert b["ZMAX"] == 0.0                       # 단상: 수면이 천장


def test_domain_box_inter_has_air():
    b = domain_box(4.0, mode="inter")
    assert b["ZMAX"] == pytest.approx(0.5 * 4.0)  # 자유수면: 공기층


def test_fs_band_refinement_in_inter_case(tmp_path):
    """수면 띠 국소 refine (격자 3라운드): fs_level>0이면 snappy에
    수면 상자(fsband) 세분 지시가 들어가야 함."""
    from src.cfd.case_builder import build_case
    case = build_case(_fake_report_dir(tmp_path / "rep"),
                      tmp_path / "case", mode="inter", fs_level=2)
    snappy = (case / "system/snappyHexMeshDict").read_text()
    assert "fsband" in snappy
    assert "(1e15 2)" in snappy          # 띠 안은 레벨 2 (칸 1/4 크기)
    # 띠는 수면(z=0) 대칭 ±0.05L = ±0.185
    assert "-0.185" in snappy and "0.185" in snappy


def test_fs_band_default_off(tmp_path):
    """기본값 fs_level=0 — 기존 케이스와 동일 (레벨 0 = 세분 없음)."""
    from src.cfd.case_builder import build_case
    case = build_case(_fake_report_dir(tmp_path / "rep"),
                      tmp_path / "case", mode="inter")
    assert "(1e15 0)" in (case / "system/snappyHexMeshDict").read_text()


def test_domain_box_grid_factor_scales_cells():
    """격자 수렴 연구용: 배율 1.5 → 각 방향 셀 수 1.5배 (칸 수 ~3.4배)."""
    base = domain_box(4.0, mode="inter")
    fine = domain_box(4.0, mode="inter", grid_factor=1.5)
    assert fine["NX"] == round(base["NX"] * 1.5)
    assert fine["NY"] == round(base["NY"] * 1.5)
    assert fine["NZ_WATER"] + fine["NZ_AIR"] == round(
        (base["NZ_WATER"] + base["NZ_AIR"]) * 1.5)
    # 상자 크기는 불변 — 칸만 잘게
    assert fine["XMIN"] == base["XMIN"] and fine["ZMAX"] == base["ZMAX"]


def test_location_in_mesh_inside_domain():
    b = domain_box(4.0, mode="simple")
    assert b["XMIN"] < b["LOC_X"] < b["XMAX"]
    assert b["YMIN"] < b["LOC_Y"] < b["YMAX"]
    assert b["ZMIN"] < b["LOC_Z"] < b["ZMAX"]


def test_prepare_hull_simple_cuts_at_waterline():
    """단상: 흘수선 위를 잘라내고 물속 부분만, 수면이 z=0."""
    box = trimesh.creation.box(bounds=[[-2, -0.5, 0.0], [2, 0.5, 0.6]])
    cut = prepare_hull(box, draft=0.4, mode="simple")
    assert cut.is_watertight                        # 캡 성공
    assert cut.bounds[1][2] <= 1e-9                 # 최고점 z ≤ 0
    assert cut.bounds[0][2] == pytest.approx(-0.4)  # 용골 z = -draft


def test_prepare_hull_inter_keeps_whole():
    box = trimesh.creation.box(bounds=[[-2, -0.5, 0.0], [2, 0.5, 0.6]])
    whole = prepare_hull(box, draft=0.4, mode="inter")
    assert whole.bounds[0][2] == pytest.approx(-0.4)
    assert whole.bounds[1][2] == pytest.approx(0.2)  # 갑판 = depth-draft


def test_render_template_fills_all():
    out = render_template("v ({{SPEED}} 0 0); n {{NX}};",
                          {"SPEED": 1.5, "NX": 96})
    assert out == "v (1.5 0 0); n 96;"


def test_render_template_leftover_raises():
    with pytest.raises(ValueError, match="OOPS"):
        render_template("{{OOPS}}", {"SPEED": 1.5})


def _fake_report_dir(d):
    """실제 파이프라인 없이 산출물 폴더 흉내 — 시험 고속화."""
    import json
    d.mkdir(parents=True)
    box = trimesh.creation.box(bounds=[[-1.85, -0.5, 0.0], [1.85, 0.5, 0.6]])
    box.export(d / "hull.stl")
    report = {"goal": {"target_speed_ms": 1.5},
              "dimensions": {"loa": 3.7},
              "hydrostatics": {"draft": 0.25},
              "mesh_file": "hull.stl"}
    (d / "report.json").write_text(json.dumps(report))
    return d


REQUIRED_SIMPLE = [
    "0/U", "0/p", "0/k", "0/omega", "0/nut",
    "constant/transportProperties", "constant/turbulenceProperties",
    "constant/triSurface/hull.stl",
    "system/blockMeshDict", "system/controlDict",
    "system/fvSchemes", "system/fvSolution", "system/snappyHexMeshDict",
]


def test_build_case_simple_complete(tmp_path):
    from src.cfd.case_builder import build_case
    case = build_case(_fake_report_dir(tmp_path / "rep"),
                      tmp_path / "case", mode="simple")
    for rel in REQUIRED_SIMPLE:
        assert (case / rel).exists(), rel
    # 안 메꿔진 구멍 0개 (render_template이 이미 막지만 이중 확인)
    for f in case.rglob("*"):
        if f.is_file() and f.suffix != ".stl":
            assert "{{" not in f.read_text(), f
    # 격자 상자가 L에 비례해 박혔는지
    bmd = (case / "system/blockMeshDict").read_text()
    assert "-5.55" in bmd and "12.95" in bmd  # -1.5*3.7, 3.5*3.7
    # 단상 STL이 정말 물속 부분만인지
    hull = trimesh.load(case / "constant/triSurface/hull.stl")
    assert hull.bounds[1][2] <= 1e-9


REQUIRED_INTER = [
    "0/U", "0/p_rgh", "0/k", "0/omega", "0/nut", "0/alpha.water",
    "constant/transportProperties", "constant/turbulenceProperties",
    "constant/g", "constant/triSurface/hull.stl",
    "system/blockMeshDict", "system/controlDict", "system/fvSchemes",
    "system/fvSolution", "system/snappyHexMeshDict", "system/setFieldsDict",
]


def test_build_case_inter_complete(tmp_path):
    from src.cfd.case_builder import build_case
    case = build_case(_fake_report_dir(tmp_path / "rep"),
                      tmp_path / "case", mode="inter")
    for rel in REQUIRED_INTER:
        assert (case / rel).exists(), rel
    # 자유수면: STL이 통짜 (갑판까지, z>0 존재)
    hull = trimesh.load(case / "constant/triSurface/hull.stl")
    assert hull.bounds[1][2] > 0
    # 물리 시간이 15L/V로 박혔는지 (15*3.7/1.5 = 37.0)
    assert "37.0" in (case / "system/controlDict").read_text()
    # 침수 버그 수리 (07-31): 입구가 물/공기로 분리돼야 함
    bmd = (case / "system/blockMeshDict").read_text()
    assert "inletWater" in bmd and "inletAir" in bmd
    assert "inletWater" in (case / "0/alpha.water").read_text()
