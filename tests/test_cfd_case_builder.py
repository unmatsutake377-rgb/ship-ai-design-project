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
