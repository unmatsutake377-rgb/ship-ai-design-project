"""능동 학습 표본 생성기 — Fn 고정·L/B 스팬 Wigley 3척."""
import json
import math

import pytest
import trimesh

from src.cfd.samples import FN_TARGET, SAMPLES, build_sample, speed_for


def test_samples_span_lb():
    lbs = [s["loa"] / s["beam"] for s in SAMPLES]
    assert lbs == pytest.approx([4.0, 7.0, 10.0], rel=0.01)
    assert all(s["loa"] == 3.0 for s in SAMPLES)


def test_speed_is_fixed_froude():
    v = speed_for(3.0)
    assert v / math.sqrt(9.81 * 3.0) == pytest.approx(FN_TARGET)


def test_build_sample_report_complete(tmp_path):
    d = build_sample(SAMPLES[0], tmp_path)
    report = json.loads((d / "report.json").read_text())
    assert report["goal"]["target_speed_ms"] == pytest.approx(speed_for(3.0))
    dims = report["dimensions"]
    assert dims["loa"] == 3.0 and dims["beam"] == 0.75
    # T = B/1.6 (표준 Wigley 비율)
    assert report["hydrostatics"]["draft"] == pytest.approx(0.75 / 1.6)
    r = report["resistance"]
    assert r["rw"] > 0 and r["rf"] > 0
    mesh = trimesh.load(d / report["mesh_file"])
    assert mesh.is_watertight
