"""CFD 힘 로그 파서 검증 — 답을 아는 가짜 force.dat로."""
from pathlib import Path

import pytest

from src.cfd.result_parser import CfdResult, parse_forces

HEADER = "# Time  (total_x total_y total_z)  (pressure...)  (viscous...)\n"


def _write_force_dat(case_dir: Path, rows: list[str]) -> None:
    d = case_dir / "postProcessing" / "forces" / "0"
    d.mkdir(parents=True)
    (d / "force.dat").write_text(HEADER + "".join(rows))


def _row(t, total_x, press_x, visc_x):
    return f"{t}\t({total_x} 0 0)\t({press_x} 0 0)\t({visc_x} 0 0)\n"


def test_parse_converged_mean(tmp_path):
    """마지막 20% 창 평균 = 손계산. 반쪽 도메인이라 ×2."""
    # 100~1000까지 수렴하다가 마지막 창(801~1000)은 정확히 10.0 고정
    rows = [_row(t, 10.0 + (50.0 / t if t <= 800 else 0.0), 6.0, 4.0)
            for t in range(100, 1001, 100)]
    _write_force_dat(tmp_path, rows)
    r = parse_forces(tmp_path)
    assert isinstance(r, CfdResult)
    assert r.converged is True
    assert r.drag_total_n == pytest.approx(2 * 10.0)   # 창(마지막 2표본) 평균 ×2
    assert r.drag_pressure_n == pytest.approx(2 * 6.0)
    assert r.drag_viscous_n == pytest.approx(2 * 4.0)


def test_parse_diverged_flags_false(tmp_path):
    """발산(진동 큰) 로그 → converged=False, 값은 그래도 반환."""
    rows = [_row(t, 10.0 * (-1) ** t, 5.0, 5.0) for t in range(1, 21)]
    _write_force_dat(tmp_path, rows)
    r = parse_forces(tmp_path)
    assert r.converged is False


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_forces(tmp_path)
