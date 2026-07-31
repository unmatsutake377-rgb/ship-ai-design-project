"""CFD 라벨 CSV 병합 — 능동 학습 밥그릇의 스키마 검증."""
from src.cfd.labels import append_label
from src.cfd.result_parser import CfdResult

R1 = CfdResult(20.0, 8.0, 12.0, True, 500)
R2 = CfdResult(22.0, 9.0, 13.0, True, 900)
EMP = {"rf": 11.0, "rw": 7.0, "total": 18.0}


def test_append_creates_schema(tmp_path):
    csv = tmp_path / "cfd_labels.csv"
    df = append_label(csv, "wigley_sp", 1.5, 0.25, R1, EMP)
    assert csv.exists()
    for col in ("case_name", "speed_ms", "draft_m", "cfd_total_n",
                "cfd_pressure_n", "cfd_viscous_n", "converged",
                "emp_rf_n", "emp_rw_n", "emp_total_n"):
        assert col in df.columns, col
    assert len(df) == 1


def test_rerun_updates_not_duplicates(tmp_path):
    csv = tmp_path / "cfd_labels.csv"
    append_label(csv, "wigley_sp", 1.5, 0.25, R1, EMP)
    df = append_label(csv, "wigley_sp", 1.5, 0.25, R2, EMP)
    assert len(df) == 1
    assert df.iloc[0]["cfd_total_n"] == 22.0


def test_different_cases_accumulate(tmp_path):
    csv = tmp_path / "cfd_labels.csv"
    append_label(csv, "wigley_sp", 1.5, 0.25, R1, EMP)
    df = append_label(csv, "wigley_if", 1.5, 0.25, R2, EMP)
    assert len(df) == 2


def test_extra_columns_stored(tmp_path):
    csv = tmp_path / "cfd_labels.csv"
    df = append_label(csv, "wigley_lb4_simple", 1.85, 0.47, R1, EMP,
                      extra={"loa_m": 3.0, "beam_m": 0.75})
    assert df.iloc[0]["loa_m"] == 3.0
    assert df.iloc[0]["beam_m"] == 0.75
