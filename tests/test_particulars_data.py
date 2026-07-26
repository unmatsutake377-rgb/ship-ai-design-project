"""수집된 실선 데이터 자체를 테스트로 고정 — 행 추가 시 자동 재검증.

규율: CSV에 행을 추가하면 이 테스트가 전 행을 다시 심사한다.
검증 실패 데이터는 병합 불가 (코드와 같은 규율).
"""
from pathlib import Path

import pandas as pd
import pytest

from data.quality import validate_dataset

CSV_PATH = Path(__file__).parent.parent / "data" / "small_craft_particulars.csv"


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    return validate_dataset(df)


def test_file_exists_and_nonempty(dataset):
    assert len(dataset) >= 5  # 1차 목표의 최소선


def test_no_quarantined_rows(dataset):
    """물리 관문 통과 실패 행은 병합 금지."""
    bad = dataset[dataset["grade"] == "QUARANTINE"]
    assert bad.empty, f"격리 행 존재: {bad[['name', 'flags']].to_dict('records')}"


def test_all_rows_grade_a_or_b(dataset):
    """통계에 못 쓰는 C급은 이 파일에 넣지 않는다 (별도 보관)."""
    assert set(dataset["grade"]) <= {"A", "B"}, dataset[["name", "grade"]]


def test_all_sources_recorded(dataset):
    assert dataset["source_url"].str.startswith("http").all()


def test_categories_are_known(dataset):
    """카테고리 자유 확장 가능하되, 오타 방지를 위해 등록부와 대조."""
    from src.ai.dimension_estimator import PURPOSE_BANDS

    unknown = set(dataset["category"]) - set(PURPOSE_BANDS)
    assert not unknown, f"등록되지 않은 카테고리: {unknown}"
