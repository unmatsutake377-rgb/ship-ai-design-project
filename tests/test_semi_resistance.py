"""반배수량 저항 테스트 (Phase C-1 Task 2)."""
import numpy as np
import pytest

from src.ai.hull_generator import (
    generate_hull_mesh,
    generate_transom_hull_mesh,
    submerged_transom_area,
)
from src.core.types import MainDimensions
from src.physics.resistance import (
    total_resistance_mesh,
    total_resistance_semi,
    transom_drag,
)

TRANSOM_DIMS = MainDimensions(loa=4.0, beam=1.3, depth=0.48,
                              draft_design=0.30, cb=0.45)
WIGLEY_DIMS = MainDimensions(loa=4.0, beam=1.3, depth=0.48,
                             draft_design=0.30, cb=0.45)
DRAFT = 0.24


@pytest.fixture(scope="module")
def transom_mesh():
    return generate_transom_hull_mesh(TRANSOM_DIMS)


@pytest.fixture(scope="module")
def wigley_mesh():
    return generate_hull_mesh(WIGLEY_DIMS)


def test_transom_drag_formula():
    assert transom_drag(2.0, 0.1) == pytest.approx(
        0.5 * 1025 * 4.0 * 0.1 * 0.10, rel=1e-9)


def test_resistance_curve_continuous_monotone_trend(transom_mesh):
    """Fn 0.2~0.9 저항곡선: 양수·연속(점프 없음)·전반 증가 추세."""
    a_t = submerged_transom_area(TRANSOM_DIMS, DRAFT)
    fns = np.arange(0.20, 0.95, 0.1)
    totals = []
    for fn in fns:
        v = fn * (9.81 * 4.0) ** 0.5
        r = total_resistance_semi(transom_mesh, 4.0, DRAFT, v, a_t)
        assert r.total > 0
        totals.append(r.total)
    totals = np.array(totals)
    assert totals[-1] > totals[0] * 3  # 고속에서 크게 증가
    # 연속성: 인접 점 비율이 폭주하지 않음
    ratio = totals[1:] / totals[:-1]
    assert (ratio < 6).all()


def test_regime_trend_wigley_vs_transom(transom_mesh, wigley_mesh):
    """물리 경향 검증 (07-30 실측 기반 정정):

    같은 Cb에서 트랜섬 계열은 부피를 선미로 몰아 선수가 날카로움 →
    조파 감소가 기저저항 페널티를 상회 (실측 비 0.95@Fn0.25 →
    0.52@Fn0.55 최저 → 회복). 검증하는 경향:
    ① 트랜섬 상대 이점이 저속→중속으로 커짐 (비 감소)
    ② 비가 상식 밴드 안 (한쪽이 압도적 오답이면 밴드 이탈)
    주의: 동적 부상 미모델 — 정량은 근사, 경향만 고정."""
    a_t = submerged_transom_area(TRANSOM_DIMS, DRAFT)

    def ratio_at(fn):
        v = fn * (9.81 * 4.0) ** 0.5
        r_t = total_resistance_semi(transom_mesh, 4.0, DRAFT, v, a_t).total
        r_w = total_resistance_mesh(wigley_mesh, 4.0, DRAFT, v).total
        return r_t / r_w

    low, mid = ratio_at(0.25), ratio_at(0.55)
    assert mid < low                 # 중속에서 트랜섬 이점 극대
    assert 0.3 < mid < low < 1.3    # 상식 밴드
