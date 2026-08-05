"""ShipGen 조건부 선별 (스펙 2단계) — Ship-D 로컬 없으면 skip."""
import pytest

from data import shipd_loader

pytestmark = pytest.mark.skipif(
    not shipd_loader.available(),
    reason="Ship-D 로컬 사본 없음",
)


def test_select_returns_gate_passing_hull():
    from src.ai.shipgen_select import select_hull
    from src.core.types import GoalSpec

    goal = GoalSpec(target_speed_ms=1.2, payload_kg=100.0,
                    purpose="survey", endurance_h=4.0)
    pick = select_hull(goal, 3.0, pool_size=60, seed=7)
    assert pick is not None
    r = pick.row
    assert r["feasible"] and r["space_ok"] and r["gm_alloc_ok"] \
        and r["trim_ok"]
    assert pick.vector.shape[0] == 45
    assert pick.n_passed >= 1


def test_condition_weights_speed_monotonic():
    """조건 연동 가중: 속도 오르면 저항 비중 단조 증가 (Fn² 증폭)."""
    from src.ai.shipgen_select import condition_weights

    slow = condition_weights("survey", 0.9, 3.0)
    fast = condition_weights("survey", 1.8, 3.0)
    assert fast["resistance"] > slow["resistance"]
    assert abs(sum(fast.values()) - 1.0) < 1e-9
    # 저 Fn에서는 프리셋 그대로 (클램프 1)
    from src.ai.recommend import preset_weights
    p = preset_weights("survey")
    crawl = condition_weights("survey", 0.3, 3.0)
    assert crawl["resistance"] == pytest.approx(p["resistance"])
