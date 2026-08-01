"""추천 가중 (#25 2단계) — 손계산 정답지."""
import math

import pandas as pd
import pytest

from src.ai.recommend import blend, elo_weights, preset_weights, recommend


def test_preset_weights_sum_to_one():
    for purpose in ("survey", "patrol", "workboat"):
        w = preset_weights(purpose)
        assert sum(w.values()) == pytest.approx(1.0)
    assert preset_weights("survey")["stability"] == 0.5   # 안정 우선
    assert preset_weights("patrol")["resistance"] == 0.5  # 저항 우선
    assert preset_weights("workboat")["mass"] == 0.5      # 중량 우선


def test_elo_weights_softmax_hand_calc(tmp_path):
    """대결 2건 (경량>안정, 안정>저항 — 실제 이력 구조) → 경량 최대."""
    csv = tmp_path / "duels.csv"
    csv.write_text(
        "winner,loser,timestamp\n"
        "x_max_stability,y_min_resistance,t1\n"
        "z_min_mass,x_max_stability,t2\n")
    w, n = elo_weights(csv)
    assert n == 2
    assert w["mass"] > w["stability"] > w["resistance"]
    assert sum(w.values()) == pytest.approx(1.0)


def test_elo_weights_missing_file(tmp_path):
    w, n = elo_weights(tmp_path / "none.csv")
    assert n == 0
    assert w["mass"] == pytest.approx(1 / 3)   # 이력 없음 → 균등


def test_blend_alpha_hand_calc():
    """α = min(1, n/10): n=2 → 프리셋 0.8 + ELO 0.2."""
    preset = {"resistance": 0.5, "mass": 0.2, "stability": 0.3}
    elo = {"resistance": 0.0, "mass": 1.0, "stability": 0.0}
    w = blend(preset, elo, n_duels=2)
    assert w["mass"] == pytest.approx(0.8 * 0.2 + 0.2 * 1.0)
    w_full = blend(preset, elo, n_duels=50)
    assert w_full["mass"] == pytest.approx(1.0)   # α 상한 1


def test_recommend_picks_weighted_best(tmp_path):
    """survey(안정 0.5)면 '안정 좋고 나머지도 무난한' 균형점이 추천.

    극단 2점은 정규화가 0/1 대칭이라 동점 — 3점째(균형점)가 이겨야
    가중이 실제로 작동한다는 증거. 손계산: C 점수 0.819 > A=B=0.5."""
    df = pd.DataFrame([
        {"resistance_n": 5.0, "total_mass_kg": 100, "stability_margin": 0.02},
        {"resistance_n": 9.0, "total_mass_kg": 140, "stability_margin": 0.20},
        {"resistance_n": 6.0, "total_mass_kg": 110, "stability_margin": 0.18},
    ])
    csv = tmp_path / "empty.csv"   # 이력 없음 → 프리셋 그대로
    idx, w, why = recommend(df, "survey", elo_csv=csv)
    assert idx == 2
    assert "안정" in why
