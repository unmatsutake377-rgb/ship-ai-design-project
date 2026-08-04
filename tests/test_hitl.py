

def test_similar_pairs_prefers_close_forms():
    """매치메이킹 (오너 결정 2026-08-04): 형태 유사쌍 우선, 중복 금지."""
    import pandas as pd

    from src.hitl.matchmaking import similar_pairs

    df = pd.DataFrame({
        "loa": [2.0, 2.05, 3.5, 3.45, 1.0],
        "lb":  [3.0, 3.05, 4.0, 4.05, 2.0],
        "cb":  [0.45, 0.45, 0.50, 0.50, 0.40],
        "resistance_n": [10.0, 16.0, 8.0, 5.0, 20.0],
        "total_mass_kg": [140, 128, 155, 165, 120],
        "stability_margin": [0.10, 0.03, 0.12, 0.20, 0.03],
    })
    pairs = similar_pairs(df, n_pairs=2)
    got = sorted(tuple(sorted((int(a.name), int(b.name))))
                 for a, b, _ in pairs)
    assert got == [(0, 1), (2, 3)]     # 유사쌍 2개, 외톨이(4) 제외
    assert pairs[0][2] < 0.5           # 가까운 쌍의 거리


def test_similar_pairs_rejects_near_duplicates():
    """준중복(형태·목적 모두 동일) 쌍은 기각 — 같은 배 대결 방지."""
    import pandas as pd

    from src.hitl.matchmaking import similar_pairs

    df = pd.DataFrame({
        "loa": [2.0, 2.0, 2.1, 3.0],
        "lb":  [3.0, 3.0, 3.0, 3.1],
        "cb":  [0.45, 0.45, 0.45, 0.46],
        "resistance_n": [10.0, 10.0, 14.0, 9.0],
        "total_mass_kg": [140, 140, 135, 150],
        "stability_margin": [0.10, 0.10, 0.05, 0.15],
    })
    pairs = similar_pairs(df, n_pairs=1)
    ids = tuple(sorted((int(pairs[0][0].name), int(pairs[0][1].name))))
    assert ids != (0, 1)               # 완전 중복쌍 금지
    assert ids == (0, 2) or ids == (1, 2)   # 형태 유사 + 목적 갈림


def test_draw_recording_and_rating(tmp_path):
    """무승부 (오너 신 ELO 1차전, 2026-08-04): 양쪽 0.5점 갱신.

    같은 초기 레이팅끼리 무승부 = 레이팅 불변 (기대 0.5 = 실득 0.5)."""
    from src.hitl.elo import compute_ratings, record_comparison

    p = tmp_path / "cmp.csv"
    record_comparison("a", "b", p, reason="차이 없음", draw=True)
    r = compute_ratings(p)
    assert r["a"] == r["b"] == 1500.0
    # 승자 있는 대결 뒤 무승부: 강자는 살짝 잃고 약자는 살짝 얻음
    record_comparison("a", "b", p, reason="a 승")
    record_comparison("a", "b", p, reason="이번엔 비김", draw=True)
    r2 = compute_ratings(p)
    assert r2["a"] > r2["b"]           # 승 1회 우위는 유지
