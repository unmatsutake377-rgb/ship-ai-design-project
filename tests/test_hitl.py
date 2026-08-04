

def test_similar_pairs_prefers_close_forms():
    """매치메이킹 (오너 결정 2026-08-04): 형태 유사쌍 우선, 중복 금지."""
    import pandas as pd

    from src.hitl.matchmaking import similar_pairs

    df = pd.DataFrame({
        "loa": [2.0, 2.05, 3.5, 3.45, 1.0],
        "lb":  [3.0, 3.05, 4.0, 4.05, 2.0],
        "cb":  [0.45, 0.45, 0.50, 0.50, 0.40],
        "resistance_n": [10.0, 14.0, 8.0, 6.0, 20.0],
        "total_mass_kg": [140, 132, 155, 160, 120],
        "stability_margin": [0.10, 0.05, 0.15, 0.18, 0.03],
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
