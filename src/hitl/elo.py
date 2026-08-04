"""쌍대비교 ELO 랭킹 (spec §7 M5-ELO, 오너 제안·결정 2026-07-27).

왜 쌍대비교인가: "두 배 중 어느 쪽이 나은가"는 절대 점수(1~5)보다
초보 평가자에게 신뢰성 높음 — 비교 판단이 절대 판단보다 쉬움.

설계 원칙:
- 저장하는 것은 **비교 이력뿐** (승자, 패자, 시각). 레이팅은 이력 재생으로
  파생 — 상태 오염 없음, 언제나 재계산 가능.
- ELO 갱신: 기대승률 E = 1/(1+10^((상대−나)/400)),
  새 레이팅 = 현재 + K·(실제 − 기대). 이변일수록 변동 큼.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

INITIAL_RATING = 1500.0
K_FACTOR = 32.0
COLUMNS = ["winner", "loser", "timestamp"]
DRAW_MARK = "draw"   # reason 열 대신 별도 열 — 구형 행은 빈 값 (승부)


def expected_score(rating_a: float, rating_b: float) -> float:
    """A가 B를 이길 기대 확률."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def record_comparison(winner_id: str, loser_id: str,
                      csv_path: str | Path, reason: str = "",
                      draw: bool = False) -> None:
    """비교 결과 1건 기록 (append-only).

    reason: 선택 이유 (오너 제안 2026-08-02 — "수정 피드백을 같이
    적어주면 도움 되나?"에서 채택). 이유가 있으면 ① 오클릭 구분
    ② 취향의 구조가 데이터화 ③ 대결 조건의 결함(제어 거동 혼입 등)
    발견 — 세 몫을 한다. 점수 계산에는 미사용, 기록·분석용.

    draw=True (2026-08-04 신 ELO 1차전에서 도입): 무승부 — 표준
    ELO대로 양쪽 실득 0.5. 무승부도 정보다: "이 정도 차이는 사람
    눈에 구분 불가"라는 매치메이킹 문턱의 실측 라벨."""
    if winner_id == loser_id:
        raise ValueError(f"자기 자신과 비교 불가: {winner_id!r}")
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(COLUMNS + ["reason", "result"])
        writer.writerow(
            [winner_id, loser_id, datetime.now(timezone.utc).isoformat(),
             reason, DRAW_MARK if draw else ""]
        )


def compute_ratings(csv_path: str | Path) -> dict[str, float]:
    """비교 이력을 순서대로 재생해 현재 레이팅 산출."""
    path = Path(csv_path)
    if not path.exists():
        return {}
    ratings: dict[str, float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            w, l = row["winner"], row["loser"]
            rw = ratings.get(w, INITIAL_RATING)
            rl = ratings.get(l, INITIAL_RATING)
            e_w = expected_score(rw, rl)
            # 무승부: 실득 0.5 (구형 CSV엔 result 열 없음 — 승부 취급)
            score = 0.5 if row.get("result") == "draw" else 1.0
            ratings[w] = rw + K_FACTOR * (score - e_w)
            ratings[l] = rl - K_FACTOR * (score - e_w)
    return ratings
