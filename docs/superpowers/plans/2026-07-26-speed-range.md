# 속도 실현가능 범위 함수 구현 계획 (백로그 #15, 오너 제안 Q2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 선체 길이 → 낼 수 있는 속도 상한(hull speed), 요구 속도 → 필요한 최소 길이를 양방향으로 계산하고, 리포트와 거절 메시지에 구체적 숫자로 표시.

**Architecture:** `src/core/regime.py`에 순수 함수 2개 추가 (물리는 기존 FN_DISPLACEMENT_MAX 상수 재사용 — 새 상수 없음, 일관성 보장). 파이프라인 리포트에 필드 1개 + 출력 1줄, 거절 메시지에 대안 숫자 포함.

**Tech Stack:** 기존 스택 그대로.

## Global Constraints

- M1~M3 계획의 Global Constraints 승계.
- 브랜치: `feat/speed-range`.
- 경계 물리: v_max = FN_DISPLACEMENT_MAX·√(g·L), L_min = (v/FN_DISPLACEMENT_MAX)²/g — 서로 역함수. 같은 상수에서 유도 (하드코딩 중복 금지).

---

### Task 1: regime.py 양방향 함수

**Files:** Modify `src/core/regime.py`, `tests/test_regime.py`

**Interfaces:**
- Produces: `max_displacement_speed(loa: float) -> float`, `min_loa_for_speed(speed_ms: float) -> float`

- [ ] **Step 1: 실패 테스트 추가** — `tests/test_regime.py`

```python
from src.core.regime import max_displacement_speed, min_loa_for_speed


def test_max_speed_known_value():
    # L=4 m: v_max = 0.4·√(9.81·4) = 2.505... m/s
    assert max_displacement_speed(4.0) == pytest.approx(2.5054, abs=1e-3)


def test_max_speed_grows_with_length():
    assert max_displacement_speed(9.0) > max_displacement_speed(4.0)


def test_roundtrip_inverse():
    """역함수 관계: min_loa(max_speed(L)) == L."""
    loa = 5.3
    assert min_loa_for_speed(max_displacement_speed(loa)) == pytest.approx(
        loa, rel=1e-9
    )


def test_boundary_consistency_with_classify():
    """v_max 바로 아래는 배수량형, 바로 위는 아님 — 경계 상수 일관성."""
    loa, vol = 4.0, 0.3
    v = max_displacement_speed(loa)
    assert classify(v * 0.999, loa, vol) is Regime.DISPLACEMENT
    assert classify(v * 1.001, loa, vol) is not Regime.DISPLACEMENT
```

- [ ] **Step 2: 실패 확인** — `python -m pytest tests/test_regime.py -v` → ImportError
- [ ] **Step 3: 구현** — `src/core/regime.py`에 추가

```python
def max_displacement_speed(loa: float) -> float:
    """이 길이의 선체가 배수량형으로 낼 수 있는 속도 상한 [m/s].

    hull speed: 자기가 만든 파도의 길이가 배 길이와 같아지는 속도(Fn≈0.4)
    부터 저항이 급증 — 긴 배일수록 빠를 수 있다.
    """
    return FN_DISPLACEMENT_MAX * math.sqrt(G * loa)


def min_loa_for_speed(speed_ms: float) -> float:
    """이 속도를 배수량형으로 내려면 필요한 최소 선체 길이 [m] (역함수)."""
    return (speed_ms / FN_DISPLACEMENT_MAX) ** 2 / G
```

- [ ] **Step 4: 통과 확인** — 기존 + 신규 전부 PASS
- [ ] **Step 5: 커밋** — `feat: hull speed 양방향 함수 (속도 상한 ↔ 최소 길이)`

---

### Task 2: 파이프라인 통합 (리포트 + 거절 메시지)

**Files:** Modify `src/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- report dict에 `"max_displacement_speed"` (float) 추가.
- 거절 시(exit 3) stderr에 두 대안 숫자: 이 크기의 한계속도 / 이 속도에 필요한 최소 길이.

- [ ] **Step 1: 실패 테스트 추가** — `tests/test_pipeline.py`

```python
def test_report_contains_speed_limit(tmp_path):
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=100.0, purpose="survey")
    report = run_pipeline(goal, tmp_path)
    vmax = report["max_displacement_speed"]
    assert vmax > report["goal"]["target_speed_ms"]  # 통과했으니 여유 있어야


def test_rejection_message_has_alternatives(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline",
         "--speed", "6.0", "--payload", "100", "--purpose", "survey",
         "--out", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 3
    err = result.stdout + result.stderr
    assert "한계속도" in err   # 이 크기가 낼 수 있는 속도
    assert "최소" in err       # 이 속도에 필요한 길이
```

- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현** — `src/pipeline.py` 수정

import에 `max_displacement_speed, min_loa_for_speed` 추가 (`src.core.regime`에서).

`run_pipeline`에서 `require_supported(regime)`를 감싸 컨텍스트 있는 메시지로 승격:

```python
    vmax = max_displacement_speed(dims.loa)
    try:
        require_supported(regime)
    except UnsupportedRegimeError as e:
        raise UnsupportedRegimeError(
            e.regime,
            f"{e} | 추정 선체 L={dims.loa:.2f} m의 배수량형 한계속도는 "
            f"{vmax:.2f} m/s입니다. 목표 {goal.target_speed_ms} m/s를 내려면 "
            f"최소 L={min_loa_for_speed(goal.target_speed_ms):.2f} m가 필요합니다."
        ) from e
```

report dict에 `"max_displacement_speed": vmax,` 추가. `_print_summary`에:

```python
    print(f"한계속도(참고)  : {report['max_displacement_speed']:.2f} m/s "
          f"(이 선체 길이의 배수량형 상한)")
```

- [ ] **Step 4: 전체 스위트 + 실물 실행** — 통과 케이스와 거절 케이스(speed 6.0) 둘 다 실행해 메시지 확인
- [ ] **Step 5: 커밋 + 병합** — `feat: 속도 실현가능 범위 리포트 (#15 오너 제안)` → main 병합 → PROGRESS.md 갱신

## Self-Review

1. 커버리지: 오너 제안 Q2(치수→속도 범위 함수) ✓, 역방향(속도→최소 길이) ✓, 사용자 노출(리포트+거절 메시지) ✓.
2. 플레이스홀더 없음.
3. 타입 일관성: FN_DISPLACEMENT_MAX 단일 출처 — classify와 max_displacement_speed가 같은 상수 ✓ (경계 일관성 테스트가 이를 고정).
