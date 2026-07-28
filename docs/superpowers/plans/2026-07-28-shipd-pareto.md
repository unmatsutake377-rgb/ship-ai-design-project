# Ship-D 파레토 (M5b-2 2차) 구현 계획 — 대규모 선별

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) 구문.

**Goal:** Ship-D 3만 척에서 표본 N척을 USV 스케일로 전이해 우리 물리(정역학+이중검증 Michell)로 전수 평가 → 실선형(벌브·트랜섬) 파레토 전선.

**접근 근거:** 45차원 NSGA-II 직접 생성은 무작위 벡터 대부분이 제약 위반(자기교차) — 유효성 보증된 기존 3만 척의 **선별**이 정직한 1차. 생성 최적화는 대리모델 이후.

## Task 1: 저항 평가의 메쉬 경로 (리팩토링)

`total_resistance`가 Wigley 해석 Michell 고정 → 파형저항 함수를 주입 가능하게:
- `total_resistance_mesh(mesh, loa, draft, speed)` 신설 (ITTC 마찰은 기존 재사용 + `michell_wave_resistance_mesh`)
- `design_spiral(mesh, dims, goal, resistance_fn=None)` — 기본은 기존(해석), 주입 시 메쉬 경로. 기존 테스트 전부 불변 통과 = 검증.

## Task 2: 선별 모듈

**Files:** Create `src/screen_shipd.py`, `tests/test_screen_shipd.py` (skipif Ship-D 부재)

- `evaluate_shipd_hull(vector, goal, target_loa) -> dict` — scaled_mesh → 설계 나선(메쉬 저항) → 필터 → {저항, 중량, 안정여유, feasible, reason}
- `screen(goal, target_loa, n_samples, seed) -> DataFrame` — 무작위 표본 평가 (진행 로그), 파레토 비지배 추출 열 `pareto=True/False`
- CLI: `python -m src.screen_shipd --speed 1.2 --payload 100 --loa 3.0 --n 300 --out outputs/shipd_pareto` → CSV + 산점도(전체 회색·파레토 강조) + 파레토 상위 3척 STL·렌더
- 테스트: 표본 8척 스모크 (feasible ≥1, 파레토 비지배성, 필드 완전성)

## Task 3: 실전 실행 + 오너 보고

N=300 백그라운드 실행 → 파레토 플롯 + 실선형 렌더 오너 전달. Wigley 파레토(어제)와 같은 목적 조건으로 비교 논평. PROGRESS 갱신.

## Self-Review
- 스펙 v3 §7 "2차: Ship-D 공간" ✓ (선별 형태 — 근거 명시), 이중검증 Michell 사용 ✓, ELO 후보 공급 ✓.
- 평가 ~3-4 s/척 × 300 ≈ 15-20분 — 백그라운드 실행.
