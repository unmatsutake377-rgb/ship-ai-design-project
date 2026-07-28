# M5b-1 구현 계획 — 물리 라벨 대리모델 + 가상 전수 선별

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. 체크박스 구문.

**Goal:** 오늘 평가한 300척(45파라미터 → 저항·중량·안정여유·타당성)으로 MLP 대리모델을 학습 → **3만 척 전수 가상 선별** → 상위 후보만 진짜 물리로 재검증. "300척 표본의 한계"(외부 검토 지적)를 수량·방법 양면에서 해소.

**Architecture:**
- `src/ai/surrogate.py`: ① 타당성 분류기 (45→sigmoid, 300척 전체 학습) ② 목적 회귀기 (45→3, feasible 134척 학습). 입력·출력 정규화, torch MPS(Apple Silicon) 우선.
- `src/virtual_screen.py`: 학습 → 3만 척 예측(밀리초/척) → 예측-타당 중 예측-파레토 상위 K척 → **진짜 물리 재검증** (대리모델은 후보 추천만, 최종 수치는 항상 실물리) → 기존 134척과 합산 파레토.

**정직 원칙:** 대리모델 정확도(검증 분할 지표)를 리포트에 명시. 134척 학습은 소표본 — 순위 추천용이지 정밀 예측 아님을 문서화. 재검증 통과분만 결과로 인정.

## Task 1: surrogate.py (+ requirements에 torch)
- `SurrogateModel` (분류+회귀 겸용 래퍼), `train_surrogate(X, y_feas, Y_obj, epochs, seed) -> (model, metrics)` — metrics: 분류 정확도, 회귀 R² (8:2 분할)
- 테스트 (skipif Ship-D): 소형 학습 후 손실 감소, 예측 shape, 분류 정확도 > 0.6

## Task 2: virtual_screen.py
- screen.csv + 벡터 결합 → 학습 → 30k 예측 → 후보 K=80 → 재검증(기존 evaluate_shipd_hull 재사용) → 합산 파레토 CSV/플롯
- CLI: `python -m src.virtual_screen --speed 1.2 --payload 100 --loa 3.0 --topk 80 --out outputs/virtual_screen`
- 테스트: 파이프라인 스모크 (topk=3, epochs 소량)

## Task 3: 실전 실행 + 보고
- 재검증 후 신기록 여부 (저항 <12.1N? 중량 <126.9kg?) + 대리모델 지표 오너 보고. PROGRESS 갱신.

## Self-Review
- 외부 검토 대응(전수 탐색) ✓, 스펙 §7 M5b-1 ✓, 최종 수치는 실물리만 ✓, MPS 학습 경험 확보 ✓.
