# 조종성 — 7번째 게이트 "돌라면 도나" (2026-08-09)

- 상태: 오너 승인 (방향 "조종성", 모델 "MMG 정통", 판정 "가장
  퀄리티" = 대형 7번째 게이트 + 소형 성적표, 진행 "A. 검증 먼저")
- 유래: 구조 강도 캠페인 완주 직후 세 번째 심화 캠페인. 현
  파이프라인은 선형 Clarke 계수(작은 타각 전용)와 민첩성 지표만 —
  표준 조종 시험(35° 선회권·지그재그)과 IMO 기준은 미개척.

## 1. 물리 배경 (학부 2학년 눈높이)

- 조종성 판정의 업계 표준 = **선회권** (타 35°: 전진거리 advance·
  횡이동 transfer·선회지름 tactical diameter, 배 길이 배수로 평가)
  + **지그재그** (타 10°↔반대 10°: 오버슈트 각 — 배가 얼마나
  과하게 돌아가나 = 침로 안정성)
- IMO MSC.137(76): advance ≤ 4.5L, 선회지름 ≤ 5L, 오버슈트 한계 등
- **MMG 모델**: 힘을 선체(H)·프로펠러(P)·러더(R)로 분해 — 선회 중
  사항각 β·각속도 r의 비선형 다항 (선체), (1−w)·Kt (프로펠러),
  프로펠러 후류 증속 + aH·xH 선체 유도력 (러더). 선형 Clarke는
  35° 선회를 못 다룸 — 비선형 필수.

## 2. 구조 — 신설 `src/physics/maneuvering/` 패키지

| 모듈 | 역할 |
|---|---|
| `mmg.py` | MMG 3자유도 (surge·sway·yaw) 운동방정식 — 계수 주입형 (배와 분리, dataclass 계수 세트) |
| `kvlcc2.py` | Yasukawa & Yoshimura (2015) KVLCC2 공개 계수 (A급 표준기) |
| `trials.py` | 표준 시험 실행기 — turning_circle(δ, 통상 35°) → advance·transfer·tactical diameter, zigzag(10/10·20/20) → 오버슈트 각. 시간 적분 (RK4 또는 semi-implicit Euler, dt 수렴 확인) |
| `estimation.py` | 우리 배 → MMG 계수 추정 (Kijima 계보 공개 회귀 — 등급 정직 표기, 원전 확보 실패 시 C급 대체 기록) |
| `criteria.py` | IMO MSC.137(76) 기준 + maneuvering_gate (대형 전용) |

## 3. 원전·검증 앵커 (기억 하드코딩 금지 관례)

- 원전 확보 순서:
  1. **Yasukawa & Yoshimura (2015)** "Introduction of MMG standard
     method for ship maneuvering predictions", J. Marine Science
     and Technology 20:37-52 — 오픈액세스, MMG 정식화 + KVLCC2 전
     계수 + **자유항주 시험 실측 수록 = 주 앵커**
  2. IMO Resolution MSC.137(76) 공개 PDF (기준값)
  3. Kijima et al. 계수 추정 회귀 (2단계 재료)
  4. SIMMAN (simman2008.dk) — 보조 (접속 불가 시 논문 실측으로 충분)
- 앵커 사다리:
  1. 타각 0 → 직진 유지 (자기검증, 전 선형 상시)
  2. ±35° 선회 거울 대칭 (좌우 대칭 모델)
  3. **KVLCC2 35° 선회권 실측 대조** (논문 수록 시험값 — 오차 대역
     정직 기록)
  4. KVLCC2 지그재그 오버슈트 대조
  5. 우리 100m 화물선 IMO 여유 sanity

## 4. 파이프라인 통합

- `run_pipeline(..., maneuvering=True)` 기본 + `--no-maneuvering`
  (내항·구조 관례 재사용)
- **대형**: estimation 계수 → 표준 시험 → IMO 판정 → **7중
  passed** = hydro ∧ space ∧ seakeeping ∧ structure ∧ maneuvering
- **소형**: 성적표만 (IMO 범위 밖 — 기준 없는 외삽 금지) + 기존
  민첩성 지표 병존. 추정식 외삽 초과 = 정직 스킵
- Gazebo 선회 교차 검증 = 3단계 마지막 조각 (기존 러더 직립 세계
  재활용 — 러더 4단계 관례)

## 5. 단계 분할 (A. 검증 먼저 — 각 단계 성적표 후 계속/보류)

1. **표준기**: 원전 확보 + mmg.py + kvlcc2.py + trials.py —
   KVLCC2 실측 재현 성적표 (모델 오차 단독 계측)
2. **사상**: estimation.py — 우리 배 계수 + 표준 시험 (추정 오차
   분리 계측)
3. **판정·통합**: criteria.py + 7번째 게이트 + Gazebo 교차 검증
   + 4축 마감

## 6. 관례·리스크

- TDD 층별, e2e 명시 플래그 격리, main 직커밋
- 리스크: ① 계수 추정 회귀 = 상선 통계 — 소형 USV 외삽 C급 표기
  (Clarke 구도 재현) ② 프로펠러·러더 파라미터는 기존 모듈 실측
  재사용 (Wageningen B 직경·rudder_specs 면적) ③ MMG 적분 비용
  낮음 (수천 스텝) — 슈트 부담 소
- 기존 자산 관계: Clarke 선형 계수·python_sim은 소형 웨이포인트
  시뮬 정본 유지 (LOS 유도) — MMG는 표준 시험·게이트 전용 (역할
  분리, 통합은 후속 검토)
