# 선박 AI 설계 파이프라인 (PoC)

> "이런 배가 필요해요"라고 말하면 → AI가 배 모양을 설계하고 → 물리 법칙으로 검증해주는 프로그램.
> 조선해양공학 + 로보틱스를 단계별로 배우는 학습 프로젝트.

## 1. 이게 뭐 하는 물건인가

자판기에 비유하면:

```
[투입]  "100kg 싣고, 1.5 m/s로 다니는, 조사용 무인 보트"
   ↓
[기계]  ① 치수 계산 → ② 3D 배 모양 생성 → ③ 물에 뜨는지·안 뒤집히는지 검증
        → ④ 밀고 가는 데 필요한 힘 계산
   ↓
[배출]  3D 모델 파일(STL) + 검증 리포트 "이 배는 2.72m, 추력 29N 필요, 합격"
```

핵심 아이디어: 배 치수를 사람이 정하는 게 아니라 **"목적"만 넣으면 나머지는 자동**.

## 2. 지금까지 만든 것 (2026-08-05 기준)

파이프라인 한 줄 실행:

```bash
python -m src.pipeline --speed 1.5 --payload 100 --purpose survey --out outputs/demo
```

내부에서 벌어지는 일, 순서대로:

| 단계 | 모듈 | 하는 일 (일상어) |
|---|---|---|
| ① 치수 추정 | `src/ai/dimension_estimator.py` | "조사용이면 보통 길이:폭 = 3:1" 같은 실선 비율 통계로 L, B, 깊이 결정 |
| ② 속도 체계 판정 | `src/core/regime.py` | 이 속도면 물을 "밀고 가는" 배(배수량형), 중간(반배수량형), "타고 달리는" 보트(활주형)인지 판별 — **세 체계 전부 설계 지원** |
| ③ 3D 선형 생성 | `src/ai/hull_generator.py` | 수학 공식으로 3D 배 표면 생성 (저속=Wigley, 고속=트랜섬 선미, 활주=V바닥 데드라이즈). 구멍 없는(watertight) 메쉬 |
| ④ 무게 추정 | `src/physics/weights.py` | 선체 무게 + 배터리·모터 + 짐 = 전체 무게, 무게중심 높이(KG) 계산 |
| ⑤ 뜨는지 검증 | `src/physics/hydrostatics.py` | 그 무게로 물에 넣으면 어디까지 잠기나(흘수), 기울여도 되돌아오나(GM) 판정 |
| ⑥ 저항 계산 | `src/physics/resistance.py` | 이 속도로 가려면 몇 N으로 밀어야 하나 = 모터 크기의 근거 |
| ⑦ 점수 매기기 | `src/hitl/scoring.py` | 사람(당신)이 결과에 1~5점 — 나중에 AI 재학습 때 반영 |

**품질 보증:** 테스트 282개. 전부 "정답을 손으로 계산할 수 있는 문제"로 검증
(예: 직육면체 바지선은 공식이 있음 → 코드 답과 대조). 저항 모듈은 국제 표준 시험 선형(Wigley)의
문헌값과 대조 — 오차 대역 안.

**Phase A 이후 추가된 능력 (7/26~30):**

| 능력 | 내용 |
|---|---|
| 실선 데이터 기반 | 치수·속도 프리셋이 실제 USV 9척 통계에서 나옴 (`--payload`와 `--purpose`만으로 실행 가능) |
| 다목적 최적화 | NSGA-II 파레토 — 저항·중량·안정여유 트레이드오프 지도에서 사람이 선택 |
| Ship-D 3만 척 | MIT 공개 선형 데이터셋 연동 + 대리모델(MLP) 가상 전수 선별 |
| 쌍대비교 ELO | "두 배 중 어느 쪽?" 클릭으로 인간 선호 랭킹 축적 |
| **Gazebo×ROS2** | 설계된 배가 로봇 시뮬레이터의 물에 떠서 **자율 웨이포인트 완주** (Phase B 완결) |
| 반배수량 | 트랜섬 선형 + 고속 영역(Fn<1.0) 설계 개방 — 고속 순찰 USV 가능 (Phase C-1) |
| **활주형** | Savitsky 1964 경험식으로 "물을 타고 달리는" 보트까지 개방 — **세 속도 체계 전부 커버** (Phase C-2) |
| **탑재 공간 검사** | 선체 안 최대 상자(MaxBox) vs 짐 부피 — 무게는 실려도 공간이 안 나오는 배를 걸러냄 (#27, `--payload-volume`) |
| **다구획 적재** | 실선처럼 칸막이로 나눠 싣기 — 예약 구역(모터·배터리) 절단 + 4중 게이트 (무게·공간·GM 배치·트림), Ship-D 선별기 재개방 (`src/physics/cargo_hold.py`) |
| **실선급 선형** | 용도별 선저 (round bilge, Ship-D 300척 실측 Cm) + 선수미 비대칭 기본 (LCB 실측·문헌 캘리브레이션) + 적재 처방 ("짐을 어디 실으세요" — 트림 0° 역산) |
| **러더(방향타)** | 조타·추력 분리 — 실물 스펙(Mandel 양력식·DNV 면적·서보) + 파이썬·Gazebo(직립 세계) 교차 검증. 저속 추종은 러더 필수 입증 (0.73 추종 vs 차동 1.61 폭주) (`--steering diff\|rudder2\|rudder1`) |
| **CFD 훅 + 능동학습** | OpenFOAM 정밀 해석 연동 — 격자 캠페인으로 방법 한계까지 정직 기록 |

물리 검증 원칙은 그대로: 모든 모듈이 해석해·문헌·독립 구현 대조를 통과해야 병합 (테스트 282개).

**핵심 용어 4개만:**
- **Cb (방형계수)**: 배가 상자에 얼마나 꽉 차는가. 0.45 = 날씬(빠름), 0.55 = 뚱뚱(짐 많이)
- **흘수 (draft)**: 물에 잠기는 깊이
- **GM**: 복원력 지표. 크면 안 뒤집힘, 너무 크면 멀미 나게 흔들림 — 그래서 "밴드"로 판정
- **Froude 수 (Fn)**: 속도÷√(중력×길이). 배의 "상대적 속도". 0.4 넘으면 물리 법칙이 달라짐

## 3. 직접 해볼 수 있는 것

```bash
# 배 한 척 설계 (용도: survey / patrol / workboat)
python -m src.pipeline --speed 1.5 --payload 100 --purpose survey --out outputs/demo

# 결과 3D 모델 보기 (맥 미리보기로 열림)
open outputs/demo/hull.stl

# 상세 수치 리포트
open outputs/demo/report.json

# 전체 테스트 (모든 물리 검증 통과 확인)
python -m pytest

# 설계 결과에 점수 매기기 (파이썬에서)
python3 -c "from src.hitl.scoring import record_score; record_score('demo_001', 4, 'data/user_scores.csv')"
```

속도를 3.0으로 올리면 반배수량 트랜섬 선형, 6.0이면 활주형(V바닥)으로 자동 전환됨.
불가능한 조합(모터 카탈로그 초과, 활주 평형 불성립)은 사유와 함께 정직하게 거절.

```bash
# 고속 순찰정 설계 (반배수량, Phase C-1)
python -m src.pipeline --speed 3.0 --payload 100 --purpose patrol --loa 4.3

# 활주정 설계 (Savitsky, Phase C-2) — 활주는 전기를 많이 먹어 항속시간을 짧게
python -m src.pipeline --speed 6.0 --payload 20 --purpose patrol --loa 2.8 --endurance 1.0

# 파레토 최적화 (후보 여러 척 → 트레이드오프 지도)
python -m src.optimize --speed 1.2 --payload 100

# CFD 훅 (원안 Step 5): 설계 결과를 OpenFOAM 정밀 해석으로 (Docker 필요)
python -m src.cfd.hook --report outputs/demo --mode simple   # 케이스 생성
cfd/docker/run_case.sh outputs/cfd_cases/demo_simple_1.5ms simpleFoam
python -m src.cfd.hook --report outputs/demo --mode simple --parse-only  # 라벨 수확
```

## 4. 앞으로 할 일 (로드맵)

| 순서 | 이름 | 내용 | 끝나면 얻는 것 |
|---|---|---|---|
| ✅ | M4·M5·Phase B·C-1·**C-2** | 동역학, 최적화, Ship-D, ELO, Gazebo 완주, 반배수량, **활주(Savitsky) — 전 속도 체계 개방** | |
| ✅ | **CFD 훅 (원안 Step 5)** | 설계 산출물 → OpenFOAM 케이스 자동 생성 → Docker 실행 → 저항 라벨 축적. Wigley 실증: 격자 7.8만 셀, CFD 점성 vs ITTC-57 마찰 자릿수 일치 | **원안 6단계 전부 최소 1회 실증** — 능동 학습 재료(cfd_labels.csv) 준비 완료 |
| ✅ | **러더 캠페인** | 가설→옵션→실물 스펙→Gazebo 교차 검증 4단계 — 자기지속 루프의 하드웨어 해법 | 저속 운용 = 러더 필수 지침 |
| ✅ | **다구획 MaxBox** | 적재 공간 모형(오너 발상) + 구간 판정 (KG/GM·트림) — 4중 게이트 | Ship-D 선별기 재개방 |
| ✅ | **특징 v3** | 다충실도(저해상 Michell) + 평형 정합 | 저항 R² 0.386→0.476 (+23%), 재검증 81% |
| ✅ | **gz 부양 자세 캠페인** | 정적 전복 발견(메타센터 부재)→collision 격자 분할 완치→직립 재캘리브레이션 (부호·이득) | 직립 신 기준선: 차동 60s σ2.54 / 러더 70s σ2.52, 저속 0.73 추종 — 교차 검증 재확정 |
| ✅ | **실선급 선형 3부작** | 용도별 선저(Cm) + 선수미 비대칭(LCB) + 적재 처방 — 오너 관찰 3연타가 출발점 | 생성 배가 실제 배 형상·수평 부양 |
| 진행 | ELO 신형식 | 유사 형태 매칭 + 무승부 + 매체 개선 (단면 겹침 등) | 사람 취향 축적 재개 |
| | 백로그 | patrol 선저 배선, 배수량 트랜섬, 실선 데이터 확충 | |

## 5. 당신(프로젝트 오너)에게 부탁할 일

**지금부터 습관으로:**
1. **점수 매기기** — 배 만들 때마다 STL 열어보고 1~5점 기록 (위 명령어). M5b에서 이 데이터가 "인간 취향" 학습 재료가 됨. 지금부터 쌓아야 그때 쓸 게 있음.
2. **결과 구경** — report.json에서 GM, 저항 숫자 눈에 익히기. "속도 올리면 조파저항이 확 뛴다" 같은 감이 생기면 그게 조선공학 학습.

**완료된 결정:** ROS2 환경(Mac Docker→WSL2 확장 경로), Ship-D 다운로드 — 전부 확정됨.

**궁금할 때:**
3. 코드·용어 아무거나 질문 — 설명이 이 프로젝트 목적의 절반. (기록: docs/feedback-log.md)

## 6. 옵시디언으로 읽기 (추천)

이 저장소는 그대로 옵시디언 vault입니다: **옵시디언 → "Open folder as
vault" → 이 프로젝트 폴더 선택** 후 `docs/00-시작-여기부터.md`를
여세요 — 쉬운 설명서를 현관으로 순서대로 안내하는 길잡이가 있습니다.
(비전공자 기준으로 쓰였습니다 — 조선공학 지식 불필요)

## 7. 저장소 지도

```
docs/INDEX.md             문서 지도 (여기부터)
docs/easy-manual.md       🚢 쉬운 설명서 — 비전공자용 전체 이야기 (상시 최신화)
docs/PROGRESS.md          현황판 · docs/worklog/ 일지 · docs/feedback-log.md 피드백 기록지
docs/superpowers/         설계 스펙(v3) + 구현 계획서들
src/                      실제 코드 · tests/ 검증 172개
data/                     실선 USV·모터 카탈로그·ELO 이력 (Ship-D는 로컬 전용)
ros2_ws/docker/           시뮬 이미지·제어 노드·실험 스크립트 (Phase B)
outputs/                  실행 결과물 (git 미포함)
```
