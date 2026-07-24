# AI 기반 목적 지향형 선박 설계 & 시뮬레이션 파이프라인 — 설계 문서 (PoC)

- 날짜: 2026-07-24
- 상태: 사용자 설계 승인 완료
- 개발 환경: macOS (Apple Silicon), Python 3.13 (anaconda). ROS2/Gazebo 미설치.

## 1. 목적과 범위

사용자가 선박의 운항 목적(목표 속도, 적재량, 용도)을 입력하면, 최적 주요 치수(L, B, T, Cb)를
추정하고 3D 선형을 생성해 물리적으로 검증한 뒤, 동역학 시뮬레이션으로 거동을 확인하는
End-to-End 파이프라인의 개념 증명(PoC). 타겟은 USV 및 소형 선박.

**프로젝트 성격: 학습 목적.** 조선해양공학(정역학→저항→동역학→자율운항)과
로보틱스(ROS2)를 단계별로 배우는 순서에 맞춰 마일스톤을 구성한다.

### 원안(AI 생성 프롬프트) 대비 변경 사항과 근거

| 원안 | 변경 | 근거 |
|---|---|---|
| 6단계 동시 추진 | Phase A(순수 Python) → B(ROS2) → C(CFD/활주형) 순차 | 6단계는 PoC 하나가 아니라 연구 프로젝트 3개 분량. ROS2가 크리티컬 패스에 있으면 전체가 블로킹됨 |
| Step 4가 선형 검증 담당 | 선형 검증은 Step 3(정역학) + Python Fossen 모델이 담당, ROS2/Gazebo는 자율운항 테스트베드로 재정의 | Gazebo 해양 플러그인은 사람이 넣어준 계수로 움직이는 단순화 모델 — 선형 우수성을 스스로 검증하지 못함 |
| ROS2 즉시 구축 | Phase B에서 Docker + VRX(Virtual RobotX)로 구축 | macOS ARM은 ROS2 Tier 3 지원. 네이티브 설치 비추천. USV 시뮬레이션은 VRX 기성 스택 활용 |
| 처음부터 Ship-D + MLP | 1단계는 수식 기반 파라메트릭 선형(Wigley 계열), AI 모델은 M5에서 교체 장착 | 파이프라인 관통이 먼저. AI 학습 실패가 전체를 멈추지 않게 함 |
| 속도 체계 언급 없음 | Froude 수 기반 체계 디스패처를 core에 내장. 배수량형 먼저 구현, 활주형(Savitsky)은 2차 사이클 | 사용자 결정: "둘 다 지원". 경험식·저항식이 체계별로 완전히 다르므로 구조로 분리 |

## 2. 아키텍처

접근법: **물리 코어 우선 + 시뮬레이터 어댑터** (사용자 선택).

```
선박 ai 모델 프로젝트/
├── src/
│   ├── core/           # GoalSpec, ShipDesign 데이터클래스 + Froude 체계 디스패처
│   ├── ai/             # Step 1: 목적→치수 경험식 / Step 2: 치수→선형 생성기
│   ├── physics/        # 정역학(trimesh), 저항(체계별), 동역학(Fossen 3자유도)
│   ├── hitl/           # 점수 로깅(user_scores.csv) + PyTorch weighted loss
│   └── sim_adapters/   # python_sim(Phase A) / ros2_export(Phase B, URDF/SDF 변환)
├── ros2_ws/            # Phase B 자리 (현재 README만)
├── data/               # data_loader.py, Ship-D 실데이터, 합성 데이터
├── tests/              # pytest — 해석해 기반 물리 검증
└── docs/
```

핵심 설계 결정:

1. **`core.ShipDesign`이 유일한 모듈 간 인터페이스.** ai ↛ physics ↛ ros2 직접 의존 금지.
   선형 생성기를 파라메트릭→MLP→Diffusion으로 교체해도 다른 모듈 무변경.
2. **선형 생성 2단 로켓.** M3까지는 L,B,T,Cb → 수식 선형(watertight STL).
   M5에서 Ship-D 45파라미터 + MLP로 교체. Ship-D는 대형선 위주 분포이므로
   USV 스케일 적용 시 무차원화/스케일링 전략을 M5에서 확정한다.
3. **체계 디스패처 선행 내장.** Froude 수 < 0.4 → 배수량형 경로(구현),
   그 이상 → 활주형 경로(Phase C까지 명시적 "미구현" 안내 후 중단, 조용한 오답 금지).

## 3. 데이터 흐름 (Phase A 1회 실행)

```
GoalSpec(속도, 적재량, 용도)
  → [ai] 치수 추정 (경험식) + Froude 체계 판정
  → [ai] 파라메트릭 선형 생성 → watertight 메쉬(STL)
  → [physics] 정역학 필터: 배수량 일치, GM > 0
       불합격 = 정상 필터링 결과. 어떤 수치가 왜 미달인지 리포트 (조용히 버리지 않음)
  → [physics] 저항 추정 → 소요 추력
  → [sim_adapters.python_sim] Fossen 3자유도 웨이포인트 추종 → 궤적 플롯 + 리포트
  → [hitl] 사용자 1~5점 → user_scores.csv → weighted loss (재학습은 M5)
```

## 4. 테스트 전략

물리 코드는 "그럴듯한 오답"이 최대 위험이므로 해석해 기반 검증:

- 정역학: 직육면체 바지선(배수량·KB·BM·GM 손계산 가능) 기준값 비교
- 메쉬: 단위 정육면체·구의 부피/침수표면적
- 동역학: 추력 0 → 정지 유지, 일정 추력 → 이론 종속도 수렴 (물리 불변량)
- 데이터: Ship-D 스키마(45파라미터) 위반 데이터 거부 확인

## 5. 마일스톤

| # | 결과물 | 비고 |
|---|---|---|
| M1 | 디렉토리 구조 + core 타입 + data_loader(더미) + HITL 로깅 + pytest | 원안 Action Item 1~4 |
| M2 | trimesh 정역학 + 바지선 해석해 테스트 통과 | |
| M3 | 목적 입력→치수→선형→정역학 리포트 End-to-End CLI | 첫 관통 |
| M4 | Fossen 3자유도 + 웨이포인트 추종 궤적 플롯 | |
| M5 | Ship-D 실데이터 + MLP(Mac MPS) + weighted loss 연결 | Phase A.1 |
| B | Docker ROS2/VRX + URDF 내보내기 어댑터 | 환경 결정 후 |
| C | OpenFOAM 훅 + 활주형(Savitsky) 체계 | 설계만 선행 |

의존성: numpy, pandas, trimesh, matplotlib, pytest (M1~M4), torch (M5부터).

## 6. 성공 기준 (PoC "완료"의 정의)

Phase A 기준: 터미널에서 명령 한 줄로 목적 입력 → 물리적으로 타당한(GM>0, 배수량 일치)
선형 메쉬와 검증 리포트, 웨이포인트 추종 궤적이 출력되고, 모든 pytest가 통과하며,
사용자가 결과에 점수를 매기면 CSV에 누적되는 상태.
