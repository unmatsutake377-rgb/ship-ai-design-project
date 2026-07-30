# CFD 훅 (원안 Step 5) 설계 — OpenFOAM 인터페이스 + Wigley 실증

- 날짜: 2026-07-30
- 상태: 오너 승인 (범위·물리 수준·대상 선체·접근 A 모두 대화로 확정)
- 위치: 원안 6단계의 마지막 조각. 완료 시 원안 전 단계가 최소 1회 실증됨

## 1. 목적과 범위

**목적**: 경험식(Michell·ITTC-57·Savitsky)으로 걸러낸 파레토 상위 후보를
고정밀 CFD(OpenFOAM)로 재검증하고, 그 정밀 라벨을 축적해 대리모델
재학습(능동 학습)의 입력을 준비한다.

**이번 범위 (오너 선택: "인터페이스 + 초소형 실증 1척")**:
- 케이스 생성 → Docker 실행 → 결과 파싱 → 라벨 병합의 **배관 전체 관통**
- 실증은 Wigley 표준 선형 1척, 2단계: 단상(simpleFoam)으로 배관 확인 후
  자유수면(interFoam) 승격
- 능동 학습의 **재학습·재선별 사이클은 범위 밖** — 라벨 CSV 축적까지

**범위 제외 근거**: 실증 없는 인터페이스는 미검증 코드(프로젝트 원칙 위반),
전체 능동 학습 루프는 척당 수 시간 × K척으로 세션 여러 개 분량.

## 2. 아키텍처 (접근 A: 케이스 템플릿 + Docker)

```
outputs/demo/                     기존 파이프라인 산출물 (hull.stl + report.json)
   ↓ 읽기
src/cfd/case_builder.py           템플릿 구멍({{SPEED}} 등) 메꾸고 STL 배치
   src/cfd/templates/simple/      단상(simpleFoam·이중모형)용
   src/cfd/templates/inter/       자유수면(interFoam)용
   → outputs/cfd_cases/<이름>/    완성된 OpenFOAM 케이스 폴더
   ↓ 실행
cfd/docker/run_case.sh            공식 OpenFOAM Docker 이미지 (~1GB)
   → postProcessing/.../force.dat 힘 시계열 로그
   ↓ 읽기
src/cfd/result_parser.py          수렴 판정 + 저항 [N] (압력/점성 성분 분해)
   ↓ 병합
src/cfd/labels.py                 data/cfd_labels.csv에 경험식 라벨과 나란히 축적
```

**결합 원칙**:
- 파이프라인과 **느슨 결합**: 훅은 산출물 폴더를 입력으로 받는 별도 CLI
  (`python -m src.cfd.hook`). `src/pipeline.py`는 무수정.
- 무거운 실행은 컨테이너에 (Phase B의 ship-sim 패턴 재사용).
- CFD는 저항을 압력/점성 성분으로 분해해 줌 → 경험식의 조파(rw)/마찰(rf)
  성분과 **성분별 대조** 가능 (총합만 맞는 우연 배제).

**거부한 대안**: B) PyFoam 등 전용 라이브러리 — 의존성·학습 비용이 PoC에
과잉, 템플릿 방식이 OpenFOAM 케이스 구조 학습에 더 적합. C) 외부 실행용
내보내기 전용 — 로컬 실증 불가로 오너 선택과 모순.

## 3. 케이스 내용

OpenFOAM 케이스 = 폴더 (0/ 초기조건, constant/ 물성·STL, system/ 수치설정).
배는 고정하고 물을 목표 속도로 흘림 (수조 예인 시험과 동일 발상).
배경 격자 상자: 선수 앞 1L · 선미 뒤 3L · 옆 1.5L · 아래 1L (L=배 길이),
좌우 대칭이라 절반 도메인. snappyHexMesh로 STL 표면에 격자 밀착.
난류는 k-ω SST 표준 모델.

| | simple (단상) | inter (자유수면) |
|---|---|---|
| STL 전처리 | 흘수선 아래만 절단 (이중모형 — 수면 위치에 미끄럼 벽) | 배 전체, setFields로 수위 지정 |
| 풀이기 | simpleFoam (정상상태) | interFoam (시간 전진, 물+공기 2상) |
| 얻는 것 | 마찰·점성 저항 | 조파 포함 전저항 |
| 예상 시간 | 분 단위 | 수십 분~ (거친 격자) |

**플랜 B 내장**: interFoam 발산 시 단상 결과만으로 1차 실증 인정, interFoam은
이슈 기록 후 후속.

## 4. 검증 계획 (3층)

1. **pytest (OpenFOAM 불필요 — 전부 analytic-answer)**
   - case_builder: 필수 파일 존재 / 안 메꿔진 `{{` 0개 / 격자 상자 좌표가
     L에 비례 / 단상 STL 최고점 z ≤ 흘수
   - result_parser: 답을 아는 가짜 force 로그 → 평균·수렴 판정 손계산 일치,
     발산 로그 → 미수렴 플래그
   - labels: 병합 스키마, 재실행 시 갱신 (중복 행 방지)
2. **Docker 실증 (수동 스크립트 — Phase B run_*_test.sh 패턴, pytest 밖)**
   - 단상: Wigley 1.5 m/s 마찰저항 vs ITTC-57 자릿수 대조
   - 자유수면: 전저항 vs (Michell 조파 + ITTC 마찰)
3. **문헌 3중 대조**: Wigley 수조 실험 공개 데이터로 경험식 vs CFD vs 실험
   대조표 (Michell 검증 때 쓴 문헌값 재사용 — Wigley 선정 이유)

**성공 기준**: 배관 관통(생성→실행→파싱→CSV 한 줄) + 단상 마찰이 ITTC-57과
같은 자릿수. interFoam 정확도는 성공 조건 아님 (기록만).

## 5. 알려진 한계 (정직 목록)

1. 거친 격자 — 숫자는 참고치. 격자 수렴 연구는 추후
2. 고정 자세 — 설계 흘수에 못박고 계산. 실제·실험은 트림·부상이 변함 →
   문헌값과 그만큼 어긋날 것
3. Wigley 배수량형 전용 — 활주·반배수량 템플릿은 후속 (활주는 동적 자세
   필수라 이 구조로 부족)
4. 능동 학습은 라벨 축적까지 — 재학습 사이클은 후속
5. Docker 이미지 ~1GB 추가 (ship-sim 4.5GB와 별개)
