# 경제성·EEDI — 8번째 게이트 "규제 통과하나" (2026-08-09)

- 상태: 오너 승인 (범위 "EEDI + 운항 경제 둘 다", 소형 "전기 등가
  성적표", 판정 "8번째 게이트")
- 유래: 3대 심화 캠페인 (내항·구조·조종) 완주 후 네 번째 축 —
  물리 사슬 (저항→동력→SFOC→연료)이 완비돼 리턴 확실. 배를
  "돈과 규제"의 언어로 번역하는 캠페인.

## 1. 배경 (학부 2학년 눈높이)

- **EEDI** (Energy Efficiency Design Index) = 짐 1톤을 1해리 나를
  때 CO₂ 몇 그램: EEDI = CF·SFC·P_ME / (Capacity·Vref)
  [gCO₂/(t·nm)] — 분자 "시간당 CO₂ 배출"(75% MCR 동력 × 연비 SFC
  × 탄소계수 CF), 분모 "시간당 수송량"(DWT × 그 동력에서의 속도)
- IMO **기준선** = a·DWT^(−c) (선종별 계수, MARPOL Annex VI reg
  24 공개) — Phase 3 (2025+) 신조선은 기준선 대비 30% 감축 요구.
  attained ≤ required = 합격
- 운항 경제: 연간 연료비 = P×SFOC×운항시간×벙커가, **수송 단가**
  = 연료비/(DWT×연간 수송 해리) — 설계 비교의 돈 차원
- 소형 전기 USV: EEDI 적용 밖 (국제항해 400GT+) — 같은 철학의
  전기 등가 지표 **Wh/(kg·km)** + 전기료 운용비 (규제 아닌
  성적표, 정직 표기)

## 2. 구조 — 신설 `src/physics/economics/`

| 모듈 | 역할 |
|---|---|
| `eedi.py` | attained EEDI (75% MCR·CF·SFC·Vref 재역산) + required (기준선 a·DWT^(−c) × Phase 감축) — 원전 계수 박제 |
| `opex.py` | 대형: 연간 연료비·톤·해리 수송 단가. 소형: Wh/(kg·km) 수송 에너지 단가 + 전기료 운용비 |
| 게이트 | `economics_gate` — 대형 attained ≤ required (Phase 3) → **8중 passed** = … ∧ economics. 소형 = 성적표만. `economics=True` 기본 + `--no-economics` (관례) |

Vref: 우리 설계는 순항 속도·엔진 부하율 실산출 — 75% MCR 속도는
저항 곡선 역산 (P(V) ∝ V³ 근방 보간, 정직 표기) 또는 설계 속도
병기 (원전 가이드라인 확인 후 확정).

## 3. 원전·검증 앵커 (기억 하드코딩 금지 관례)

- 원전 확보 순서:
  1. MARPOL Annex VI Ch.4 (reg 24 기준선 계수 — 일반화물선
     a=107.48·c=0.216 예상, 원전 대조 후 박제) + Phase 감축률표
  2. MEPC EEDI 계산 가이드라인 (MEPC.364(79) 계열 — imo.org 공개,
     MSC.137 확보 성공 경로 재사용): CF 표 (HFO 3.114·MDO 3.206
     gCO₂/g 예상), 75% MCR 관례, Capacity=DWT (일반화물선)
  3. 벙커가 (VLSFO $/t)·산업 전기료 = 공개 시세 대역 C급 —
     #17 수집 후보 등록
- 앵커 사다리:
  1. EEDI 손계산 재현 (수치 대입 항등식)
  2. 기준선 공식값 재현 + DWT 단조 감소 성질
  3. 실선 sanity — 일반화물선 attained 자릿수 대역 (문헌 대역
     정직 표기)
  4. 우리 100m급 화물선 첫 판정 (Phase 3)
  5. 소형 Wh/(kg·km) 손계산 (배터리 Wh·항속·짐)

### §3 보강 — 원전 확보 완료 (2026-08-09, 1단계)

- **MEPC.328(76)** (개정 MARPOL Annex VI, imo.org 공식, 86쪽) →
  `references/MEPC328_76_annexvi.pdf`: p39 Table 2 기준선
  (General cargo **a=107.48·c=0.216** — 예상 적중), p37 Table 1
  감축률 (**이미지 렌더 판독 관례**: General cargo 15,000 DWT+
  Phase 3 (2022-04+) = 30%, 3,000~15,000 = 0→30 선형 보간 각주,
  <3,000 적용 밖)
- **MEPC.364(79)** (2022 EEDI 계산 가이드라인) →
  `references/MEPC364_79_eedi_calc.pdf`: CF 표 (MDO 3.206·HFO
  3.114 tCO₂/t — 예상 적중), 75% MCR 관례
- **MEPC.231(65)** (기준선 산출 방법론·선종 정의) →
  `references/MEPC231_65_reflines.pdf`
- Vref: 설계점 P∝V³ 역산 채택 (프로펠러 법칙 근사 — 정직 표기)

## 4. 단계 분할 (각 단계 성적표 후 계속 — 오너 "변수 없는 한
전 단계" 관례)

1. **EEDI 정본**: 원전 확보 + eedi.py + 손계산·기준선 앵커
2. **운항 경제**: opex.py — 대형 연료비·수송 단가 + 소형 전기
   등가
3. **게이트 통합**: economics_gate + run_pipeline 8중 + 회귀 +
   4축 마감 (23일차)

## 5. 관례·리스크

- TDD 층별, e2e 명시 플래그, main 직커밋
- 리스크: ① 기준선 계수는 선종별 — 우리는 일반화물선(cargo)
  계보만, 타 선종 정직 표기 ② Vref 75% MCR 재역산은 저항 곡선
  근사 (V³ 스케일) — 가이드라인 원문 확인 후 확정 ③ 벙커가 변동
  큼 — 대역 C급 + 민감도 병기 ④ 소형 지표는 규제 아님 (비교용
  성적표) 명시
- 수집 연동: 벙커가·실선 EEDI 공표값 (IMO GISIS 공개 DB) —
  검증 재료 후보
