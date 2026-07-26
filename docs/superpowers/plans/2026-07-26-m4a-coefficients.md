# M4a 구현 계획 — Fossen 계수 추정 + 중량 분포모델

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 생성된 선형의 Fossen 3자유도(전진·횡이동·선회) 계수 세트를 산출해 리포트에 추가 — M4b 시뮬레이션과 Phase B ROS2 내보내기의 입력.

**Architecture:** ① weights를 분포모델로 승격 (LCG·Izz — 오너 Q4 반영). ② `src/physics/coefficients.py` 신설: 선형 미계수는 Clarke(1983) 회귀 (대형선 외삽 경고 포함), 전진 부가질량은 질량 비율 개략, 전진 감쇠는 자체 저항곡선 수치미분 (spec §2.3). SNAME prime 무차원계 + 차원값 동시 제공.

**Tech Stack:** 기존 스택. 새 의존성 없음.

## Global Constraints

- 기존 계획들의 Global Constraints 승계.
- 부호 규약: 부가질량·감쇠는 **크기(양수)**로 저장, Fossen 방정식 조립 시 부호 적용 — docstring에 명시.
- Clarke 회귀는 대형 상선 통계 — 리포트에 `"extrapolation_warning": true` 명시 (spec §2.3 약속).
- 브랜치: `feat/m4a-coefficients`.

## Task 1: weights 분포모델 (LCG·Izz)

**Files:** Modify `src/physics/weights.py`, `tests/test_weights.py`

**Interfaces:**
- `WeightEstimate`에 추가: `lcg: float` (선체 중앙 기준, +선수), `izz: float` (선회 관성모멘트, kg·m²)
- `estimate_weights(..., loa: float)` 파라미터 추가 (LCG·Izz 계산에 필요)
- 배치 가정 (명명 상수): 구조 LCG=0 (선체 대칭), 적재 LCG=0 (중앙 탑재), 추진 LCG=−0.45·L (선미). 구조 Izz = m_s·(0.25·L)² (선회 회전반경 k≈0.25L 통상값), 점질량은 m·x².
- Wigley는 선수미 대칭 → LCB=0. LCG≠0이면 트림 발생 — `trim_warning` 필드로 보고 (|LCG| > 0.01·L 기준).

Step 1 실패 테스트 (추가):

```python
def test_lcg_izz_computed():
    est = estimate_weights(hull_area_m2=12.0, depth=0.5, payload_kg=100.0,
                           propulsion_mass_kg=4.5, loa=4.0)
    # 추진계만 선미(-0.45L) → LCG는 약간 음수(선미 쪽)
    assert -0.2 < est.lcg < 0.0
    assert est.izz > 0
    # Izz 하한: 점질량 항만으로도 추진계 기여 존재
    assert est.izz >= 4.5 * (0.45 * 4.0) ** 2 * 0.9


def test_trim_warning_flag():
    est = estimate_weights(12.0, 0.5, 100.0, propulsion_mass_kg=40.0, loa=4.0)
    assert est.trim_warning  # 무거운 추진계가 선미에 → 트림 경고
```

Step 3 구현 요지 (완전 코드는 태스크 실행 시 이 시그니처·상수 준수):

```python
LCG_STRUCTURE_OVER_L = 0.0
LCG_PAYLOAD_OVER_L = 0.0
LCG_PROPULSION_OVER_L = -0.45   # 선미
KZZ_OVER_L = 0.25               # 구조 선회 회전반경/L
TRIM_WARN_LCG_OVER_L = 0.01

# estimate_weights 내부:
lcg = (m_s*x_s + m_p*x_p + m_m*x_m) / total
izz = m_s*(KZZ_OVER_L*loa)**2 + m_s*x_s**2 + m_p*x_p**2 + m_m*x_m**2
trim_warning = abs(lcg) > TRIM_WARN_LCG_OVER_L * loa
```

기존 호출부(파이프라인·테스트)는 `loa` 인자 추가만 필요. 커밋: `feat: 중량 분포모델 (LCG·Izz·트림 경고) — 오너 Q4`

## Task 2: coefficients 모듈 (Clarke + 저항 미분)

**Files:** Create `src/physics/coefficients.py`, `tests/test_coefficients.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class CoefficientSet:
    # 무차원 (SNAME prime): 부가질량류 4개 + 감쇠류 4개
    nondim: dict[str, float]   # Yv_dot_p, Yr_dot_p, Nv_dot_p, Nr_dot_p,
                               # Yv_p, Yr_p, Nv_p, Nr_p (전부 크기, 양수)
    # 차원값 [SI]
    xu_dot: float   # 전진 부가질량 [kg]
    yv_dot: float   # 횡 부가질량 [kg]
    nr_dot: float   # 선회 부가관성 [kg·m²]
    yv: float; yr: float; nv: float; nr: float   # 감쇠 (속도 U 기준 선형화)
    xu: float       # 전진 감쇠 = dR/du @ U [N/(m/s)]
    straight_line_stable: bool   # Clarke 직진 안정 판별식
    extrapolation_warning: bool  # 항상 True (대형선 회귀 외삽)

estimate_coefficients(dims, draft, mass, lcg, speed, mesh, n_exp, m_exp) -> CoefficientSet
```

- Clarke(1983) 회귀 (T=평형 흘수, 전부 −부호 생략한 크기):
  - Yv̇' = π(T/L)²·[1 + 0.16·Cb·B/T − 5.1·(B/L)²]
  - Yṙ' = π(T/L)²·[0.67·B/L − 0.0033·(B/T)²]
  - Nv̇' = π(T/L)²·[1.1·B/L − 0.041·B/T]
  - Nṙ' = π(T/L)²·[1/12 + 0.017·Cb·B/T − 0.33·B/L]
  - Yv' = π(T/L)²·[1 + 0.40·Cb·B/T]
  - Yr' = π(T/L)²·[−1/2 + 2.2·B/L − 0.080·B/T]
  - Nv' = π(T/L)²·[1/2 + 2.4·T/L]
  - Nr' = π(T/L)²·[1/4 + 0.039·B/T − 0.56·B/L]
- 차원화 (SNAME): 질량류 ×½ρL³ (Nṙ'는 ×½ρL⁵, Yṙ'·Nv̇'는 ×½ρL⁴), 감쇠류 ×½ρUL² (Yr·Nv ×½ρUL³, Nr ×½ρUL⁴)
- Xu̇ = XU_DOT_MASS_FRACTION(=0.05)·mass (세장체에서 전진 부가질량은 작음 — 개략 상수)
- Xu = (R(1.05U) − R(0.95U)) / (0.1U) — total_resistance 중앙차분
- 직진 안정: C = Nr'·Yv' − Nv'·(Yr' − m') > 0, m' = mass/(½ρL³)

Step 1 실패 테스트 요지:

```python
def test_clarke_pinned_value():
    """수계산 고정: L=4, B=1, T=0.25, Cb=0.5 → Yv' = π(0.0625)²·[1+0.40·0.5·4] = 0.02209"""
    ...  # π·(T/L)²·(1+0.4·0.5·(B/T)) 직접 대조

def test_symmetric_hull_signs_and_positivity():  # 주요 계수 양수, Nv̇'<Yv̇'
def test_surge_damping_matches_finite_difference():  # 저항곡선 수치미분 대조
def test_dimensional_scaling():  # L 2배 → yv_dot 8배 (½ρL³ 스케일)
def test_reports_extrapolation_warning():  # 항상 True
```

커밋: `feat: M4a Fossen 계수 추정 (Clarke 회귀 + 저항 미분, 외삽 경고)`

## Task 3: 파이프라인 통합

**Files:** Modify `src/pipeline.py`, `tests/test_pipeline.py`

- 나선 수렴 후: `coeffs = estimate_coefficients(...)` → report["coefficients"] (asdict) 추가
- weights 호출에 `loa=dims.loa` 반영, report["weights"]에 lcg/izz/trim_warning 자동 포함
- 출력 1줄: `동역학 계수     : Yv̇ {yv_dot:.0f} kg 급 · 직진안정 {O/X} · ⚠ 대형선 회귀 외삽`
- 테스트: report에 coefficients 키 + straight_line_stable bool + 나선 이후에도 전체 통과

커밋 + main 병합 + PROGRESS 갱신: `feat: M4a 통합 — 파이프라인이 동역학 계수까지 산출`

## Self-Review

1. 스펙 §2.3 coefficients (세장체/Clarke/저항 미분/외삽 경고) ✓, 오너 Q4 (분포 모델) ✓, M4b 입력 (질량·Izz·LCG·계수 세트) ✓.
2. 플레이스홀더: Task 1·2에 핵심 수식·상수·테스트 명시, 실행 시 시그니처 준수.
3. 일관성: estimate_coefficients가 받는 mass/lcg는 Task 1 WeightEstimate 산출물, draft는 hydro.draft, 저항 미분은 기존 total_resistance 재사용 ✓.
