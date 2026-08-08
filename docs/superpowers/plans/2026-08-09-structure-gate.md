# 구조 강도 3단계 — 판정·수렴·6번째 게이트 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) 문법.

**Goal:** 1단계 하중 + 2단계 단면을 만나게 해 σ=M/Z 판정 → 부족 시 증육 수렴 → `run_pipeline` 6번째 게이트 통합.

**Architecture:** `strength.py` = 순수 판정·수렴 (메쉬 무관, 수치 입력) → `pipeline.py` 대형/소형 분기에서 하중 산출(대형 IACS·소형 준정적) 후 호출. 내항성 관례 재사용: `structure=True` 기본, 기존 시험 필요 시 명시 `structure=False`.

## Global Constraints

- 허용 굽힘 응력 σ = 175·f1 N/mm² (UR S11 p5 원전 — k=1/f1 관계)
- 단위: Z[m³] = M[kN·m]/(σ[N/mm²]·1000)
- 구조 중량 = 병기 전용 (Watson 정본 유지 — 이중 계산 금지, 스펙 §4)
- 소형(<90m)은 IACS 범위 밖 — 준정적 표준파 (a=L/40) 하중, C급 표기
- 기존 시험 360 + strip_loads 4 전부 통과 유지

---

### Task 1: `strength.py` — 판정 + 증육 수렴 루프

**Files:**
- Create: `src/physics/structure/strength.py`
- Test: `tests/test_structure_strength.py`

**Interfaces:**
- Produces: `longitudinal_strength(loa, beam, depth, draft, m_still_knm, m_wave_hog_knm, m_wave_sag_knm, material, spacing_m=None, max_iter=20) -> dict`
  - 키: `passed, z_required_m3, z_deck_m3, z_keel_m3, governing("deck"|"keel"), t_bottom_mm, t_side_mm, t_deck_mm, iterations, sigma_allow_nmm2, structure_mass_per_m_kgm, note`
  - 수렴: 규칙 두께 시작 → Z 부족 시 선저·갑판 +0.5mm씩 (지배 위치 우선) 반복
- 시험 앵커: ① 100m 화물선 규칙 두께 1회 이하 증육 합격 (2단계 비 1.06 실증 재확인) ② 인위 과대 모멘트 → 증육 후 합격, 반복 수 > 0 ③ 두께 단조 증가·수렴 실패 시 passed=False 정직 반환 ④ 알루 소형 — f1 환산 경로

### Task 2: `run_pipeline` 통합 — 6번째 게이트

**Files:**
- Modify: `src/pipeline.py` (대형 분기 + 소형 경로 + passed 합성)
- Test: `tests/test_structure_gate.py`

**요지:**
- 시그니처 `run_pipeline(..., structure=True)`
- 대형: `still_water_curves`(메쉬+lightship 블록) → M_still 미드십, IACS 호깅/새깅 → `longitudinal_strength` → report["structure"], `passed ∧= structure["passed"]`
- 소형: 준정적 표준파 (crest/trough 두 방향) M_wave + M_still, 재료 알루 기본 — 동일 판정. 메쉬 절단 실패 시 정직 스킵 (note 기록, passed 불변 — 데이터 없음 ≠ 불합격)
- 구조 중량 병기: `structure_mass_t` (단면적×L×ρ×1.15 개략) vs Watson 구조 성분 비율 기록
- 기존 시험 영향 최소화: 실패 시험만 `structure=False` 명시 (내항성 관례)

### Task 3: 회귀 + 성적표

- 분할 회귀 (기존 360 + 신규), worklog 3단계 성적표, 커밋·푸시
