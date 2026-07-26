# M4b 구현 계획 — 웨이포인트 추종 시뮬레이션

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M4a 계수로 Fossen 3자유도 운동방정식을 적분해, 차동 추력 + LOS 유도 + 선수각 PD 제어로 웨이포인트를 순회하는 시뮬레이션 + 궤적 플롯.

**Architecture:** `src/sim_adapters/python_sim.py` 단일 모듈 (spec §2.4). 파이프라인 리포트(report.json)에서 선박 모델을 재구성 → 시뮬 → 궤적 PNG + 지표. 전진 저항은 자체 저항곡선 보간(8점 사전 샘플, 시뮬 루프에서 Michell 재호출 금지 — 속도).

**운동방정식 (단순화 명시):**
- m_x·u̇ = (T_L + T_R) − R(u) + m_y·v·r        [전진: 저항은 비선형 곡선]
- m_y·v̇ = −m_x·u·r − Yv·v                     [횡: 선형 감쇠]
- I_z·ṙ = (T_R − T_L)·d/2 − Nr·r − Nv·v       [선회]
- ẋ = u·cosψ − v·sinψ, ẏ = u·sinψ + v·cosψ, ψ̇ = r
- m_x = m + Xu̇, m_y = m + Yv̇, I_z = Izz + Nṙ (M4a 크기 규약 → 여기서 부호 조립)
- 생략한 항(Yr·r 교차 감쇠, Yṙ·Nv̇ 교차 부가질량)은 docstring에 명시 — 2차 사이클
- 적분: 전진 오일러, dt=0.05 s (감쇠 지배 시스템 — 안정)

**유도·제어 (spec §2.4):**
- LOS: ψ_d = atan2(y_wp−y, x_wp−x), 수용 반경 = 2·L
- 선수각 PD: δ = Kp·ssa(ψ_d−ψ) − Kd·r (ssa = 최단각 정규화)
- 속도 P: T_common = Kp_u·(u_d−u), 추력기별 T = clip(T_common ± δ, ±T_max)
- 추력기 간격 d = 0.8·B, T_max = 선택 모터 추력 (M4a 리포트에서)

**게인 (명명 상수, 물리 스케일 기반):** Kp_psi = 2·Izz_total/T_char², Kd — 구현에서 무차원 튜닝 상수로 두되 테스트는 게인 값이 아니라 거동(도달·수렴)만 검증.

## Task 1: 동역학 코어 + 물리 불변량 테스트

`VesselModel` dataclass (m_x, m_y, i_z, yv, nv, nr, thrust_max, thruster_sep, resistance_interp) + `vessel_from_report(report, mesh_dir)` (dims에서 메쉬·저항곡선 재구성) + `step(state, t_l, t_r, dt)`.

테스트 (spec §4 불변량):
```python
def test_zero_thrust_stays_at_rest():      # 정지 유지
def test_constant_thrust_terminal_speed(): # R(u_t)=2T 이론 종속도 수렴 (보간 역산 대조)
def test_straight_running_no_drift():      # 등추력 직진: |y| 미소
def test_differential_thrust_turns_correct_direction():  # T_R>T_L → 좌회전(r>0, 반시계)
```

## Task 2: LOS + PD + 웨이포인트 루프

`simulate_waypoints(vessel, waypoints, u_d, dt=0.05, t_max=600) -> SimResult`
(SimResult: 시계열 배열, waypoints_reached, success, duration). 사각 코스(변 10·L) 기본.

테스트:
```python
def test_demo_vessel_completes_square_course():  # 데모 설계로 4개 전부 도달
def test_result_arrays_consistent_length():
```

## Task 3: 플롯 + CLI + 통합

`plot_trajectory(result, waypoints, path)` (matplotlib, 궤적+웨이포인트+시작점) +
CLI `python -m src.sim_adapters.python_sim --report outputs/demo/report.json --out outputs/demo`
→ trajectory.png + sim_result.json. README에 실행법 한 줄 추가.

테스트: CLI smoke (파일 생성 확인).

## Self-Review
- spec §2.4 (차동추력·LOS·PID·포화) ✓, §4 불변량 3종 ✓, M4a 계수 소비 ✓.
- 단순화(교차항 생략, 오일러)는 docstring 명시 — 조용한 근사 금지 원칙 준수.
