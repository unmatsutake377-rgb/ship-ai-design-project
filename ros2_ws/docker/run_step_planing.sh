#!/usr/bin/env bash
# 활주정 게인 캠페인 2차 — 격리 스텝 실험 (2026-08-02)
# 1단계 0-25s: 양쪽 최대 (가속 → gz 종단속도 실측)
# 2단계 25-65s: 추력 0 (관성 항주 — 감속 곡선 = gz 실제 항력의 지문)
# 3단계 65-95s: 좌만 최대 (편추력 — 선회율·속도 이득 실측)
set -euo pipefail
EXPORT_DIR="$1"
THRUST="${2:-157.9}"

docker run --rm -v "$EXPORT_DIR:/sim" ship-sim bash -c '
  cd /sim
  gz sim -s -r world.sdf >/sim/gz_server.log 2>&1 &
  SIM_PID=$!
  sleep 6
  pub() { gz topic -t "/model/generated_hull/joint/${1}_prop_joint/cmd_thrust" -m gz.msgs.Double -p "data: $2" >/dev/null 2>&1; }
  echo "phase,t,simt,x,y,yaw" > /sim/step_planing.csv
  sample() {
    P=$(gz model -m generated_hull --pose 2>/dev/null | grep -A2 XYZ | tail -2 | tr -d "[]" | xargs)
    X=$(echo $P | cut -d" " -f1); Y=$(echo $P | cut -d" " -f2); YAW=$(echo $P | cut -d" " -f6)
    ST=$(gz topic -e -t /stats -n 1 2>/dev/null | grep -A2 sim_time | grep sec | head -1 | grep -o "[0-9]*")
    echo "$1,$2,${ST:-0},$X,$Y,$YAW" >> /sim/step_planing.csv
  }
  T='"$THRUST"'
  # 매초 재발행 (멱등) — 기동 직후 1회성 pub 유실 실측(0.41 m/s 모순)의 수리
  for i in $(seq 1 25); do
    pub left $T; pub right $T
    sleep 1; sample 1 $i
  done
  for i in $(seq 1 40); do
    pub left 0; pub right 0
    sleep 1; sample 2 $i
  done
  for i in $(seq 1 30); do
    pub left $T; pub right 0
    sleep 1; sample 3 $i
  done
  kill $SIM_PID 2>/dev/null || true
  echo "done"
'
