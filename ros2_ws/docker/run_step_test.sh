#!/usr/bin/env bash
# B-3c 격리 실험: 수동 추력 스텝 3단계 — 제어기 없음 (2026-07-30)
# 1단계 0-25s: 좌+20 우+20 (직진)  2단계 25-50s: 좌+20 우0 (편추력)
# 3단계 50-75s: 좌-20 우+20 (차동·후진 포함)
set -euo pipefail
EXPORT_DIR="$1"

docker run --rm -v "$EXPORT_DIR:/sim" ship-sim bash -c '
  cd /sim
  gz sim -s -r world.sdf >/sim/gz_server.log 2>&1 &
  SIM_PID=$!
  sleep 6
  pub() { gz topic -t "/model/generated_hull/joint/${1}_prop_joint/cmd_thrust" -m gz.msgs.Double -p "data: $2" >/dev/null 2>&1; }
  echo "phase,t,x,y,yaw" > /sim/step_test.csv
  sample() {
    P=$(gz model -m generated_hull --pose 2>/dev/null | grep -A2 XYZ | tail -2 | tr -d "[]" | xargs)
    X=$(echo $P | cut -d" " -f1); Y=$(echo $P | cut -d" " -f2); YAW=$(echo $P | cut -d" " -f6)
    echo "$1,$2,$X,$Y,$YAW" >> /sim/step_test.csv
  }
  for phase in 1 2 3; do
    case $phase in
      1) pub left 20; pub right 20;;
      2) pub left 20; pub right 0;;
      3) pub left -- -20; pub right 20;;
    esac
    for t in $(seq 1 10); do sleep 2.5; sample $phase $t; done
  done
  kill $SIM_PID 2>/dev/null || true
  echo "done"
'
