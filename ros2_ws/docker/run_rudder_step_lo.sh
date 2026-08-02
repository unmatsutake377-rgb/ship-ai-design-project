#!/usr/bin/env bash
# 저속 러더 스텝: 추력 5N 고정 (u_ss 낮음) + 타각 고정
set -euo pipefail
EXPORT_DIR="$1"; DELTA="$2"; DURATION="${3:-60}"
docker run --rm -v "$EXPORT_DIR:/sim" ship-sim bash -c "
  cd /sim
  gz sim -s -r world.sdf >/sim/gz_server.log 2>&1 &
  SIM_PID=\$!
  sleep 6
  ( for i in \$(seq 1 $DURATION); do
      gz topic -t /model/generated_hull/joint/left_prop_joint/cmd_thrust -m gz.msgs.Double -p 'data: 5.0' >/dev/null 2>&1
      gz topic -t /model/generated_hull/joint/right_prop_joint/cmd_thrust -m gz.msgs.Double -p 'data: 5.0' >/dev/null 2>&1
      gz topic -t /model/generated_hull/joint/rudder_joint/0/cmd_pos -m gz.msgs.Double -p \"data: $DELTA\" >/dev/null 2>&1
      sleep 1
    done ) &
  PUB_PID=\$!
  timeout $DURATION gz topic -e -t /model/generated_hull/odometry --json-output > /sim/step_odom.jsonl || true
  kill \$PUB_PID \$SIM_PID 2>/dev/null || true
"
