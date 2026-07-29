#!/usr/bin/env bash
# B-3b 궤적 실험: 헤드리스 gz sim 주행 중 자세를 주기 샘플링 → CSV
# 사용: run_trajectory_test.sh <export_dir 절대경로> [샘플 수, 기본 240] [간격 s, 기본 0.5]
set -euo pipefail

EXPORT_DIR="$1"
SAMPLES="${2:-240}"
INTERVAL="${3:-0.5}"

docker run --rm -v "$EXPORT_DIR:/sim" ship-sim bash -c "
  cd /sim
  gz sim -s -r world.sdf >/sim/gz_server.log 2>&1 &
  SIM_PID=\$!
  sleep 5
  echo 't_wall,x,y,z,roll,pitch,yaw' > /sim/trajectory.csv
  for i in \$(seq 1 $SAMPLES); do
    POSE=\$(gz model -m generated_hull --pose 2>/dev/null \
      | grep -A2 'XYZ' | tail -2 | tr -d '[]' | xargs)
    if [ -n \"\$POSE\" ]; then
      echo \"\$i,\$(echo \$POSE | tr ' ' ',')\" >> /sim/trajectory.csv
    fi
    sleep $INTERVAL
  done
  kill \$SIM_PID 2>/dev/null || true
  echo \"샘플 \$(( \$(wc -l < /sim/trajectory.csv) - 1 ))개 기록\"
"
