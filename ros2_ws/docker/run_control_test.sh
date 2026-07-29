#!/usr/bin/env bash
# B-3c 제어 실험: gz 서버 + 우리 제어기(LOS+PD) 동시 구동
# 사용: run_control_test.sh <export_dir 절대경로> [제한 시간 s, 기본 240]
set -euo pipefail

EXPORT_DIR="$1"
TIMEOUT="${2:-240}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp "$SCRIPT_DIR/waypoint_controller.py" "$EXPORT_DIR/"

docker run --rm -v "$EXPORT_DIR:/sim" ship-sim bash -c "
  cd /sim
  gz sim -s -r world.sdf >/sim/gz_server.log 2>&1 &
  SIM_PID=\$!
  sleep 6
  timeout $TIMEOUT python3 /sim/waypoint_controller.py || true
  kill \$SIM_PID 2>/dev/null || true
  echo \"기록 \$(( \$(wc -l < /sim/trajectory.csv) - 1 ))행\"
"
