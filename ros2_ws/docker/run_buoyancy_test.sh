#!/usr/bin/env bash
# B-2c 부양 실험: 헤드리스 gz sim에서 정착 자세 측정 (Phase B-2)
# 사용: run_buoyancy_test.sh <export_dir 절대경로> [시뮬 대기 초, 기본 12]
set -euo pipefail

EXPORT_DIR="$1"
WAIT="${2:-12}"

docker run --rm -v "$EXPORT_DIR:/sim" ship-sim bash -c "
  cd /sim
  gz sim -s -r world.sdf >/sim/gz_server.log 2>&1 &
  SIM_PID=\$!
  sleep $WAIT
  echo '--- pose ---'
  gz model -m generated_hull --pose || true
  kill \$SIM_PID 2>/dev/null || true
"
