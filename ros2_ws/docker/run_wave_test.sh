#!/usr/bin/env bash
# 파랑 응답 실험 (내항 교차 검증): 규칙파에 배를 띄워 heave 시계열 채집
# 사용: run_wave_test.sh <wave_exp 디렉토리> [채집 초, 기본 45]
# 전제: world.sdf + hull.stl (생성: 내항 스펙 §10 실험 절차), ship-sim-waves 이미지
set -euo pipefail
DIR="$1"; WAIT="${2:-45}"
docker run --rm -v "$DIR:/sim" ship-sim-waves bash -c "
  cd /sim
  gz sim -s -r world.sdf > gz.log 2>&1 &
  P=\$!
  sleep 6
  timeout $WAIT gz topic -e -t /world/wave_exp/dynamic_pose/info > poses.log 2>/dev/null || true
  kill \$P 2>/dev/null || true
  echo 프레임: \$(grep -c 'name: \"ship\"' poses.log)
"
