#!/usr/bin/env bash
# OpenFOAM 케이스 실행 (Docker) — CFD 훅 2단계.
# 사용법: cfd/docker/run_case.sh <케이스폴더> [simpleFoam|interFoam]
set -euo pipefail
CASE_DIR=$(cd "$1" && pwd)
SOLVER=${2:-simpleFoam}
IMG=opencfd/openfoam-default:2406

# 공식 이미지 엔트리포인트가 OpenFOAM 환경을 자동 소싱함.
# (안 되면 플랜 B: bash -c "source /usr/lib/openfoam/openfoam2406/etc/bashrc && ...")
docker run --rm -v "$CASE_DIR":/case -w /case "$IMG" bash -lc '
  set -e
  blockMesh
  snappyHexMesh -overwrite
  if [ -f system/setFieldsDict ]; then setFields; fi
  '"$SOLVER"'
' 2>&1 | tee "$CASE_DIR/run.log"
echo "완료 — 라벨 수확: python -m src.cfd.hook ... --parse-only"
