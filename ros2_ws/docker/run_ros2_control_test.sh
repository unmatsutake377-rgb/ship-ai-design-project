#!/usr/bin/env bash
# B-3c 완성형: gz 서버 + ros_gz_bridge + rclpy 제어 노드
# 사용: run_ros2_control_test.sh <export_dir 절대경로> [제한 시간 s, 기본 420]
set -euo pipefail

EXPORT_DIR="$1"
TIMEOUT="${2:-420}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp "$SCRIPT_DIR/waypoint_node.py" "$EXPORT_DIR/"

docker run --rm -v "$EXPORT_DIR:/sim" ship-sim bash -c "
  source /opt/ros/jazzy/setup.bash
  cd /sim
  gz sim -s -r world.sdf >/sim/gz_server.log 2>&1 &
  SIM_PID=\$!
  sleep 6
  ros2 run ros_gz_bridge parameter_bridge \
    '/model/generated_hull/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry' \
    '/model/generated_hull/joint/left_prop_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double' \
    '/model/generated_hull/joint/right_prop_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double' \
    >/sim/bridge.log 2>&1 &
  BRIDGE_PID=\$!
  sleep 4
  timeout $TIMEOUT python3 /sim/waypoint_node.py || true
  kill \$BRIDGE_PID \$SIM_PID 2>/dev/null || true
  echo \"기록 \$(( \$(wc -l < /sim/trajectory.csv) - 1 ))행\"
"
