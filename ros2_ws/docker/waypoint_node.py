#!/usr/bin/env python3
"""ROS2 웨이포인트 제어 노드 (B-3c 완성형) — rclpy.

이력: gz CLI 스폰 방식은 호출당 ~0.5s 지연 → 10Hz 스트림 낙오 →
낡은 상태 제어(진동)+파이프 붕괴(72s 조기 종료) 실측 (07-30).
→ ros_gz_bridge + 영속 pub/sub로 이식 (원래 B-3c 설계).

제어 법칙은 control.yaml (python_sim과 동일: LOS+PD 고유값 게인,
전진 전용 차동 — gz 실측 규약 반영: 왼쪽 강함=반시계, 후진 금지).
"""
import math

import rclpy
import yaml
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64


def ssa(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class WaypointNode(Node):
    def __init__(self):
        super().__init__("waypoint_controller")
        self.cfg = yaml.safe_load(open("/sim/control.yaml"))
        self.waypoints = [tuple(w) for w in self.cfg["waypoints"]]
        self.wp_index = 0
        self.prev_wp = (0.0, 0.0)
        self.state = None
        self.done = False

        self.pub_l = self.create_publisher(
            Float64, "/model/generated_hull/joint/left_prop_joint/cmd_thrust", 10)
        self.pub_r = self.create_publisher(
            Float64, "/model/generated_hull/joint/right_prop_joint/cmd_thrust", 10)
        self.create_subscription(
            Odometry, "/model/generated_hull/odometry", self.on_odom, 10)
        self.create_timer(0.2, self.control)  # 5 Hz

        self.log = open("/sim/trajectory.csv", "w")
        self.log.write("x,y,yaw,u,tl,tr,wp\n")

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        self.state = (p.x, p.y, yaw,
                      msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def thrust(self, tl, tr):
        self.pub_l.publish(Float64(data=float(tl)))
        self.pub_r.publish(Float64(data=float(tr)))

    def control(self):
        if self.state is None or self.done:
            return
        cfg = self.cfg
        x, y, psi, u, r = self.state

        wx, wy = self.waypoints[self.wp_index]
        # 도달 판정 2중 (2026-08-02 활주 검증이 잡은 잠복 버그):
        # 수용 반경 스침 + LOS 경로선 무한 연장 → 직진 폭주(1,265m 실측)
        # — 종점 통과(진행 거리 > 구간 길이)도 도달로 취급
        px0, py0 = self.prev_wp
        alpha0 = math.atan2(wy - py0, wx - px0)
        seg_len = math.hypot(wx - px0, wy - py0)
        s_along = ((x - px0) * math.cos(alpha0)
                   + (y - py0) * math.sin(alpha0))
        if (math.hypot(wx - x, wy - y) < cfg["accept_radius"]
                or s_along > seg_len):
            self.prev_wp = (wx, wy)
            self.wp_index += 1
            if self.wp_index >= len(self.waypoints):
                self.get_logger().info("완주 — 웨이포인트 전부 도달")
                self.thrust(0.0, 0.0)
                self.done = True
                return
            wx, wy = self.waypoints[self.wp_index]

        px, py = self.prev_wp
        alpha = math.atan2(wy - py, wx - px)
        e_ct = -(x - px) * math.sin(alpha) + (y - py) * math.cos(alpha)
        psi_d = alpha + math.atan2(-e_ct, cfg["lookahead"])

        dist = math.hypot(wx - x, wy - y)
        frac = max(cfg["slowdown_min_fraction"],
                   min(1.0, dist / cfg["slowdown_radius"]))
        u_cmd = cfg["u_desired"] * frac

        moment = cfg["kp_psi"] * ssa(psi_d - psi) - cfg["kd_psi"] * r
        diff = max(-cfg["thrust_max"],
                   min(cfg["thrust_max"], moment / cfg["thruster_separation"]))
        headroom = cfg["thrust_max"] - abs(diff)
        common = max(0.0, min(headroom, cfg["kp_u"] * (u_cmd - u)))

        tl, tr = common + diff, common - diff  # 왼쪽 강함 = 반시계 (실측)
        if min(tl, tr) < 0:  # 후진 금지 (실측: 음수 = 플러그인 고장)
            shift = -min(tl, tr)
            tl += shift
            tr += shift
        tl = min(tl, cfg["thrust_max"])
        tr = min(tr, cfg["thrust_max"])

        self.thrust(tl, tr)
        self.log.write(f"{x:.3f},{y:.3f},{psi:.3f},{u:.3f},"
                       f"{tl:.1f},{tr:.1f},{self.wp_index}\n")
        self.log.flush()


def main():
    rclpy.init()
    node = WaypointNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
    finally:
        node.log.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
