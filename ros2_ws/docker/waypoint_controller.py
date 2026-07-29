#!/usr/bin/env python3
"""Gazebo 웨이포인트 제어기 (B-3c) — 컨테이너 안에서 실행.

python_sim과 동일한 제어 법칙 (control.yaml로 전달받음):
lookahead LOS + 선수각 PD(고유값 게인) + 속도 P + 코너 감속 + 선회 우선 배분.
입력: gz 오도메트리 토픽 스트림 (JSON 라인)
출력: 좌/우 추력기 cmd_thrust + trajectory.csv (자체 기록)
"""
import json
import math
import subprocess
import sys

import yaml


def ssa(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_from_quat(q):
    x, y, z, w = q.get("x", 0), q.get("y", 0), q.get("z", 0), q.get("w", 1)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def pub_thrust(side, value):
    subprocess.run(
        ["gz", "topic", "-t",
         f"/model/generated_hull/joint/{side}_prop_joint/cmd_thrust",
         "-m", "gz.msgs.Double", "-p", f"data: {value:.3f}"],
        capture_output=True, timeout=5,
    )


def main():
    cfg = yaml.safe_load(open("/sim/control.yaml"))
    waypoints = [tuple(w) for w in cfg["waypoints"]]
    if not waypoints:
        print("웨이포인트 없음", file=sys.stderr)
        return 1

    stream = subprocess.Popen(
        ["gz", "topic", "-e", "-t", "/model/generated_hull/odometry",
         "--json-output"],
        stdout=subprocess.PIPE, text=True,
    )

    wp_index = 0
    prev_wp = (0.0, 0.0)
    log = open("/sim/trajectory.csv", "w")
    log.write("x,y,yaw,u,tl,tr,wp\n")
    tick = 0
    control_every = 3  # 오도메트리 10Hz → 제어 ~3.3Hz

    for line in stream.stdout:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        tick += 1
        if tick % control_every:
            continue

        pos = msg.get("pose", {}).get("position", {})
        x, y = pos.get("x", 0.0), pos.get("y", 0.0)
        psi = yaw_from_quat(msg.get("pose", {}).get("orientation", {}))
        tw = msg.get("twist", {})
        u = tw.get("linear", {}).get("x", 0.0)
        r = tw.get("angular", {}).get("z", 0.0)

        wx, wy = waypoints[wp_index]
        if math.hypot(wx - x, wy - y) < cfg["accept_radius"]:
            prev_wp = (wx, wy)
            wp_index += 1
            if wp_index >= len(waypoints):
                print(f"완주 — 웨이포인트 {len(waypoints)}개 전부 도달")
                break
            wx, wy = waypoints[wp_index]

        # lookahead LOS (python_sim.simulate_waypoints와 동일)
        px, py = prev_wp
        alpha = math.atan2(wy - py, wx - px)
        e_ct = -(x - px) * math.sin(alpha) + (y - py) * math.cos(alpha)
        psi_d = alpha + math.atan2(-e_ct, cfg["lookahead"])

        dist = math.hypot(wx - x, wy - y)
        frac = max(cfg["slowdown_min_fraction"],
                   min(1.0, dist / cfg["slowdown_radius"]))
        u_cmd = cfg["u_desired"] * frac

        moment = cfg["kp_psi"] * ssa(psi_d - psi) - cfg["kd_psi"] * r
        # 부호 반전: gz Thruster의 차동→요 모멘트 방향이 우리 규약과 반대
        # (07-29 실측: 반시계 명령에 시계 회전 — 플러그인 방향 규약)
        diff = -max(-cfg["thrust_max"],
                    min(cfg["thrust_max"],
                        moment / cfg["thruster_separation"]))
        headroom = cfg["thrust_max"] - abs(diff)
        common = max(-headroom, min(headroom, cfg["kp_u"] * (u_cmd - u)))

        tl, tr = common - diff, common + diff
        pub_thrust("left", tl)
        pub_thrust("right", tr)
        log.write(f"{x:.3f},{y:.3f},{psi:.3f},{u:.3f},"
                  f"{tl:.1f},{tr:.1f},{wp_index}\n")
        log.flush()

    pub_thrust("left", 0.0)
    pub_thrust("right", 0.0)
    stream.terminate()
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
