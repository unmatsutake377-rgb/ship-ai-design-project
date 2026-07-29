#!/usr/bin/env python3
"""Gazebo 웨이포인트 제어기 (B-3c) — 컨테이너 안에서 실행.

python_sim과 동일 제어 법칙 (control.yaml): lookahead LOS + 선수각 PD
(고유값 게인) + 속도 P + 코너 감속 + 전진 전용 차동 배분.

gz 실측 규약 (2026-07-30 스텝 실험):
- 왼쪽 추력 강함 = 반시계(+요) → +모멘트는 tl > tr
- 음수(후진) 추력 = 플러그인 침묵 고장 → 전진 전용 배분 필수
- gz topic echo 스트림이 ~50s에 조용히 EOF → 자동 재접속 필수
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


class Controller:
    def __init__(self, cfg):
        self.cfg = cfg
        self.waypoints = [tuple(w) for w in cfg["waypoints"]]
        self.wp_index = 0
        self.prev_wp = (0.0, 0.0)
        self.tick = 0
        self.log = open("/sim/trajectory.csv", "w")
        self.log.write("x,y,yaw,u,tl,tr,wp\n")

    def step(self, msg) -> bool:
        """오도메트리 1건 처리. 완주 시 True."""
        cfg = self.cfg
        self.tick += 1
        if self.tick % 3:  # 10Hz → ~3.3Hz 제어
            return False

        pos = msg.get("pose", {}).get("position", {})
        x, y = pos.get("x", 0.0), pos.get("y", 0.0)
        psi = yaw_from_quat(msg.get("pose", {}).get("orientation", {}))
        tw = msg.get("twist", {})
        u = tw.get("linear", {}).get("x", 0.0)
        r = tw.get("angular", {}).get("z", 0.0)

        wx, wy = self.waypoints[self.wp_index]
        if math.hypot(wx - x, wy - y) < cfg["accept_radius"]:
            self.prev_wp = (wx, wy)
            self.wp_index += 1
            if self.wp_index >= len(self.waypoints):
                print(f"완주 — 웨이포인트 {len(self.waypoints)}개 전부 도달")
                return True
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
                   min(cfg["thrust_max"],
                       moment / cfg["thruster_separation"]))
        headroom = cfg["thrust_max"] - abs(diff)
        common = max(0.0, min(headroom, cfg["kp_u"] * (u_cmd - u)))

        tl, tr = common + diff, common - diff  # 왼쪽 강함 = 반시계
        if min(tl, tr) < 0:  # 후진 금지 — 모멘트 보존 상향
            shift = -min(tl, tr)
            tl += shift
            tr += shift
        tl = min(tl, cfg["thrust_max"])
        tr = min(tr, cfg["thrust_max"])

        pub_thrust("left", tl)
        pub_thrust("right", tr)
        self.log.write(f"{x:.3f},{y:.3f},{psi:.3f},{u:.3f},"
                       f"{tl:.1f},{tr:.1f},{self.wp_index}\n")
        self.log.flush()
        return False


def main():
    cfg = yaml.safe_load(open("/sim/control.yaml"))
    if not cfg["waypoints"]:
        print("웨이포인트 없음", file=sys.stderr)
        return 1
    ctl = Controller(cfg)
    done = False
    while not done:
        stream = subprocess.Popen(
            ["gz", "topic", "-e", "-t", "/model/generated_hull/odometry",
             "--json-output"],
            stdout=subprocess.PIPE, text=True,
        )
        got_any = False
        for line in stream.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            got_any = True
            if ctl.step(msg):
                done = True
                break
        stream.terminate()
        if not done and not got_any:
            break  # 스트림이 즉시 빈 채 끝나면 서버 죽은 것 — 탈출

    pub_thrust("left", 0.0)
    pub_thrust("right", 0.0)
    ctl.log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
