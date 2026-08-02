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


def _pub_async(topic, value):
    """비동기 발행 — gz topic 1회 호출이 ~0.5s 블로킹이라 run()으로
    틱당 2~3번 부르면 제어 주기가 초 단위로 늘어져 오도메트리 백로그
    (옛 상태 기반 조타 → 뱅뱅) 발생 (2026-08-03 러더 4단계 검시).
    Popen 발사 후 잊기 — 대기 없음."""
    subprocess.Popen(
        ["gz", "topic", "-t", topic, "-m", "gz.msgs.Double",
         "-p", f"data: {value:.4f}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def pub_thrust(side, value):
    _pub_async(f"/model/generated_hull/joint/{side}_prop_joint/cmd_thrust",
               value)


def pub_rudder(value):
    """타각 명령 [rad] — 매 틱 재발행 (gz 1회성 pub 유실 관례)."""
    _pub_async("/model/generated_hull/joint/rudder_joint/0/cmd_pos", value)


class Controller:
    def __init__(self, cfg):
        self.cfg = cfg
        self.waypoints = [tuple(w) for w in cfg["waypoints"]]
        self.wp_index = 0
        self.prev_wp = (0.0, 0.0)
        self.tick = 0
        self.log = open("/sim/trajectory.csv", "w")
        self.log.write("t,x,y,yaw,u,tl,tr,delta,wp\n")

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
        # 도달 판정 2중 (2026-08-03 이식): ① 수용 반경 ② 종점 통과 —
        # python_sim·waypoint_node엔 있던 수리가 이 파일엔 누락돼
        # 부표 옆을 스친 배가 영원히 궤도를 도는 회귀 실측 (러더 4단계
        # 기준선 검시에서 발굴)
        px0, py0 = self.prev_wp
        alpha0 = math.atan2(wy - py0, wx - px0)
        seg_len = math.hypot(wx - px0, wy - py0)
        s_along = ((x - px0) * math.cos(alpha0)
                   + (y - py0) * math.sin(alpha0))
        if math.hypot(wx - x, wy - y) < cfg["accept_radius"] \
                or s_along > seg_len:
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
        delta = 0.0
        if cfg.get("steering") == "rudder2":
            # 러더 우선 (스펙 4단계, python 1단계와 같은 법칙):
            # 현재 유속의 각도당 모멘트로 필요 타각 역산, 잔여만 차동.
            rd = cfg["rudder"]
            n_per_rad = (0.5 * 1025.0 * max(u, 0.05) ** 2 * rd["area"]
                         * rd["cla"] * abs(rd["x_pos"]))
            delta = rd["sign"] * max(-rd["max_rad"],
                                     min(rd["max_rad"],
                                         moment / max(n_per_rad, 1e-9)))
            d_eff = max(-rd["stall_rad"], min(rd["stall_rad"], delta))
            n_rudder = n_per_rad * d_eff * rd["sign"]
            moment_res = moment - n_rudder
            # 속도 예산 안 차동 + 조타 예비 20% (gz 실측 2026-08-03:
            # 전속 가속 중 common=천장 → d_max=0 → 저속(러더 무력)
            # 구간에서 조타 실종, 초반 표류. 스로틀 상한을 80%로 눌러
            # 차동 여유를 상시 확보 — 실선 "전타 시 감속" 관행)
            cap = 0.8 * cfg["thrust_max"]
            common = max(0.0, min(cap, cfg["kp_u"] * (u_cmd - u)))
            d_max = min(common + 1e-9, cfg["thrust_max"] - common)
            diff = max(-d_max, min(d_max,
                                   moment_res / cfg["thruster_separation"]))
            tl, tr = common + diff, common - diff
            pub_rudder(delta)
        else:
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
        st = msg.get("header", {}).get("stamp", {})
        t_sim = float(st.get("sec", 0)) + float(st.get("nsec", 0)) * 1e-9
        self.log.write(f"{t_sim:.1f},{x:.3f},{y:.3f},{psi:.3f},{u:.3f},"
                       f"{tl:.1f},{tr:.1f},{delta:.3f},{self.wp_index}\n")
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
