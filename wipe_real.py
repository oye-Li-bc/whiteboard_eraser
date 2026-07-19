#!/usr/bin/env python3
# coding=utf-8
"""真机 wipe 节点:把夹爪夹着的白板擦移到错字处,沿白板平面弓字形擦拭。

安全设计:
  * 默认【只接近、不接触】—— 擦头尖停在白板前方 standoff(默认6cm),用于验证姿态/落点/可达。
  * 确认无误后加 --press 才真正接触并擦拭。
  * 全程慢速;IK 失败(够不着)直接跳过并提示。

关键几何:
  * 擦头尖相对 Link6 偏移 TCP(默认沿 Link6 +z 伸出 0.18m,横向居中)。
  * 白板平面 (p0, n) 来自 /wipe_plane;n 朝相机/基座侧。擦头工具轴(+z) 对齐 -n(压入板面)。
  * Link6 目标 = 擦头尖目标 + TCP * n_hat (Link6 在尖后方,靠基座侧)。

订阅: /wipe_targets(PoseArray,base) /wipe_target_sizes /wipe_plane /joint_states
依赖真机 MoveIt: /compute_ik /compute_cartesian_path /rm_group_controller/follow_joint_trajectory
"""
import argparse
import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from std_msgs.msg import Float32MultiArray
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState, PositionIKRequest
from rm_ros_interfaces.msg import Movej
from scipy.spatial.transform import Rotation as Rot


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tcp", type=float, default=0.18, help="擦头尖沿Link6 +z的伸出长度(m)")
    p.add_argument("--standoff", type=float, default=0.06, help="接近位:擦头尖距板面(m)")
    p.add_argument("--press", action="store_true", help="真正接触并擦拭(否则只接近验证)")
    p.add_argument("--press-depth", type=float, default=0.0, help="压入板面深度(m,默认0=只贴到板面不压)")
    p.add_argument("--row-step", type=float, default=0.02, help="弓字形行距(m)")
    p.add_argument("--wipe-span", type=float, default=0.03, help="水平来回幅度(m,默认3cm)")
    p.add_argument("--wipe-h", type=float, default=0.0, help="竖直覆盖高(m,0=自动:用检测字高)")
    p.add_argument("--min-wipe", type=float, default=0.045, help="自动模式下竖直覆盖下限(m),防退化尺寸")
    p.add_argument("--passes", type=int, default=3, help="来回擦拭遍数(每遍反向,连续来回)")
    p.add_argument("--base-frame", default="base_link")
    p.add_argument("--ee-frame", default="Link6")
    p.add_argument("--group", default="rm_group")
    p.add_argument("--joints", nargs="+",
                   default=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"])
    p.add_argument("--speed", type=int, default=20, help="接近/移动关节速度百分比(1-100,越小越慢越稳)")
    p.add_argument("--wipe-speed", type=int, default=15, help="擦拭逐点关节速度百分比")
    known, _ = p.parse_known_args()
    return known


ROLLS = [0.0, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2, np.pi]  # 绕板面法向的候选滚转


def orientation_from_normal(n_hat, roll=0.0):
    """由板面法向(朝基座侧)构造擦头姿态:工具+z = -n(压入板面),+x取板面内竖直方向。
    roll: 绕工具z轴(即法向)的附加自转,擦拭不在乎,用来选行程最小的手腕姿态。
    返回 (R_base_tool 3x3, quat xyzw)。"""
    z = -n_hat / np.linalg.norm(n_hat)          # 工具轴指向板面
    up = np.array([0.0, 0.0, 1.0])
    x = up - np.dot(up, z) * z                  # 世界竖直投影到板面
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0, 0], z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    if abs(roll) > 1e-9:                         # 绕工具z自转
        cs, sn = np.cos(roll), np.sin(roll)
        R = R @ np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1.0]])
    return R, Rot.from_matrix(R).as_quat()


def back_forth_uv(span, height, row_step, margin=0.003):
    """水平来回擦拭路点:沿水平方向 ±span/2 来回,沿竖直方向按 row_step 铺满 height。
    返回 [(horiz, vert), ...],horiz 沿板面水平(y_col),vert 沿板面竖直(x_col)。"""
    hs = span / 2.0
    hh = height / 2.0 + margin
    n_rows = max(1, int(np.ceil(2 * hh / max(row_step, 1e-3))) + 1)
    rows = np.linspace(hh, -hh, n_rows)          # 从上到下逐行
    pts = []
    for i, v in enumerate(rows):
        pts += [(-hs, v), (hs, v)] if i % 2 == 0 else [(hs, v), (-hs, v)]  # 每行左右来回
    return pts


class WipeReal(Node):
    def __init__(self, args):
        super().__init__("wipe_real")
        self.args = args
        self.latest_js = None
        self.targets = self.sizes = self.plane = None
        self._started = False

        self.create_subscription(JointState, "/joint_states", self._cb_js, 10)
        latched = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseArray, "/wipe_targets", self._cb_targets, latched)
        self.create_subscription(Float32MultiArray, "/wipe_target_sizes", self._cb_sizes, latched)
        self.create_subscription(Float32MultiArray, "/wipe_plane", self._cb_plane, latched)

        self.ik_cli = self.create_client(GetPositionIK, "/compute_ik")
        self.movej_pub = self.create_publisher(Movej, "/rm_driver/movej_cmd", 10)
        mode = "接触擦拭(--press)" if args.press else "只接近不接触(安全验证)"
        self.get_logger().info(f"wipe_real 启动,模式={mode},TCP={args.tcp}m standoff={args.standoff}m。等待 /wipe_targets+/wipe_plane ...")

    def _cb_js(self, m): self.latest_js = m
    def _cb_targets(self, m): self.targets = m; self._maybe()
    def _cb_sizes(self, m): self.sizes = list(m.data); self._maybe()
    def _cb_plane(self, m): self.plane = list(m.data); self._maybe()

    def _maybe(self):
        if self._started or self.targets is None or self.plane is None:
            return
        if self.sizes is None:
            self.sizes = []
        self._started = True
        threading.Thread(target=self._run, daemon=True).start()

    # ---- MoveIt 管线(复用仿真结构) ----
    def _wait(self, fut, timeout=30.0):
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                return None
            time.sleep(0.02)
        return fut.result()

    def _robot_state(self):
        rs = RobotState()
        if self.latest_js is not None:
            rs.joint_state = self.latest_js
        return rs

    def _make_pose(self, pos, quat):
        p = Pose()
        p.position.x, p.position.y, p.position.z = [float(v) for v in pos]
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = [float(v) for v in quat]
        return p

    def _compute_ik(self, pos, quat):
        req = GetPositionIK.Request()
        ikr = PositionIKRequest()
        ikr.group_name = self.args.group
        ikr.ik_link_name = self.args.ee_frame
        ikr.robot_state = self._robot_state()
        ikr.avoid_collisions = False
        ps = PoseStamped(); ps.header.frame_id = self.args.base_frame
        ps.pose = self._make_pose(pos, quat)
        ikr.pose_stamped = ps
        ikr.timeout.sec = 2
        req.ik_request = ikr
        res = self._wait(self.ik_cli.call_async(req))
        if res is None or res.error_code.val != 1:
            return None
        n2p = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        try:
            return [n2p[j] for j in self.args.joints]
        except KeyError:
            return None

    def _cur_joints(self):
        if self.latest_js is None:
            return None
        cur = dict(zip(self.latest_js.name, self.latest_js.position))
        try:
            return np.array([cur[j] for j in self.args.joints])
        except KeyError:
            return None

    def _unwrap(self, target):
        """把目标关节角按'离当前角最近的圈数'展开(±2π),返回(展开后, 最大行程rad)。"""
        c = self._cur_joints()
        t = np.array([float(x) for x in target], dtype=float)
        if c is None:
            return t, None
        for i in range(len(t)):
            while t[i] - c[i] > np.pi:
                t[i] -= 2 * np.pi
            while t[i] - c[i] < -np.pi:
                t[i] += 2 * np.pi
            if abs(t[i]) > 2 * np.pi + 0.05:      # 展开越界了就退回原值,交给行程上限拦
                t[i] = float(target[i])
        return t, float(np.max(np.abs(t - c)))

    def _movej(self, joints, speed, timeout=40.0, max_travel=2.6):
        """发原生 movej 到关节角(弧度)。先按当前角就近展开圈数,行程超上限则拒绝(防绕线/大甩)。
        监测 joint_states 收敛判断到位。可靠送达。"""
        target, travel = self._unwrap(joints)
        if travel is not None and travel > max_travel:
            self.get_logger().warn(
                f"  关节行程过大({np.degrees(travel):.0f}°>{np.degrees(max_travel):.0f}°),"
                f"拒绝执行以防绕线/大甩。请先把手臂手动移到更接近目标的起始姿态(手腕大致回中)。")
            return False
        joints = [float(x) for x in target]
        t0 = time.time()
        while self.movej_pub.get_subscription_count() < 1:   # 等订阅匹配,否则丢消息
            if time.time() - t0 > 5.0:
                self.get_logger().error("  /rm_driver/movej_cmd 没有订阅者(rm_driver 没在跑?)")
                return False
            time.sleep(0.05)
        msg = Movej()
        msg.joint = joints
        msg.speed = int(max(1, min(100, speed)))
        msg.block = False
        msg.trajectory_connect = 0
        msg.dof = len(joints)
        for _ in range(3):                 # 多发几次确保送达
            self.movej_pub.publish(msg)
            time.sleep(0.08)
        return self._wait_reach(joints, timeout)

    def _best_ik(self, link6_pos, n_hat, rolls=ROLLS):
        """在法向上试多个滚转角,挑关节行程最小的 IK 解。返回 (展开后关节角, 选用quat, 行程) 或 (None,None,None)。"""
        best, best_q, best_tr = None, None, 1e9
        for roll in rolls:
            _, quat = orientation_from_normal(n_hat, roll)
            ik = self._compute_ik(link6_pos, quat)
            if ik is None:
                continue
            uw, tr = self._unwrap(ik)
            tr = 0.0 if tr is None else tr
            if tr < best_tr:
                best, best_q, best_tr = uw, quat, tr
        if best is None:
            return None, None, None
        return best, best_q, best_tr

    def _wait_reach(self, target, timeout, tol=0.02, settle=0.4):
        """等 joint_states 收敛到 target(最大关节残差<tol),到位后停 settle。"""
        target = np.array(target)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.latest_js is not None:
                cur = dict(zip(self.latest_js.name, self.latest_js.position))
                try:
                    c = np.array([cur[j] for j in self.args.joints])
                except KeyError:
                    time.sleep(0.05); continue
                if np.max(np.abs(c - target)) < tol:
                    time.sleep(settle); return True
            time.sleep(0.05)
        self.get_logger().warn("  等待到位超时(可能没动/残差未收敛,检查急停或速度)")
        return False

    # ---- 主流程 ----
    def _run(self):
        time.sleep(2.0)
        self.ik_cli.wait_for_service()
        p0 = np.array(self.plane[:3]); n = np.array(self.plane[3:6])
        n_hat = n / np.linalg.norm(n)
        a = self.args
        n_t = len(self.targets.poses)
        self.get_logger().info(f"共 {n_t} 个错字目标, 平面法向={n_hat.round(3)}")

        for idx, pose in enumerate(self.targets.poses):
            P = np.array([pose.position.x, pose.position.y, pose.position.z])  # 擦头尖目标(板面)
            det_h = self.sizes[2 * idx + 1] if 2 * idx + 1 < len(self.sizes) else 0.0
            span = a.wipe_span                                  # 水平来回幅度(默认5cm)
            height = a.wipe_h if a.wipe_h > 0 else max(det_h, a.min_wipe)  # 竖直覆盖(字高)

            link6_at = lambda tip: tip + a.tcp * n_hat          # 擦头尖tip -> Link6位置
            prep_tip = P + a.standoff * n_hat                   # 接近位(板前standoff)
            # 试多个滚转角,挑关节行程最小的接近解(避免大转腕/绕线)
            ik, quat, travel = self._best_ik(link6_at(prep_tip), n_hat)
            if ik is None:
                self.get_logger().warn(f"[{idx+1}/{n_t}] 接近位 IK 失败(可能够不着),跳过。目标={P.round(3)}")
                continue
            R = Rot.from_quat(quat).as_matrix()                 # 选定姿态,擦拭沿用同一滚转
            x_col, y_col = R[:, 0], R[:, 1]
            self.get_logger().info(f"[{idx+1}/{n_t}] 移动到接近位(距板面{a.standoff*100:.0f}cm),"
                                   f"预计最大关节行程{np.degrees(travel):.0f}°,movej 慢速...")
            if not self._movej(ik, a.speed):
                self.get_logger().warn("  未执行/未到位(可能行程超限被拦),跳过"); continue

            if not a.press:
                self.get_logger().info("  已到接近位,未接触(安全模式)。看擦头尖是否正对错字、方向垂直板面。")
                continue

            # 接触 + 水平来回擦拭:逐点 IK -> movej(沿用同一滚转姿态,当前角作种子)
            contact = P + a.press_depth * (-n_hat)              # 压入板面
            base_uv = back_forth_uv(span, height, a.row_step)
            # 来回多遍:偶数遍正向、奇数遍反向,首尾相接连续来回
            wps_uv = []
            for pi in range(max(1, a.passes)):
                wps_uv += base_uv if pi % 2 == 0 else list(reversed(base_uv))
            self.get_logger().info(f"  接触擦拭:水平来回{span*100:.0f}cm x 覆盖高{height*100:.1f}cm, "
                                   f"压深{a.press_depth*1000:.0f}mm, {a.passes}遍来回, 共{len(wps_uv)}点...")
            done = 0
            for (h, v) in wps_uv:
                tip = contact + h * y_col + v * x_col           # h沿水平(y_col), v沿竖直(x_col)
                ikw = self._compute_ik(link6_at(tip), quat)
                if ikw is None:
                    continue
                if self._movej(ikw, a.wipe_speed):
                    done += 1
            ik_up = self._compute_ik(link6_at(prep_tip), quat)   # 抬离到接近位
            if ik_up is not None:
                self._movej(ik_up, a.speed)
            self.get_logger().info(f"  擦拭完成({done}/{len(wps_uv)} 点已走,{a.passes}遍),已抬离。")
        self.get_logger().info("全部目标处理完毕。")


def main():
    args = parse_args()
    rclpy.init()
    node = WipeReal(args)
    ex = MultiThreadedExecutor(); ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
