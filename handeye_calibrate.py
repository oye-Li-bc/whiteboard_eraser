#!/usr/bin/env python3
# coding=utf-8
"""ROS 2 眼在手上(eye-in-hand)手眼标定工具。

用现成的 ROS 数据源标定 D435 相对机械臂末端(Link6)的外参:
  - 相机图像 + 内参:  /camera/camera/color/image_raw, /camera/camera/color/camera_info
  - 末端位姿:          TF  base_link -> Link6 (由 rm_65_bringup 的 robot_state_publisher 发布)

流程:
  1) 保证 ROS realsense 驱动 + rm_65_bringup 都在跑,机械臂切到"拖动/示教"模式(可徒手摆姿态)。
  2) 运行本脚本,弹出实时窗口(叠加检测到的棋盘格角点)。
  3) 徒手把机械臂拖到一个姿态,使【整块棋盘格清晰可见且与相机成一定角度】,按 's' 采一对。
     采 15~20 个姿态,每个姿态末端绕 X/Y/Z 都尽量大角度旋转(>30°),不要只平移。
  4) 按 'c' 计算 -> 打印 相机↔Link6 的 旋转矩阵/平移/四元数,并保存到 config/camera_extrinsic.yaml,
     同时生成可贴进 URDF 的 fixed joint 片段。按 'q'/ESC 退出。

原理:cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
      -> R_cam2gripper, t_cam2gripper  (即相机光学系相对 Link6 的位姿,正是我们要的外参)
"""
import argparse
import os
import sys

import numpy as np
import cv2
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import tf2_ros
from scipy.spatial.transform import Rotation as Rot


def parse_args():
    p = argparse.ArgumentParser(description="ROS2 eye-in-hand hand-eye calibration")
    # 棋盘格:内角点数(格子数-1)。默认沿用睿尔曼 config.yaml 的 11x8 / 15mm。改成你实际板子的值!
    p.add_argument("--cols", type=int, default=11, help="棋盘格长边内角点数(默认11)")
    p.add_argument("--rows", type=int, default=8, help="棋盘格短边内角点数(默认8)")
    p.add_argument("--square", type=float, default=0.015, help="单格边长(米,默认0.015)")
    p.add_argument("--base-frame", default="base_link", help="机械臂基座 TF 帧")
    p.add_argument("--ee-frame", default="Link6", help="末端 TF 帧")
    p.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    p.add_argument("--info-topic", default="/camera/camera/color/camera_info")
    p.add_argument("--min-samples", type=int, default=8, help="计算所需最少样本数")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "config", "camera_extrinsic.yaml"))
    # 去掉 ros2 传入的 --ros-args 等
    known, _ = p.parse_known_args()
    return known


class HandEyeCalib(Node):
    def __init__(self, args):
        super().__init__("handeye_calibrate")
        self.args = args
        self.bridge = CvBridge()
        self.K = None          # 3x3 相机内参
        self.D = None          # 畸变系数
        self.last_frame = None
        self.last_found = False
        self.last_corners = None

        # 棋盘格 3D 物点 (Z=0 平面),单位米
        cols, rows, sq = args.cols, args.rows, args.square
        self.pattern = (cols, rows)
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        self.objp = objp * sq

        # 采集到的样本(eye-in-hand 约定)
        self.R_gripper2base = []
        self.t_gripper2base = []
        self.R_target2cam = []
        self.t_target2cam = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, args.info_topic, self._cb_info, 10)
        self.create_subscription(Image, args.image_topic, self._cb_image, 10)

        cv2.namedWindow("handeye", cv2.WINDOW_NORMAL)
        self.get_logger().info("等待相机图像与 camera_info ...")
        self.get_logger().info("按键: s=采集当前姿态  c=计算标定  q/ESC=退出")

    def _cb_info(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
            self.get_logger().info(f"收到内参 fx={self.K[0,0]:.2f} fy={self.K[1,1]:.2f} "
                                   f"cx={self.K[0,2]:.2f} cy={self.K[1,2]:.2f}, 畸变系数 {self.D.ravel()}")

    def _lookup_gripper2base(self):
        """查 TF: Link6 在 base 下的位姿 (gripper2base)。返回 (R 3x3, t 3,) 或 None。"""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.args.base_frame, self.args.ee_frame, rclpy.time.Time())
        except Exception as ex:
            self.get_logger().warn(f"TF {self.args.base_frame}->{self.args.ee_frame} 查询失败: {ex}")
            return None
        q = tf.transform.rotation
        t = tf.transform.translation
        R = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return R, np.array([t.x, t.y, t.z], dtype=np.float64)

    def _cb_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.last_frame = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, self.pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        self.last_found = found
        self.last_corners = corners

        disp = frame.copy()
        if found:
            cv2.drawChessboardCorners(disp, self.pattern, corners, found)
        color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(disp, f"board:{'FOUND' if found else 'no'}  samples:{len(self.R_target2cam)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(disp, "s=capture  c=compute  q=quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("handeye", disp)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord('s'), ord('S')):
            self._capture()
        elif k in (ord('c'), ord('C')):
            self._compute()
        elif k in (ord('q'), ord('Q'), 27):
            raise KeyboardInterrupt

    def _capture(self):
        if self.K is None:
            self.get_logger().warn("还没收到 camera_info,无法采集")
            return
        if not self.last_found:
            self.get_logger().warn("当前帧没检测到棋盘格,换个角度让整块板清晰可见")
            return
        gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerSubPix(
            gray, self.last_corners, (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
        ok, rvec, tvec = cv2.solvePnP(self.objp, corners, self.K, self.D)
        if not ok:
            self.get_logger().warn("solvePnP 失败,跳过")
            return
        g2b = self._lookup_gripper2base()
        if g2b is None:
            self.get_logger().warn("TF 拿不到,跳过(确认 rm_65_bringup 在跑、机械臂在报状态)")
            return
        R_g2b, t_g2b = g2b
        R_t2c, _ = cv2.Rodrigues(rvec)
        self.R_target2cam.append(R_t2c)
        self.t_target2cam.append(tvec.reshape(3))
        self.R_gripper2base.append(R_g2b)
        self.t_gripper2base.append(t_g2b)
        # 重投影误差作为质量反馈
        proj, _ = cv2.projectPoints(self.objp, rvec, tvec, self.K, self.D)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)))
        self.get_logger().info(
            f"采集 #{len(self.R_target2cam)}  重投影误差={err:.3f}px  "
            f"末端位置 base=({t_g2b[0]:.3f},{t_g2b[1]:.3f},{t_g2b[2]:.3f})")

    def _compute(self):
        n = len(self.R_target2cam)
        if n < self.args.min_samples:
            self.get_logger().warn(f"样本只有 {n} 个,建议至少 {self.args.min_samples} 个(姿态多样、大角度旋转)")
            return
        R_c2g, t_c2g = cv2.calibrateHandEye(
            self.R_gripper2base, self.t_gripper2base,
            self.R_target2cam, self.t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI)
        quat = Rot.from_matrix(R_c2g).as_quat()          # [x,y,z,w]
        rpy = Rot.from_matrix(R_c2g).as_euler("xyz")     # URDF rpy
        t = t_c2g.reshape(3)

        # ---- 自一致性验证:板固定不动,反算回 base 的板位置应恒定,离散度=标定误差 ----
        T_c2g = np.eye(4); T_c2g[:3, :3] = R_c2g; T_c2g[:3, 3] = t
        board_pts = []
        for i in range(n):
            T_g2b = np.eye(4); T_g2b[:3, :3] = self.R_gripper2base[i]; T_g2b[:3, 3] = self.t_gripper2base[i]
            T_t2c = np.eye(4); T_t2c[:3, :3] = self.R_target2cam[i]; T_t2c[:3, 3] = self.t_target2cam[i]
            board_pts.append((T_g2b @ T_c2g @ T_t2c)[:3, 3])
        board_pts = np.array(board_pts)
        mean = board_pts.mean(axis=0)
        rms_mm = float(np.sqrt(np.mean(np.sum((board_pts - mean) ** 2, axis=1))) * 1000.0)
        std_mm = board_pts.std(axis=0) * 1000.0

        print("\n================ 手眼标定结果 (相机光学系 相对 Link6) ================")
        print(f"样本数: {n}")
        print(f"[验证] 标定板在 base 下的位置离散度: RMS={rms_mm:.2f}mm  "
              f"各轴std=({std_mm[0]:.2f},{std_mm[1]:.2f},{std_mm[2]:.2f})mm  "
              f"-> <5mm优秀 / <10mm可用 / >15mm建议重采")
        print("旋转矩阵 R_cam2ee:\n", np.array2string(R_c2g, precision=6, suppress_small=True))
        print(f"平移 t_cam2ee (m): [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")
        print(f"四元数 (x,y,z,w): [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
        print(f"欧拉角 rpy (rad): [{rpy[0]:.6f}, {rpy[1]:.6f}, {rpy[2]:.6f}]")

        # 保存 yaml(外参单一真源)
        os.makedirs(os.path.dirname(self.args.out), exist_ok=True)
        out = {
            "parent_frame": self.args.ee_frame,
            "child_frame": "wrist_camera_color_optical",
            "num_samples": n,
            "method": "CALIB_HAND_EYE_TSAI",
            "translation_xyz": [float(v) for v in t],
            "quaternion_xyzw": [float(v) for v in quat],
            "rpy": [float(v) for v in rpy],
            "rotation_matrix": R_c2g.tolist(),
        }
        with open(self.args.out, "w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)
        print(f"\n已保存: {self.args.out}")

        # 可直接贴进 URDF 的 fixed joint(Link6 -> 相机光学系)
        print("\n---- 贴进 URDF 的 fixed joint (Link6 -> 相机光学系) ----")
        print(f'''<joint name="Link6_to_wrist_camera" type="fixed">
  <parent link="{self.args.ee_frame}"/>
  <child link="wrist_camera_color_optical"/>
  <origin xyz="{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>
</joint>''')
        print("\n---- 或临时用 static_transform_publisher 验证 ----")
        print(f"ros2 run tf2_ros static_transform_publisher "
              f"--x {t[0]:.6f} --y {t[1]:.6f} --z {t[2]:.6f} "
              f"--qx {quat[0]:.6f} --qy {quat[1]:.6f} --qz {quat[2]:.6f} --qw {quat[3]:.6f} "
              f"--frame-id {self.args.ee_frame} --child-frame-id wrist_camera_color_optical")
        print("=====================================================================\n")


def main():
    args = parse_args()
    rclpy.init()
    node = HandEyeCalib(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
