#!/usr/bin/env python3
# coding=utf-8
"""ArUco 四角 -> 白板平面 检测验证工具(独立跑,不改 TF 树)。

用途:贴好四角 marker 后,验证能否稳定检测 4 个角、算出它们在 base 系的 3D 坐标、
     并拟合出白板平面。这套"读外参yaml + 查TF base->Link6 + 链式相乘"的逻辑,
     之后原样搬进真机 detect 节点。

坐标链(不依赖 realsense 发布的相机 TF,避免双父冲突):
  T_base_marker = T_base_Link6(查TF) · T_Link6_camoptical(读camera_extrinsic.yaml) · T_camoptical_marker(solvePnP)

数据源: /camera/camera/color/image_raw + camera_info, TF base_link->Link6, config/camera_extrinsic.yaml
"""
import argparse
import os
import numpy as np
import cv2
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf2_ros
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dict", default="DICT_5X5_100")
    p.add_argument("--marker-mm", type=float, default=60.0, help="marker 物理边长(mm)")
    p.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3], help="四角期望 id")
    p.add_argument("--base-frame", default="base_link")
    p.add_argument("--ee-frame", default="Link6")
    p.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    p.add_argument("--info-topic", default="/camera/camera/color/camera_info")
    p.add_argument("--extrinsic", default=os.path.join(HERE, "config", "camera_extrinsic.yaml"))
    known, _ = p.parse_known_args()
    return known


def homog(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


class ArucoPlaneTest(Node):
    def __init__(self, args):
        super().__init__("aruco_plane_test")
        self.args = args
        self.bridge = CvBridge()
        self.K = None
        self.D = None

        # 读手眼外参 -> T_Link6_camoptical
        with open(args.extrinsic, "r", encoding="utf-8") as f:
            ext = yaml.safe_load(f)
        q = ext["quaternion_xyzw"]
        t = ext["translation_xyz"]
        self.T_ee_cam = homog(Rot.from_quat(q).as_matrix(), t)
        self.get_logger().info(f"读外参 {args.extrinsic}: t={t}")

        # marker 3D 角点(marker 自身坐标系,z=0),顺序与 aruco 输出一致:TL,TR,BR,BL
        s = args.marker_mm / 1000.0 / 2.0
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)

        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, args.info_topic, self._cb_info, 10)
        self.create_subscription(Image, args.image_topic, self._cb_image, 10)
        cv2.namedWindow("aruco_plane", cv2.WINDOW_NORMAL)
        self.get_logger().info("等待图像... (按 q/ESC 退出)")

    def _cb_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, np.float64).reshape(3, 3)
            self.D = np.array(msg.d, np.float64).reshape(-1, 1)

    def _T_base_cam(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.args.base_frame, self.args.ee_frame,
                                                 rclpy.time.Time())
        except Exception as ex:
            self.get_logger().warn(f"TF 查询失败: {ex}")
            return None
        q = tf.transform.rotation
        tr = tf.transform.translation
        T_base_ee = homog(Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
                          [tr.x, tr.y, tr.z])
        return T_base_ee @ self.T_ee_cam

    def _cb_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        disp = frame.copy()
        if self.K is not None:
            corners, ids, _ = self.detector.detectMarkers(frame)
            centers_base = {}
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(disp, corners, ids)
                T_bc = self._T_base_cam()
                for c, mid in zip(corners, ids.flatten()):
                    if mid not in self.args.ids:
                        continue
                    ok, rvec, tvec = cv2.solvePnP(self.obj, c[0], self.K, self.D)
                    if not ok or T_bc is None:
                        continue
                    p_base = (T_bc @ np.array([tvec[0, 0], tvec[1, 0], tvec[2, 0], 1.0]))[:3]
                    centers_base[int(mid)] = p_base

            found = sorted(centers_base.keys())
            cv2.putText(disp, f"markers found: {found}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if len(found) >= 4 else (0, 165, 255), 2)

            if len(centers_base) >= 3:
                pts = np.array([centers_base[k] for k in found])
                centroid = pts.mean(axis=0)
                # SVD 拟合平面法向
                u, s_, vt = np.linalg.svd(pts - centroid)
                normal = vt[2]
                resid = np.abs((pts - centroid) @ normal)  # 各点到平面距离
                y2 = 60
                cv2.putText(disp, f"plane fit resid max={resid.max()*1000:.1f}mm",
                            (10, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if len(centers_base) == 4:
                    # 相邻角距离(核对 marker 尺寸/检测是否正确)
                    order = [0, 1, 2, 3]
                    have = [i for i in order if i in centers_base]
                    if have == order:
                        d = lambda a, b: np.linalg.norm(centers_base[a] - centers_base[b])
                        self.get_logger().info(
                            f"平面法向={normal.round(3)} 残差max={resid.max()*1000:.1f}mm | "
                            f"边长(m) 上{d(0,1):.3f} 右{d(1,2):.3f} 下{d(2,3):.3f} 左{d(3,0):.3f} | "
                            f"板心base={centroid.round(3)}", throttle_duration_sec=1.0)
        cv2.imshow("aruco_plane", disp)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), ord('Q'), 27):
            raise KeyboardInterrupt


def main():
    args = parse_args()
    rclpy.init()
    node = ArucoPlaneTest(args)
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
