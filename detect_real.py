#!/usr/bin/env python3
# coding=utf-8
"""真机 detect 节点(纯感知,不动机械臂)。

把三块拼起来,输出错字在 base 系的 3D 坐标:
  1) ArUco 四角  -> 白板平面(base系: 点 p0 + 法向 n)。用 camera_extrinsic.yaml + TF base->Link6 链式相乘。
  2) PaddleOCR    -> 逐行文字 + 错字像素框(行框按字数均分)。
  3) poems.py     -> OCR结果自动匹配内置诗库(差异最小那首)-> compare_lines 定位错字。
  4) 反投影       -> 错字像素射线 与 ArUco平面 求交 -> 错字 3D(base)。发布 /wipe_targets(+sizes)。

安全:本节点不控制机械臂。请先把眼在手上的相机(手动/MoveIt)摆到能同时看到白板文字+4个角marker
     的姿态,再跑本节点。调试图存 /tmp/wb_real_view.png。

数据源: /camera/camera/color/image_raw + camera_info, TF base_link->Link6, config/camera_extrinsic.yaml
"""
import argparse
import os
import threading
import time
import numpy as np
import cv2
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import tf2_ros
from scipy.spatial.transform import Rotation as Rot

from whiteboard_eraser.poems import POEMS, compare_lines

HERE = os.path.dirname(os.path.abspath(__file__))

os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "modelscope")
OCR_KWARGS = dict(ocr_version="PP-OCRv5", lang="ch", enable_mkldnn=False,
                  use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dict", default="DICT_5X5_100")
    p.add_argument("--marker-mm", type=float, default=60.0)
    p.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--base-frame", default="base_link")
    p.add_argument("--ee-frame", default="Link6")
    p.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    p.add_argument("--info-topic", default="/camera/camera/color/camera_info")
    p.add_argument("--extrinsic", default=os.path.join(HERE, "config", "camera_extrinsic.yaml"))
    p.add_argument("--live", action="store_true",
                   help="开实时预览窗口:持续显示画面+ArUco框,按空格跑检测,q退出")
    known, _ = p.parse_known_args()
    return known


def homog(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(t).reshape(3); return T


def run_ocr_lines(ocr, img_bgr):
    """跑 OCR,返回按图像从上到下排序的行 [{text,xmin,ymin,xmax,ymax}, ...]。"""
    res = ocr.predict(img_bgr)
    lines = []
    for r in res:
        texts = r.get("rec_texts") or []
        polys = r.get("rec_polys")
        for t, poly in zip(texts, polys):
            xs = [float(p[0]) for p in poly]; ys = [float(p[1]) for p in poly]
            lines.append({"text": t, "xmin": min(xs), "ymin": min(ys),
                          "xmax": max(xs), "ymax": max(ys)})
    lines.sort(key=lambda d: d["ymin"])
    return lines


def char_pixel_box(line, col, ncol, shrink=0.5):
    """行框内按字数均分,取第 col 个字的像素框 (cx, cy, x0, y0, x1, y1)。"""
    cw = (line["xmax"] - line["xmin"]) / max(ncol, 1)
    cx = line["xmin"] + (col + 0.5) * cw
    hw = cw * shrink / 2.0
    y0, y1 = line["ymin"], line["ymax"]
    return (cx, (y0 + y1) / 2.0, cx - hw, y0, cx + hw, y1)


def match_poem(ocr_texts):
    """OCR逐行文字 自动匹配内置诗库,返回(最佳诗, 字符差异分)。差异最小=标准答案。"""
    best, best_score = None, 1e9
    for poem in POEMS:
        pl = poem["lines"]
        n = min(len(ocr_texts), len(pl))
        if n == 0:
            continue
        score = abs(len(ocr_texts) - len(pl)) * 5  # 行数不符惩罚
        for i in range(n):
            a, b = ocr_texts[i], pl[i]
            score += sum(1 for ca, cb in zip(a, b) if ca != cb) + abs(len(a) - len(b))
        if score < best_score:
            best_score, best = score, poem
    return best, best_score


class DetectReal(Node):
    def __init__(self, args):
        super().__init__("detect_real")
        self.args = args
        self.bridge = CvBridge()
        self.K = self.D = None
        self.published = False
        self._frames = 0
        self.live = args.live
        self.latest = None        # 实时预览用:最近一帧原图

        with open(args.extrinsic, "r", encoding="utf-8") as f:
            ext = yaml.safe_load(f)
        self.T_ee_cam = homog(Rot.from_quat(ext["quaternion_xyzw"]).as_matrix(),
                              ext["translation_xyz"])

        s = args.marker_mm / 1000.0 / 2.0
        self.marker_obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
        adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
        self.detector = cv2.aruco.ArucoDetector(adict, cv2.aruco.DetectorParameters())

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("加载 PaddleOCR ...")
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(**OCR_KWARGS)
        self.get_logger().info("PaddleOCR 就绪。")

        latched = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_targets = self.create_publisher(PoseArray, "/wipe_targets", latched)
        self.pub_sizes = self.create_publisher(Float32MultiArray, "/wipe_target_sizes", latched)
        self.pub_plane = self.create_publisher(Float32MultiArray, "/wipe_plane", latched)

        self.create_subscription(CameraInfo, args.info_topic, self._cb_info, 10)
        self.create_subscription(Image, args.image_topic, self._cb_image, 10)
        self.get_logger().info("detect_real 启动,等待图像... (把相机摆到能看全白板+4角marker)")

    def _cb_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, np.float64).reshape(3, 3)
            self.D = np.array(msg.d, np.float64).reshape(-1, 1)

    def _T_base_cam(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.args.base_frame, self.args.ee_frame,
                                                 rclpy.time.Time())
        except Exception as ex:
            self.get_logger().warn(f"TF 失败: {ex}", throttle_duration_sec=2.0)
            return None
        q = tf.transform.rotation; tr = tf.transform.translation
        T_be = homog(Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), [tr.x, tr.y, tr.z])
        return T_be @ self.T_ee_cam

    def _fit_plane(self, T_bc, corners, ids):
        """4角marker中心 -> base系 -> 拟合平面。返回 (p0, n) 或 None。"""
        pts = []
        for c, mid in zip(corners, ids.flatten()):
            if mid not in self.args.ids:
                continue
            ok, rvec, tvec = cv2.solvePnP(self.marker_obj, c[0], self.K, self.D)
            if ok:
                pts.append((T_bc @ np.array([tvec[0, 0], tvec[1, 0], tvec[2, 0], 1.0]))[:3])
        if len(pts) < 3:
            return None, None, len(pts)
        pts = np.array(pts)
        p0 = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - p0)
        n = vt[2]
        if n @ (T_bc[:3, 3] - p0) < 0:   # 法向朝相机侧
            n = -n
        return p0, n, len(pts)

    def _backproject(self, u, v, T_bc, p0, n):
        """像素射线 与 平面(p0,n) 求交,返回 base 系 3D 点。"""
        C = T_bc[:3, 3]
        d_opt = np.array([(u - self.K[0, 2]) / self.K[0, 0],
                          (v - self.K[1, 2]) / self.K[1, 1], 1.0])
        d = T_bc[:3, :3] @ d_opt
        denom = d @ n
        if abs(denom) < 1e-9:
            return None
        t = ((p0 - C) @ n) / denom
        if t <= 0:
            return None
        return C + t * d

    def _cb_image(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        if self.live:
            self.latest = img          # 实时模式:只存帧,检测由主线程按空格触发
            return
        if self.K is None or self.published:
            return
        self.process(img, publish=True)

    def _overlay(self, img):
        """每帧廉价叠加:ArUco框 + 状态(marker数/TF/平面法向)。不跑OCR。"""
        dbg = img.copy()
        if self.K is None:
            cv2.putText(dbg, "waiting camera_info...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return dbg
        corners, ids, _ = self.detector.detectMarkers(img)
        nmark = 0
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(dbg, corners, ids)
            nmark = sum(1 for m in ids.flatten() if m in self.args.ids)
        T_bc = self._T_base_cam()
        line1 = f"markers {nmark}/4 | TF {'ok' if T_bc is not None else 'NO'}"
        color = (0, 200, 0) if (nmark >= 3 and T_bc is not None) else (0, 165, 255)
        cv2.putText(dbg, line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if nmark >= 3 and T_bc is not None:
            p0, n, _ = self._fit_plane(T_bc, corners, ids)
            if p0 is not None:
                d = float(np.linalg.norm(p0))
                cv2.putText(dbg, f"plane n={n.round(2)} p0dist={d:.2f}m", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        cv2.putText(dbg, "SPACE=detect  q=quit", (10, dbg.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return dbg

    def process(self, img, publish=True):
        """完整检测:ArUco平面 + OCR + 匹配 + 反投影 + (可选)发布。返回叠加调试图。"""
        dbg = img.copy()
        self._frames += 1

        T_bc = self._T_base_cam()
        if T_bc is None:
            self.get_logger().warn("TF 未就绪,无法检测")
            return dbg
        corners, ids, _ = self.detector.detectMarkers(img)
        if ids is None:
            self._log_wait("没检测到 ArUco marker")
            cv2.imwrite("/tmp/wb_real_view.png", dbg); return dbg
        cv2.aruco.drawDetectedMarkers(dbg, corners, ids)
        p0, n, nmark = self._fit_plane(T_bc, corners, ids)
        if p0 is None:
            self._log_wait(f"marker 不足({nmark}<3),看不全4角")
            cv2.imwrite("/tmp/wb_real_view.png", dbg); return dbg

        ocr_lines = run_ocr_lines(self.ocr, img)
        ocr_texts = [l["text"] for l in ocr_lines]
        if not ocr_texts:
            self._log_wait("OCR 没识别到文字"); cv2.imwrite("/tmp/wb_real_view.png", dbg); return dbg

        poem, score = match_poem(ocr_texts)
        errors = compare_lines(ocr_texts, poem["lines"])
        self.get_logger().info(f"OCR={ocr_texts} | 匹配《{poem['title']}》(差异分{score}) | "
                               f"错字数 {len(errors)}")

        targets, sizes, boxes = [], [], []
        for e in errors:
            li, col = e["line"], e["col"]
            if li >= len(ocr_lines):
                continue
            line = ocr_lines[li]
            ncol = len(line["text"])
            pcx, pcy, x0, y0, x1, y1 = char_pixel_box(line, col, ncol)
            center = self._backproject(pcx, pcy, T_bc, p0, n)
            if center is None:
                continue
            corners3d = [self._backproject(u, v, T_bc, p0, n)
                         for (u, v) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]]
            corners3d = [c for c in corners3d if c is not None]
            if len(corners3d) == 4:
                cc = np.array(corners3d)
                wy = float(cc[:, 1].max() - cc[:, 1].min())
                hz = float(cc[:, 2].max() - cc[:, 2].min())
            else:
                wy, hz = 0.04, 0.04
            targets.append(center); sizes.append((wy, hz))
            label = f"{e['seen']}!={e['correct']}"
            boxes.append((int(x0), int(y0), int(x1), int(y1), label))
            cv2.rectangle(dbg, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
            cv2.putText(dbg, label, (int(x0), max(0, int(y0) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imwrite("/tmp/wb_real_view.png", dbg)
        if not targets:
            self._log_wait("识别到诗句但未定位到错字(或行字数不齐)"); return dbg

        pa = PoseArray()
        pa.header.frame_id = self.args.base_frame
        pa.header.stamp = self.get_clock().now().to_msg()
        sz = Float32MultiArray()
        order = sorted(range(len(targets)), key=lambda k: -targets[k][2])  # 从上往下
        for k in order:
            p = Pose()
            p.position.x, p.position.y, p.position.z = [float(v) for v in targets[k]]
            p.orientation.w = 1.0
            pa.poses.append(p)
            sz.data.extend([sizes[k][0], sizes[k][1]])
        plane = Float32MultiArray()
        plane.data = [float(p0[0]), float(p0[1]), float(p0[2]), float(n[0]), float(n[1]), float(n[2])]
        if publish:
            self.pub_targets.publish(pa); self.pub_sizes.publish(sz); self.pub_plane.publish(plane)
            self.published = True
            self.get_logger().info(f"发布 {len(pa.poses)} 个错字目标 + 平面, 调试图 /tmp/wb_real_view.png")
        self.get_logger().info(f"  平面 p0={p0.round(4)} n={n.round(4)}")
        for p in pa.poses:
            self.get_logger().info(f"  错字 base=({p.position.x:.4f},{p.position.y:.4f},{p.position.z:.4f})")
        return dbg

    def _log_wait(self, msg):
        if self._frames % 15 == 1:
            self.get_logger().info(msg)


def main():
    args = parse_args()
    rclpy.init()
    node = DetectReal(args)
    if args.live:
        _run_live(node)
    else:
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


def _run_live(node):
    """主线程刷 cv2 窗口,rclpy 后台 spin,OCR 后台连续跑刷新错字框。空格=发布3D目标,q=退出。"""
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    win = "detect_real (live)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    node.get_logger().info("实时预览开启:把相机对准白板,空格跑检测(只框错字),q退出。")
    try:
        while rclpy.ok():
            img = node.latest
            if img is not None:
                cv2.imshow(win, node._overlay(img))
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            if key == 32 and node.latest is not None:   # 空格
                node.get_logger().info("跑检测中(OCR)...")
                dbg = node.process(node.latest.copy(), publish=True)
                cv2.imshow(win, dbg)
                cv2.waitKey(800)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
