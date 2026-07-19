"""视觉识别节点：用 PaddleOCR 识别白板上的古诗，与标准诗句比对找错别字，
只把"错字"的板面世界坐标发布出去让机械臂擦掉。

输入: /board_camera/.../image_raw, /board_camera/.../camera_info
     config/poem_layout.yaml (由 gen_poem_texture 生成，提供 correct_lines 标准答案)
输出: /wipe_targets (PoseArray，每个 pose = 一个错字中心的世界坐标)
     /wipe_target_sizes (Float32MultiArray，[w0,h0,...] 错字板面包围盒)
调试: /tmp/whiteboard_view.png(每帧相机视角) /tmp/whiteboard_detect.png(标注错字)

识别流程见 detect_errors_in_image()：整图 OCR→按行→与标准诗逐字比对→行框按字数
均分定位错字像素中心→backproject_pixel 反投影到板面。OCR 是真识别，poem_layout 只给答案。
"""
import os
import yaml
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

from whiteboard_eraser.geometry_utils import backproject_pixel
from whiteboard_eraser.poems import compare_lines

# PaddleOCR 在本机跑通的必需参数（见 paddleocr-paddle-quirks 记忆）
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "modelscope")
OCR_KWARGS = dict(ocr_version="PP-OCRv5", lang="ch", enable_mkldnn=False,
                  use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=False)


def run_ocr_lines(ocr, img_bgr):
    """跑 OCR，返回按板面从上到下排序的行列表 [{text, xmin,ymin,xmax,ymax}, ...]。"""
    res = ocr.predict(img_bgr)
    lines = []
    for r in res:
        texts = r.get("rec_texts") or []
        polys = r.get("rec_polys")
        scores = r.get("rec_scores") or [1.0] * len(texts)
        for t, poly, s in zip(texts, polys, scores):
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            lines.append({"text": t, "score": float(s),
                          "xmin": min(xs), "ymin": min(ys),
                          "xmax": max(xs), "ymax": max(ys)})
    lines.sort(key=lambda d: d["ymin"])  # 图像上→下
    return lines


def char_pixel_box(line, col, ncol, shrink=0.5):
    """行框内按字数均分，取第 col 个字的像素框 (cx, cy, x0, y0, x1, y1)。

    定位中心用整格中点；但返回的尺寸框只取字格中间 shrink(默认50%)宽度，
    因为字形本身不占满字距，用整格宽会导致擦拭/白补丁糊到左右邻字。
    """
    cw = (line["xmax"] - line["xmin"]) / ncol
    cx = line["xmin"] + (col + 0.5) * cw
    hw = cw * shrink / 2.0
    x0, x1 = cx - hw, cx + hw
    y0, y1 = line["ymin"], line["ymax"]
    return (cx, (y0 + y1) / 2.0, x0, y0, x1, y1)


def detect_errors_in_image(ocr, img_bgr, correct_lines, K, cam_xyz, cam_rpy, plane_x,
                           dbg=None):
    """核心：识别→比对→定位错字世界坐标。

    返回 (targets, sizes, ocr_texts)：
      targets: [np.array([x,y,z]), ...] 错字板面世界坐标
      sizes:   [(wy, hz), ...] 错字板面包围盒
      ocr_texts: OCR 逐行文本(调试用)
    """
    fx, fy, cx, cy = K
    ocr_lines = run_ocr_lines(ocr, img_bgr)
    ocr_texts = [l["text"] for l in ocr_lines]
    errors = compare_lines(ocr_texts, correct_lines)

    targets, sizes = [], []
    for e in errors:
        li, col = e["line"], e["col"]
        line = ocr_lines[li]
        ncol = len(line["text"])
        pcx, pcy, x0, y0, x1, y1 = char_pixel_box(line, col, ncol)
        center = backproject_pixel(pcx, pcy, fx, fy, cx, cy, cam_xyz, cam_rpy, plane_x)
        if center is None:
            continue
        # 反投影单字四角估板面尺寸
        corners = [backproject_pixel(u, v, fx, fy, cx, cy, cam_xyz, cam_rpy, plane_x)
                   for (u, v) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]]
        corners = [c for c in corners if c is not None]
        if len(corners) == 4:
            wy = max(c[1] for c in corners) - min(c[1] for c in corners)
            hz = max(c[2] for c in corners) - min(c[2] for c in corners)
        else:
            wy, hz = 0.06, 0.06
        targets.append(center)
        sizes.append((float(wy), float(hz)))
        if dbg is not None:
            cv2.rectangle(dbg, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
            cv2.putText(dbg, f"{e['seen']}!={e['correct']}", (int(x0), max(0, int(y0) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return targets, sizes, ocr_texts


class DetectTextNode(Node):
    def __init__(self):
        super().__init__('detect_text_node')
        self.declare_parameter('scene_yaml', '')
        self.declare_parameter('poem_layout', '')
        scene_path = self.get_parameter('scene_yaml').get_parameter_value().string_value
        layout_path = self.get_parameter('poem_layout').get_parameter_value().string_value
        with open(scene_path, 'r') as f:
            self.scene = yaml.safe_load(f)
        with open(layout_path, 'r') as f:
            self.layout = yaml.safe_load(f)
        self.correct_lines = list(self.layout['correct_lines'])
        typo = self.layout.get('typo', {})
        self.get_logger().info(
            f"标准答案《{self.layout.get('title','?')}》，"
            f"预置错字 '{typo.get('correct')}'->'{typo.get('wrong')}' "
            f"@行{typo.get('line')}列{typo.get('col')}")

        b, c = self.scene['board'], self.scene['camera']
        self.plane_x = float(b['Xb'])
        self.cam_xyz = (float(c['x']), float(c['y']), float(c['z']))
        self.cam_rpy = (float(c['roll']), float(c['pitch']), float(c['yaw']))

        self.bridge = CvBridge()
        self.K = None
        self.published = False
        self._frames = 0

        self.get_logger().info('正在加载 PaddleOCR 模型...')
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(**OCR_KWARGS)
        self.get_logger().info('PaddleOCR 就绪。')

        latched = QoSProfile(depth=1,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_targets = self.create_publisher(PoseArray, '/wipe_targets', latched)
        self.pub_sizes = self.create_publisher(Float32MultiArray, '/wipe_target_sizes', latched)

        self.create_subscription(CameraInfo, '/board_camera/board_camera/camera_info', self.cb_info, 10)
        self.create_subscription(Image, '/board_camera/board_camera/image_raw', self.cb_image, 10)
        self.get_logger().info('detect_text_node 启动，等待相机图像...')

    def cb_info(self, msg: CameraInfo):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])  # fx,fy,cx,cy

    def cb_image(self, msg: Image):
        if self.K is None or self.published:
            return
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        dbg = img.copy()
        try:
            targets, sizes, ocr_texts = detect_errors_in_image(
                self.ocr, img, self.correct_lines, self.K,
                self.cam_xyz, self.cam_rpy, self.plane_x, dbg=dbg)
        except Exception as ex:  # OCR 偶发失败不崩节点
            self.get_logger().warn(f'OCR 处理异常: {ex}')
            return

        cv2.imwrite('/tmp/whiteboard_view.png', dbg)
        self._frames += 1
        if not ocr_texts:
            if self._frames % 15 == 1:
                self.get_logger().warn('OCR 未识别到文字 —— 看 /tmp/whiteboard_view.png 核对相机视角/遮挡')
            return
        self.get_logger().info(f'OCR 识别: {ocr_texts}')

        if not targets:
            if self._frames % 15 == 1:
                self.get_logger().info('识别到诗句但未发现错别字。')
            return

        pa = PoseArray()
        pa.header.frame_id = self.scene['robot']['base_frame']
        pa.header.stamp = self.get_clock().now().to_msg()
        sz = Float32MultiArray()
        # 按 z 从高到低（从上往下擦）
        order = sorted(range(len(targets)), key=lambda k: -targets[k][2])
        for k in order:
            p = Pose()
            p.position.x, p.position.y, p.position.z = \
                float(targets[k][0]), float(targets[k][1]), float(targets[k][2])
            p.orientation.w = 1.0
            pa.poses.append(p)
            sz.data.extend([sizes[k][0], sizes[k][1]])

        self.pub_targets.publish(pa)
        self.pub_sizes.publish(sz)
        cv2.imwrite('/tmp/whiteboard_detect.png', dbg)
        self.published = True
        self.get_logger().info(
            f'发现 {len(pa.poses)} 个错别字，已发布 /wipe_targets，调试图存 /tmp/whiteboard_detect.png')
        for p in pa.poses:
            self.get_logger().info(
                f'  错字 world=({p.position.x:.3f}, {p.position.y:.3f}, {p.position.z:.3f})')


def main(args=None):
    rclpy.init(args=args)
    node = DetectTextNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
