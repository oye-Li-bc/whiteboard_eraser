"""几何工具：像素→板面反投影、板面点→Link6目标位姿、弓字形覆盖路点。

约定：世界系=base_link，x 向前(指向白板)，z 向上。
相机：Gazebo 相机沿其 link 的 +x 看出去；光学系(z 前,x 右,y 下)与 link 的关系为
      link = (z_opt, -x_opt, -y_opt)。相机 link 在世界的旋转由 spawn 的 rpy 决定，
      R_wl = Rz(yaw)·Ry(pitch)·Rx(roll)，与 Gazebo pose 的 rpy 一致。
所有反投影只依赖 (相机位姿 from yaml) + (内参 from camera_info)，确定性强、无 tf 歧义。
"""
import numpy as np


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_to_quat(R):
    """旋转矩阵 → 四元数 (x, y, z, w)。"""
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def backproject_pixel(u, v, fx, fy, cx, cy, cam_xyz, cam_rpy, plane_x):
    """把像素 (u,v) 反投影到世界平面 x=plane_x，返回世界坐标 (3,)。"""
    d_opt = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    d_link = np.array([d_opt[2], -d_opt[0], -d_opt[1]])
    R_wl = rpy_to_matrix(*cam_rpy)
    d_world = R_wl @ d_link
    C = np.asarray(cam_xyz, dtype=float)
    if abs(d_world[0]) < 1e-9:
        return None  # 射线平行于板面
    t = (plane_x - C[0]) / d_world[0]
    if t <= 0:
        return None  # 交点在相机背后
    return C + t * d_world


# 擦头标称姿态：Link6 局部 +z(工具轴)→世界 +x(朝板)，局部 +x→世界 +z(向上)。
# R 各列 = 局部轴在世界的像：x->(0,0,1), y->(0,-1,0), z->(1,0,0)
_ERASER_R = np.array([[0.0, 0.0, 1.0],
                      [0.0, -1.0, 0.0],
                      [1.0, 0.0, 0.0]])


def eraser_quat():
    """擦头水平压板的标称四元数 (x,y,z,w)。实测如不贴合，调此矩阵或在 yaml 加偏置。"""
    return matrix_to_quat(_ERASER_R)


def link6_target(world_pt, plane_x, l_erase, press):
    """给定板面上的世界点，求 Link6 目标 (position xyz, quat xyzw)，使擦头面贴住该点。
    擦头沿 +x 凸出 l_erase，press 为压入深度（擦头尖 x = plane_x + press）。"""
    pos = np.array([plane_x + press - l_erase, world_pt[1], world_pt[2]])
    return pos, eraser_quat()


def boustrophedon(center_y, center_z, w, h, row_step, margin=0.003):
    """在板面 (y,z) 内生成弓字形覆盖路点序列 [(y,z), ...]，覆盖 w×h 包围盒。"""
    half_w = w / 2.0 + margin
    half_h = h / 2.0 + margin
    z_top = center_z + half_h
    z_bot = center_z - half_h
    n_rows = max(2, int(np.ceil((z_top - z_bot) / max(row_step, 1e-3))) + 1)
    zs = np.linspace(z_top, z_bot, n_rows)
    y_left = center_y - half_w
    y_right = center_y + half_w
    pts = []
    for i, z in enumerate(zs):
        if i % 2 == 0:
            pts.append((y_left, z))
            pts.append((y_right, z))
        else:
            pts.append((y_right, z))
            pts.append((y_left, z))
    return pts
