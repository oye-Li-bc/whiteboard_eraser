# whiteboard_eraser — 真机版(RM65 + D435 眼在手上)

把仿真版的"识别古诗错字并擦除"迁到**真实 RM65 机械臂**上:相机(Intel RealSense D435)
装在末端(眼在手上),夹爪夹一块真白板擦,识别手写白板上的古诗错字,移到错字处贴着板面
来回擦掉。

> 仿真版说明见 [`README.md`](README.md)。本文件只讲真机链路。

---

## 1. 整体流程

```
  相机(眼在手上) ──► ArUco 四角 ──► 白板平面(base系: 点p0 + 法向n)
                          │            用 手眼标定外参 + TF(base→Link6) 链式相乘
                          │
  相机图像 ──► PaddleOCR 逐行文字 ──► 匹配内置诗库(差异最小) ──► compare_lines 定位错字
                          │
                          ▼
        错字像素 ──射线与平面求交(反投影)──► 错字 3D(base) ──发布 /wipe_targets + /wipe_plane
                          │
                          ▼(另一个节点)
        订阅目标 ──► IK(试多个滚转取行程最小) ──► 原生 movej 到接近位 ──► 贴板面水平来回擦
```

**感知与运动分离**:`detect_real.py` 只看不动(发布目标);`wipe_real.py` 才控制机械臂。
避免一检测到就乱动,安全。

---

## 2. 硬件 / 环境

- Ubuntu 22.04 + ROS 2 Humble + MoveIt2
- RM65 真机 + 睿尔曼官方 `rm_driver` / `rm_bringup` / `rm_65_config`(通过 UDP/TCP 连真机)
- Intel RealSense D435,`realsense2_camera` 驱动(出厂内参,走 `camera_info`)
- 夹爪:步进电机(`step_motor` 包,串口),**不是** RM 自带末端夹爪
- Python:`paddleocr`(PP-OCRv5)、`opencv-python==4.10.0.84`、`numpy==1.26.4`、`scipy`

> **numpy 必须锁 1.26.4**:装 opencv 时别让它升到 2.x,否则 ROS Humble / cv_bridge 崩
> (`numpy.core.multiarray failed to import`)。
> **国内 pip 报 SSL EOF**:先 `unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY`
> 绕开 Clash 代理再装。

---

## 3. 启动(每个一个终端,保持后台)

```bash
# 终端1 —— RealSense 相机
source /opt/ros/humble/setup.bash
ros2 launch realsense2_camera rs_launch.py

# 终端2 —— RM65 bringup(驱动 + MoveIt + TF base_link→Link6)
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch rm_bringup rm_65_bringup.launch.py     # 等 RViz 弹出、不再刷错即就绪

# 终端3 —— 夹爪步进电机(串口号可能变,先 ls /dev/ttyACM* 确认)
source /opt/ros/humble/setup.bash
source ~/grasp_yoloV8-main/workspaces/step_motor_ros2_ws/install/setup.bash
ros2 run step_motor motor_node --ros-args -p usart_port_name:=/dev/ttyACM1
```

夹爪张开/闭合(持续发布确保送达,`--once` 会因订阅没匹配上而丢消息):

```bash
# 闭合(夹紧)dir=1;张开 dir=0
ros2 topic pub --times 5 -r 5 /motor_control step_motor/msg/Motor \
  "{id: 1, speed: 200, dir: 1, mode: 2, angle: 30000, state: 0, sub_divide: 32}"
```

---

## 4. 标定(一次性,结果已存 `config/`)

### 4.1 手眼标定(眼在手上)→ `config/camera_extrinsic.yaml`
手拖机械臂到不同姿态看棋盘格,每个姿态按 `s` 采样,`c` 计算(cv2.calibrateHandEye TSAI):

```bash
python3 handeye_calibrate.py            # 默认棋盘 11x8 格、格宽 0.015m
```
当前结果:17 样本,自洽 RMS ≈ **4mm**,外参 `Link6 → 相机光心`(平移≈[0.093,-0.029,0.031])。

### 4.2 生成 ArUco 四角标记 → `config/aruco_corners_print.pdf`
```bash
python3 gen_aruco.py                    # DICT_5X5_100, 60mm, id 0=左上 1=右上 2=右下 3=左下
```
打印后贴白板四角(围住诗句)。**贴得越散、越贴到板四角,平面越准**。

### 4.3 验证平面(可选)
```bash
python3 aruco_plane_test.py             # 报 marker 数、平面残差、边长、板心
```

---

## 5. 运行

### 5.1 检测(纯感知,不动机械臂)
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/ros2_ws/src/whiteboard_eraser
python3 detect_real.py --live
```
- 弹窗实时显示相机画面 + 左上状态(`markers x/4` / `TF` / `plane n / p0dist`)。
- 把相机对准白板、四角 marker 都框进去,**按空格**跑一次 OCR 检测:只框错字、
  打印 `错字 base=(x,y,z)`,并 latched 发布 `/wipe_targets` `/wipe_plane`。
- `q` 退出。调试图存 `/tmp/wb_real_view.png`。

### 5.2 擦拭(控制机械臂)
先**只接近验证**(不接触),确认落点/姿态/可达 + 深度:
```bash
python3 wipe_real.py --tcp 0.225 --standoff 0.06 --speed 15      # 无 --press
```
确认接近位真的离板面 ~6cm 后,加 `--press` 真擦:
```bash
python3 wipe_real.py --tcp 0.225 --standoff 0.06 --speed 15 --wipe-speed 12 \
  --wipe-span 0.02 --wipe-h 0.05 --passes 3 --press
```

**最终跑通的参数(重要)**:

| 参数 | 值 | 含义 |
|------|-----|------|
| `--tcp` | **0.225** | 擦头尖沿 Link6 +z 的实际凸出长度(标 0.18 偏短 ~4.5cm,见 §6) |
| `--standoff` | 0.06 | 接近位:擦头尖距板面 6cm |
| `--press-depth` | 0.0 | 压入板面深度,0=只贴到板面不压(擦头有弹性) |
| `--wipe-span` | 0.02 | 水平来回幅度 2cm(沿板面水平 y_col) |
| `--wipe-h` | 0.05 | 竖直覆盖 5cm(沿板面竖直 x_col),别串到上下行 |
| `--passes` | 3 | 来回擦几遍,每遍反向 |
| `--speed` / `--wipe-speed` | 15 / 12 | 关节运动速度百分比(慢,稳) |

⚠️ 真机运动**手放急停上**;工作区清空;第一次接近是关节 movej,可能甩,盯着。

---

## 6. 关键设计 / 踩坑(务必知道)

| 问题 | 结论 / 解决 |
|------|-------------|
| **真机不执行轨迹** | RM 的 `follow_joint_trajectory`(canfd 透传)在真机上不驱动。**改用原生 `/rm_driver/movej_cmd`**:IK 解出关节角直接发 movej,靠监测 `/joint_states` 收敛判到位。 |
| **发指令没反应** | `ros2 topic pub --once` 发太快,订阅还没匹配就退出→消息丢。**要持续发布**(`--times N -r R`,或节点里等 `get_subscription_count()>0` 再发)。 |
| **末端转圈把线绕进去** | 起始姿态某关节接近 ±2π 极限,IK 解落在不同圈数→movej 走近一整圈。**已修**:movej 前把目标角按"离当前角最近圈数"展开(±2π),并加**关节行程上限(~149°)超限拒绝**。 |
| **大转腕/构型跳变** | 擦拭不在乎绕板面法向的自转。**已修**:接近位在法向上试多个滚转角(0/±45/±90/180°),挑关节行程最小的解,擦拭沿用同一滚转。 |
| **接近位差点撞板 / 擦拭压死** | TCP(擦头凸出长)低估~4.5cm→擦头实际更靠板。用"命令 6cm 接近、实测只剩 1~2cm"反推 real TCP≈**0.225m**。校准后接触深度才准。 |
| **把整行都擦了** | 上一条压死 + 擦头有宽度 + 幅度大。校准 TCP + 收小 `--wipe-span`/`--wipe-h` 后只擦单字。 |
| **平面法向/坐标不准** | ArUco marker 挤在一处→平面拟合差、反投影坐标飘。marker **散贴到板四角**最准。 |
| numpy 2.x 崩 ROS / pip SSL | 见 §2。 |

---

## 7. 真机相关文件

| 文件 | 职责 |
|------|------|
| `detect_real.py` | 纯感知节点:ArUco 平面 + PaddleOCR + 诗库匹配 + 反投影 → 发布 `/wipe_targets` `/wipe_plane`;`--live` 实时预览 |
| `wipe_real.py` | 运动节点:订阅目标 → IK(滚转择优 + 关节就近展开 + 行程上限)→ 原生 movej 接近 → 贴板面水平来回擦 |
| `handeye_calibrate.py` | 眼在手上手眼标定(棋盘格 + calibrateHandEye TSAI)→ `config/camera_extrinsic.yaml` |
| `gen_aruco.py` | 生成四角 ArUco 打印页 `config/aruco_corners_print.pdf` + 单张 PNG |
| `aruco_plane_test.py` | 验证 ArUco 四角 → base 系平面拟合(残差/边长/板心) |
| `config/camera_extrinsic.yaml` | 手眼标定外参(Link6→相机光心) |
| `whiteboard_eraser/poems.py` | 内置诗库 + 混淆表 + `compare_lines`(真机版复用) |

> 真机脚本都是**独立脚本**,在包目录下 `python3 xxx.py` 运行(依赖 `from whiteboard_eraser.poems import ...`,
> 需在包根目录跑)。不通过 `ros2 run`。
