"""擦拭运动节点。

流程：订阅 /wipe_targets(+sizes) → 对每个黑字：
  1) /compute_ik 求"板前5cm预备位姿"的关节解 → 控制器 action 移动过去
  2) geometry_utils 生成板面弓字形路点 → /compute_cartesian_path 规划直线轨迹
  3) 给轨迹补上均匀时间参数 → /rm_group_controller/follow_joint_trajectory 执行
  4) 末尾抬离 5cm，再处理下一个

设计为后台线程 spin executor，工作线程顺序阻塞执行。
"""
import time
import threading
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from std_msgs.msg import Float32MultiArray
from moveit_msgs.srv import GetPositionIK, GetCartesianPath
from moveit_msgs.msg import RobotState, PositionIKRequest
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.srv import SpawnEntity

from whiteboard_eraser.geometry_utils import link6_target, eraser_quat, boustrophedon


class WipeNode(Node):
    def __init__(self):
        super().__init__('wipe_node')
        self.declare_parameter('scene_yaml', '')
        scene_path = self.get_parameter('scene_yaml').get_parameter_value().string_value
        with open(scene_path, 'r') as f:
            self.scene = yaml.safe_load(f)
        b, e, r = self.scene['board'], self.scene['eraser'], self.scene['robot']
        self.Xb = float(b['Xb'])
        self.l_erase = float(e['length'])
        self.half = float(e['half'])
        self.press = float(e['press'])
        self.group = r['group']
        self.base = r['base_frame']
        self.ee = r['ee_link']
        self.joints = list(r['joints'])
        self.ctrl_action = r['controller_action']

        self.latest_js = None
        self.targets = None
        self.sizes = None
        self._started = False

        self.create_subscription(JointState, '/joint_states', self._cb_js, 10)
        latched = QoSProfile(depth=1,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseArray, '/wipe_targets', self._cb_targets, latched)
        self.create_subscription(Float32MultiArray, '/wipe_target_sizes', self._cb_sizes, latched)

        self.ik_cli = self.create_client(GetPositionIK, '/compute_ik')
        self.cart_cli = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        self.traj_ac = ActionClient(self, FollowJointTrajectory, self.ctrl_action)
        self.get_logger().info('wipe_node 启动，等待 /wipe_targets ...')

    # ---------- 回调 ----------
    def _cb_js(self, msg):
        self.latest_js = msg

    def _cb_targets(self, msg):
        self.targets = msg
        self._maybe_start()

    def _cb_sizes(self, msg):
        self.sizes = list(msg.data)
        self._maybe_start()

    def _maybe_start(self):
        if self._started or self.targets is None or self.sizes is None:
            return
        self._started = True
        threading.Thread(target=self._run, daemon=True).start()

    # ---------- 工具 ----------
    def _wait(self, future, timeout=30.0):
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > timeout:
                return None
            time.sleep(0.02)
        return future.result()

    def _robot_state(self):
        rs = RobotState()
        if self.latest_js is not None:
            rs.joint_state = self.latest_js
        return rs

    def _make_pose(self, pos, quat):
        p = Pose()
        p.position.x, p.position.y, p.position.z = float(pos[0]), float(pos[1]), float(pos[2])
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = \
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        return p

    def _compute_ik(self, pos, quat):
        req = GetPositionIK.Request()
        ikr = PositionIKRequest()
        ikr.group_name = self.group
        ikr.ik_link_name = self.ee
        ikr.robot_state = self._robot_state()
        ikr.avoid_collisions = False
        ps = PoseStamped()
        ps.header.frame_id = self.base
        ps.pose = self._make_pose(pos, quat)
        ikr.pose_stamped = ps
        ikr.timeout.sec = 2
        req.ik_request = ikr
        res = self._wait(self.ik_cli.call_async(req))
        if res is None or res.error_code.val != 1:
            return None
        # 取规划组关节顺序的解
        name2pos = dict(zip(res.solution.joint_state.name,
                            res.solution.joint_state.position))
        try:
            return [name2pos[j] for j in self.joints]
        except KeyError:
            return None

    def _send_traj(self, jt: JointTrajectory, settle=0.5):
        if not self.traj_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('控制器 action 不可用')
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt
        gh = self._wait(self.traj_ac.send_goal_async(goal))
        if gh is None or not gh.accepted:
            self.get_logger().error('轨迹被拒绝')
            return False
        self._wait(gh.get_result_async(), timeout=60.0)
        time.sleep(settle)
        return True

    def _goto_joints(self, target_joints, duration=3.0):
        if self.latest_js is None:
            return False
        cur = dict(zip(self.latest_js.name, self.latest_js.position))
        jt = JointTrajectory()
        jt.joint_names = self.joints
        p0 = JointTrajectoryPoint()
        p0.positions = [cur.get(j, 0.0) for j in self.joints]
        p0.time_from_start.sec = 0
        p0.time_from_start.nanosec = int(1e7)
        p1 = JointTrajectoryPoint()
        p1.positions = [float(x) for x in target_joints]
        p1.time_from_start.sec = int(duration)
        p1.time_from_start.nanosec = int((duration % 1) * 1e9)
        jt.points = [p0, p1]
        return self._send_traj(jt)

    def _cartesian(self, waypoints, dt=0.08):
        req = GetCartesianPath.Request()
        req.header.frame_id = self.base
        req.start_state = self._robot_state()
        req.group_name = self.group
        req.link_name = self.ee
        req.waypoints = waypoints
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = False
        res = self._wait(self.cart_cli.call_async(req))
        if res is None:
            return None, 0.0
        jt = res.solution.joint_trajectory
        # 补均匀时间参数（GetCartesianPath 返回无时序）
        for i, pt in enumerate(jt.points):
            tt = (i + 1) * dt
            pt.time_from_start.sec = int(tt)
            pt.time_from_start.nanosec = int((tt % 1) * 1e9)
            pt.velocities = []
            pt.accelerations = []
        return jt, res.fraction

    def _spawn_white_patch(self, idx, center, wy, hz):
        """在错字世界坐标贴一块白色薄板盖住它，让错字视觉上"被擦掉"。
        贴在白板前表面前方 3mm(朝机械臂侧)，尺寸略大于错字包围盒。"""
        if not self.spawn_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('  /spawn_entity 不可用，跳过白色补丁')
            return
        # 补丁略大于错字即可，并封顶避免盖住邻字（字距约 0.06~0.07m）
        sy = min(max(wy, 0.02) + 0.004, 0.05)
        sz = min(max(hz, 0.02) + 0.004, 0.06)
        sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="patch_{idx}">
    <static>true</static>
    <link name="link">
      <visual name="v">
        <geometry><box><size>0.004 {sy:.4f} {sz:.4f}</size></box></geometry>
        <material>
          <ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
        req = SpawnEntity.Request()
        req.name = f'patch_{idx}'
        req.xml = sdf
        req.reference_frame = 'world'
        # 前表面 x=Xb，补丁贴到其前方 3mm(x 更小=朝机械臂)
        req.initial_pose.position.x = float(self.Xb - 0.003)
        req.initial_pose.position.y = float(center[1])
        req.initial_pose.position.z = float(center[2])
        req.initial_pose.orientation.w = 1.0
        res = self._wait(self.spawn_cli.call_async(req), timeout=10.0)
        if res is None or not res.success:
            self.get_logger().warn(f'  白色补丁 spawn 失败: {getattr(res, "status_message", "timeout")}')
        else:
            self.get_logger().info('  已用白色补丁盖住错字')

    # ---------- 主流程 ----------
    def _run(self):
        time.sleep(2.0)  # 等 move_group 服务就绪
        self.ik_cli.wait_for_service()
        self.cart_cli.wait_for_service()
        quat = eraser_quat()
        n = len(self.targets.poses)
        self.get_logger().info(f'开始擦拭，共 {n} 个错别字')
        for idx, pose in enumerate(self.targets.poses):
            cy, cz = pose.position.y, pose.position.z
            wy = self.sizes[2 * idx] if 2 * idx < len(self.sizes) else 0.06
            hz = self.sizes[2 * idx + 1] if 2 * idx + 1 < len(self.sizes) else 0.04
            self.get_logger().info(f'[{idx+1}/{n}] 错字 y={cy:.3f} z={cz:.3f} 尺寸 {wy:.3f}x{hz:.3f}')

            world_center = np.array([self.Xb, cy, cz])
            link6_pos, _ = link6_target(world_center, self.Xb, self.l_erase, self.press)
            prep_pos = link6_pos.copy()
            prep_pos[0] -= 0.05  # 板前 5cm

            # 1) 预备位姿
            ik = self._compute_ik(prep_pos, quat)
            if ik is None:
                self.get_logger().warn('  预备位姿 IK 失败，跳过该错字')
                continue
            if not self._goto_joints(ik):
                self.get_logger().warn('  移动到预备位姿失败，跳过')
                continue

            # 2) 板面弓字形路点 → 笛卡尔轨迹（含尾部抬离）
            yz = boustrophedon(cy, cz, wy, hz, row_step=2 * self.half)
            x_contact = self.Xb + self.press - self.l_erase
            waypoints = [self._make_pose([x_contact, y, z], quat) for (y, z) in yz]
            lift = self._make_pose([x_contact - 0.05, yz[-1][0], yz[-1][1]], quat)
            waypoints.append(lift)
            jt, frac = self._cartesian(waypoints)
            if jt is None or frac < 0.5 or len(jt.points) < 2:
                self.get_logger().warn(f'  笛卡尔规划覆盖率 {frac:.2f} 过低，跳过')
                continue
            self.get_logger().info(f'  笛卡尔规划覆盖率 {frac:.2f}，执行擦拭 {len(jt.points)} 点')
            self._send_traj(jt)

            # 擦完贴白色补丁盖住错字，视觉上真正"擦除"
            self._spawn_white_patch(idx, world_center, wy, hz)

        self.get_logger().info('全部错别字处理完毕。')


def main(args=None):
    rclpy.init(args=args)
    node = WipeNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
