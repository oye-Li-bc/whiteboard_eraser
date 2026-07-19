"""擦白板总启动：Gazebo(带擦头的RM65)+竖直白板(诗句贴图)+相机 + MoveIt + 视觉/擦拭节点。

启动时先随机选一首诗、制造一个错别字、渲染成白板贴图并写 poem_layout.yaml；
detect 节点用 PaddleOCR 识别白板、与标准诗句比对找错字，wipe 节点擦掉并贴白补丁。
"""
import os
import yaml
from launch import LaunchDescription
from launch.actions import (ExecuteProcess, IncludeLaunchDescription,
                            RegisterEventHandler, TimerAction, SetEnvironmentVariable)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro

from whiteboard_eraser.gen_poem_texture import generate


def generate_launch_description():
    pkg = get_package_share_directory('whiteboard_eraser')
    scene_path = os.path.join(pkg, 'config', 'scene.yaml')
    with open(scene_path, 'r') as f:
        scene = yaml.safe_load(f)
    b, c = scene['board'], scene['camera']

    # 启动即生成诗句贴图(写入 share 的 models 目录)+ 布局标注(供 detect 比对)
    layout_path = '/tmp/poem_layout.yaml'
    generate(scene_path, pkg, layout_path)  # index/seed=None → 每次随机换诗换错字

    # 让 gazebo 能解析白板 material 里的 file://materials/... uri。
    # 关键：必须保留 gazebo 系统资源路径(/usr/share/gazebo-11：含 worlds/shader/基础材质)，
    # 只把包内 models 目录"追加"上去，否则会把系统路径挤掉导致 world/shader 加载失败。
    models_dir = os.path.join(pkg, 'models')
    sys_res = os.environ.get('GAZEBO_RESOURCE_PATH', '')
    if not sys_res:
        for cand in ('/usr/share/gazebo-11', '/usr/share/gazebo'):
            if os.path.isdir(cand):
                sys_res = cand
                break
    gazebo_res = os.pathsep.join([p for p in (sys_res, models_dir) if p])

    # 带擦头的机器人描述
    xacro_path = os.path.join(pkg, 'urdf', 'gazebo_65_eraser.urdf.xacro')
    doc = xacro.parse(open(xacro_path))
    xacro.process_doc(doc)
    robot_desc = {'robot_description': doc.toxml()}

    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen')

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'use_sim_time': True}, robot_desc, {'publish_frequency': 15.0}],
               output='screen')

    spawn_robot = Node(package='gazebo_ros', executable='spawn_entity.py',
                       arguments=['-topic', 'robot_description', '-entity', 'rm_65_description'],
                       output='screen')

    load_jsb = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster'],
        output='screen')
    load_arm = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'rm_group_controller'],
        output='screen')
    evt1 = RegisterEventHandler(OnProcessExit(target_action=spawn_robot, on_exit=[load_jsb]))
    evt2 = RegisterEventHandler(OnProcessExit(target_action=load_jsb, on_exit=[load_arm]))

    # 白板（模型原点=前表面中心）
    spawn_board = Node(package='gazebo_ros', executable='spawn_entity.py',
                       arguments=['-file', os.path.join(pkg, 'models', 'whiteboard.sdf'),
                                  '-entity', 'whiteboard',
                                  '-x', str(b['Xb']), '-y', str(b['center_y']), '-z', str(b['center_z'])],
                       output='screen')
    # 相机
    spawn_cam = Node(package='gazebo_ros', executable='spawn_entity.py',
                     arguments=['-file', os.path.join(pkg, 'models', 'board_camera.sdf'),
                                '-entity', 'board_camera',
                                '-x', str(c['x']), '-y', str(c['y']), '-z', str(c['z']),
                                '-R', str(c['roll']), '-P', str(c['pitch']), '-Y', str(c['yaw'])],
                     output='screen')

    # 相机 TF（与 spawn 位姿一致；detect 节点用 yaml 解析，这里仅供 RViz/tf 查看）
    tf_cam = Node(package='tf2_ros', executable='static_transform_publisher',
                  arguments=['--frame-id', 'world', '--child-frame-id', c['link_frame'],
                             '--x', str(c['x']), '--y', str(c['y']), '--z', str(c['z']),
                             '--roll', str(c['roll']), '--pitch', str(c['pitch']), '--yaw', str(c['yaw'])])
    tf_opt = Node(package='tf2_ros', executable='static_transform_publisher',
                  arguments=['--frame-id', c['link_frame'], '--child-frame-id', c['optical_frame'],
                             '--roll', '-1.5708', '--pitch', '0', '--yaw', '-1.5708'])

    # MoveIt move_group + rviz
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('rm_65_config'), 'launch', 'gazebo_moveit_demo.launch.py')))

    # 视觉 + 擦拭节点（延迟启动，等 gazebo/move_group 起来）
    detect = Node(package='whiteboard_eraser', executable='detect_text_node', output='screen',
                  parameters=[{'use_sim_time': True}, {'scene_yaml': scene_path},
                              {'poem_layout': layout_path}])
    wipe = Node(package='whiteboard_eraser', executable='wipe_node', output='screen',
                parameters=[{'use_sim_time': True}, {'scene_yaml': scene_path}])
    delayed_nodes = TimerAction(period=12.0, actions=[detect, wipe])

    return LaunchDescription([
        SetEnvironmentVariable('GAZEBO_RESOURCE_PATH', gazebo_res),
        evt1, evt2,
        gazebo, rsp, spawn_robot,
        spawn_board, spawn_cam,
        tf_cam, tf_opt,
        moveit,
        delayed_nodes,
    ])
