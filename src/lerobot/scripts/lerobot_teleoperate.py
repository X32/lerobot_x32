# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple script to control a robot from teleoperation.

【中文导读】遥操作入口脚本：主臂(teleop)读人手动作 → 处理链 → 从臂(robot)执行，
固定频率循环，不写数据集（要录制请用 lerobot-record）。
执行链：main() → teleoperate()（draccus 把 CLI 参数解析成 TeleoperateConfig 并连接设备）
       → teleop_loop()（observe → teleop → send 三段式主循环）。

Requires: pip install 'lerobot[hardware]'

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

To stream the data to Foxglove instead of Rerun, add ``--display_mode=foxglove``
(then connect the Foxglove app to ``ws://127.0.0.1:8765``; override the port with ``--display_port=<port>``):

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true \
    --display_mode=foxglove
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

# ── 导入说明 ──────────────────────────────────────────────────────
# 下面 cameras/robots/teleoperators 的一堆子模块 import 并不是为了直接用，
# 而是触发其 @register_subclass 装饰器执行，把 CLI 里 "--robot.type=xxx" 的
# 名字注册进 draccus 注册表（side-effect import），因此都标了 noqa: F401。
# make_robot_from_config / make_teleoperator_from_config 再按注册表多态构造实例。
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    bi_yam_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    sim_robot,
    so_follower,
    unitree_g1 as unitree_g1_robot,
    yam_follower,
)
from lerobot.robots.utils import check_control_rate, episode_ended, start_episode
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_openarm_mini,
    bi_rebot_102_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
)
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig            # 主臂（遥操端）配置：动作的来源
    robot: RobotConfig                    # 从臂（被控机器人）配置：动作的执行者
    # Limit the maximum frames per second.
    fps: int = 60                         # 控制循环目标频率（默认 60Hz）
    teleop_time_s: float | None = None    # 遥操作时长（秒）；None = 一直跑到 Ctrl+C
    # Display all cameras on screen
    display_data: bool = False            # 开实时可视化：终端动作表格 + rerun/foxglove
    # Visualization backend used when display_data is True: "rerun" or "foxglove".
    display_mode: str = "rerun"           # 可视化后端选择
    # For "rerun": IP of a remote server to send to. For "foxglove": interface to bind the WebSocket
    # server to (127.0.0.1 for local only, 0.0.0.0 for all interfaces).
    display_ip: str | None = None         # 远程查看地址（rerun=服务器 IP；foxglove=本机绑定网卡）
    # For "rerun": port of the remote server. For "foxglove": port to bind the WebSocket server to.
    display_port: int | None = None       # 远程查看端口
    # Whether to display compressed (JPEG) images instead of raw frames
    display_compressed_images: bool = False  # 图像压缩后再传（远程查看时省带宽）


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    display_mode: str = "rerun",
    duration: float | None = None,
    display_compressed_images: bool = False,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and the
            visualization backend.
        display_mode: Visualization backend to use when display_data is True ("rerun" or "foxglove").
        display_compressed_images: If True, compresses images before sending them to the backend for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    # 终端动作表格 NAME 列的宽度 = 最长的动作键名
    display_len = max(len(key) for key in robot.action_features)
    # Teleoperation writes no dataset, so a missed deadline costs control smoothness
    # only.  The live readout below is the instantaneous rate; the timer adds the
    # warning when the loop cannot keep up and the summary of where the time went.
    # CycleTimer 是节拍器：tick() 分段计时，wait() 睡掉剩余时间维持 1/fps 节拍；
    # 遥操作不写数据集，丢节拍只影响顺滑度（record 录制就不能这么宽容了）。
    timer = CycleTimer(fps, records_data=False)
    start = time.perf_counter()
    try:
        while True:
            timer.tick()
            loop_start = time.perf_counter()  # for the live readout below

            with timer.section("observe"):
                # ① 观测：读从臂状态与相机帧
                # Get robot observation
                # Not really needed for now other than for visualization
                # teleop_action_processor can take None as an observation
                # given that it is the identity processor as default
                # 遥操作本身不依赖 obs（主要供可视化/处理链用；默认处理链可接受 None）
                obs = robot.get_observation()

                # 宇树 G1 特殊逻辑：把观测回传给遥操端（其 leader 需要反馈信号）
                if robot.name == "unitree_g1":
                    teleop.send_feedback(obs)

            with timer.section("teleop"):
                # ② 动作：主臂原始动作 → 遥操处理链（重定向/映射）
                # Get teleop action
                raw_action = teleop.get_action()

                # Process teleop action through pipeline
                teleop_action = teleop_action_processor((raw_action, obs))

                # 再过机器人处理链（限幅/安全化），得到真正要下发的目标动作
                # Process action for robot through pipeline
                robot_action_to_send = robot_action_processor((teleop_action, obs))

            with timer.section("send"):
                # ③ 下发：写入从臂电机（机器人处理链的 to_output 必须产出 RobotAction）
                # Send processed action to robot (robot_action_processor.to_output should return RobotAction)
                _ = robot.send_action(robot_action_to_send)

            # A sim robot stops when its episode ends: start a new one, without counting the reset as loop time.
            # 仿真机器人跑完一个 episode 会停：立即重开，并重置节拍器（重置不计入循环耗时）
            if episode_ended(robot):
                start_episode(robot)
                timer.restart()

            # ④ 可视化（可选）：观测过处理链后推给 rerun/foxglove，终端原地刷新动作数值表
            if display_data:
                with timer.section("telemetry"):
                    # Process robot observation through pipeline
                    obs_transition = robot_observation_processor(obs)

                    log_visualization_data(
                        display_mode,
                        observation=obs_transition,
                        action=teleop_action,
                        compress_images=display_compressed_images,
                    )

                    print("\n" + "-" * (display_len + 10))
                    print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                    # Display the final robot action that was sent
                    for motor, value in robot_action_to_send.items():
                        print(f"{motor:<{display_len}} | {value:>7.2f}")
                    move_cursor_up(len(robot_action_to_send) + 3)

            # wait() 睡掉本 tick 剩余时间保证节拍；随后打印实测耗时/频率并原地刷新
            timer.wait()
            loop_s = time.perf_counter() - loop_start
            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(1)

            # 达到设定时长（--teleop_time_s）则结束
            if duration is not None and time.perf_counter() - start >= duration:
                return
    finally:
        # In `finally` so ^C — how a teleop session normally ends — still reports.
        # 放 finally：即使以 Ctrl+C 结束，也能打印整场各阶段耗时统计
        timer.log_run_summary()


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    # @parser.wrap()：draccus 入口装饰器——命令行参数在此被解析成上面的 TeleoperateConfig
    init_logging()
    logging.info(pformat(asdict(cfg)))  # 回显完整配置，方便核对
    # 需要可视化才初始化后端（rerun viewer / foxglove websocket）
    if cfg.display_data:
        init_visualization(
            cfg.display_mode, session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port
        )
    # 配置了远程查看(ip+port)时强制压缩图像传输
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    # 按注册表多态构造主臂/从臂实例
    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    check_control_rate(robot, cfg.fps, "--fps")  # 校验该机器人总线能否跑得动设定的 fps
    # 三条默认处理链：遥操动作 / 机器人动作 / 机器人观测
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # 连接设备（缺标定文件时会触发自动校准流程）
    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            display_mode=cfg.display_mode,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass  # Ctrl+C 是遥操作的正常结束方式
    finally:
        # 无论怎么退出：关可视化、断开设备（电机断力矩/相机停流）
        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()  # 注册第三方硬件插件包（BYOH）进 draccus 注册表
    teleoperate()


if __name__ == "__main__":
    main()
