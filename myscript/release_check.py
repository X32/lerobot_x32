"""释放从臂力矩 + 打印两臂当前姿势（归一化角度）
运行: cd /mnt/nvme01/workplace/lerobot && PYTHONPATH= uv run python myscript/release_check.py
用途：脚本/遥操作异常中断后先跑这个，确认从臂是松的、姿势正常。
"""
from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus
from lerobot.motors.motors_bus import MotorCalibration
import json

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_bus(port, cal_file):
    data = json.load(open(cal_file))
    cal = {m: MotorCalibration(id=c["id"], drive_mode=c["drive_mode"], homing_offset=c["homing_offset"],
                               range_min=c["range_min"], range_max=c["range_max"])
           for m, c in data.get("calibration", data).items()}
    modes = {m: (MotorNormMode.RANGE_0_100 if m == "gripper" else MotorNormMode.DEGREES) for m in NAMES}
    bus = SerialMotorsBus(port=port,
                          motors={m: Motor(i + 1, "sts3215", modes[m]) for i, m in enumerate(NAMES)},
                          calibration=cal)
    bus.connect()
    return bus


follower = load_bus("/dev/ttyACM0", "/home/x32/.cache/huggingface/lerobot/calibration/robots/so_follower/zihao_follower_arm.json")
try:
    follower.disable_torque()
    print("从臂力矩已释放（现在是松的，可徒手摆动）")
except Exception as e:
    print(f"释放力矩时出错（若报没使能属正常）: {type(e).__name__}: {e}")

leader = load_bus("/dev/ttyACM1", "/home/x32/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/zihao_leader_arm.json")

print("\n== 当前姿势（度，gripper 为 0-100）==")
f_pos = follower.sync_read("Present_Position")
l_pos = leader.sync_read("Present_Position")
for m in NAMES:
    print(f"{m:<15} 主臂={l_pos[m]:8.1f}  从臂={f_pos[m]:8.1f}")

follower.disconnect()
leader.disconnect()
