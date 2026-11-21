"""主从臂角度一致性实测：走 lerobot 相同归一化路径，20 秒镜像并排打印两臂各关节角度
运行: cd /mnt/nvme01/workplace/lerobot && PYTHONPATH= uv run python myscript/compare_arms.py
⚠️ 3 秒缓冲后从臂上力矩跟随主臂，持续 20 秒。
"""
import json
import time

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus
from lerobot.motors.motors_bus import MotorCalibration

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_bus(port: str, cal_file: str, norm_gripper) -> SerialMotorsBus:
    data = json.load(open(cal_file))
    cal = {}
    for m in NAMES:
        c = data.get("calibration", data)[m]
        cal[m] = MotorCalibration(
            id=c["id"], drive_mode=c["drive_mode"], homing_offset=c["homing_offset"],
            range_min=c["range_min"], range_max=c["range_max"],
        )
    modes = {m: (norm_gripper if m == "gripper" else MotorNormMode.DEGREES) for m in NAMES}
    bus = SerialMotorsBus(
        port=port,
        motors={m: Motor(i + 1, "sts3215", modes[m]) for i, m in enumerate(NAMES)},
        calibration=cal,
    )
    bus.connect()
    return bus


leader = load_bus("/dev/ttyACM1", "/home/x32/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/zihao_leader_arm.json", MotorNormMode.RANGE_0_100)
follower = load_bus("/dev/ttyACM0", "/home/x32/.cache/huggingface/lerobot/calibration/robots/so_follower/zihao_follower_arm.json", MotorNormMode.RANGE_0_100)
print("两臂已连接（已加载标定）。")

print("⏳ 3 秒后从臂上力矩并跟随主臂，持续 20 秒…")
time.sleep(3)
follower.enable_torque()

samples = {m: [] for m in NAMES}
t0 = time.perf_counter()
last_print = 0.0
try:
    while time.perf_counter() - t0 < 20:
        leader_deg = {m: leader.read("Present_Position", m) for m in NAMES}
        follower.sync_write("Goal_Position", leader_deg)  # 与 teleoperate 相同：主臂角度直接作为从臂目标
        present = follower.sync_read("Present_Position")
        for m in NAMES:
            samples[m].append((leader_deg[m], present[m]))
        now = time.perf_counter() - t0
        if now - last_print >= 2.0:
            last_print = now
            print(f"[{now:5.1f}s] " + " | ".join(
                f"{m.split('_')[0][:4]}:{leader_deg[m]:6.1f}/{present[m]:6.1f}" for m in NAMES))
        time.sleep(0.03)
finally:
    follower.disable_torque()

print("\n== 20 秒统计：主臂目标 vs 从臂实际（度）==")
for m in NAMES:
    diffs = [abs(a - b) for a, b in samples[m]]
    leads = [a for a, _ in samples[m]]
    print(f"{m:<15} 主臂范围[{min(leads):6.1f},{max(leads):6.1f}] 平均偏差={sum(diffs)/len(diffs):5.1f} 最大偏差={max(diffs):5.1f}")

leader.disconnect()
follower.disconnect()
print("完成")
