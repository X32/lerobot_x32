"""
单独校准一个电机的零点与行程（无力矩，只动标定文件里这一个电机）
用法:
  cd /mnt/nvme01/workplace/lerobot
  PYTHONPATH= uv run python myscript/calibrate_single_motor.py follower elbow_flex
  PYTHONPATH= uv run python myscript/calibrate_single_motor.py leader shoulder_lift
  （arm: follower|leader；电机名: shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll/gripper）

流程: 把该关节摆到行程中位 → 回车(半圈归零) → 缓慢扫该关节全行程 → 回车(记录范围) → 写回标定文件
注意:
  - wrist_roll 是整圈关节，官方流程不扫行程，本脚本会自动跳过 sweep、保留旧 range
  - 修改前自动备份标定文件为 <原名>.json.bak
  - 主从配对: 若单标了一臂的某关节，另一臂同关节的中位姿势要和这次摆的一致，
    否则两臂 0° 不再对应（建议两臂背靠背同姿势各标一次）
"""
import json
import shutil
import sys
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus
from lerobot.motors.motors_bus import MotorCalibration

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARMS = {
    "follower": ("/dev/ttyACM0", Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/zihao_follower_arm.json"),
    "leader":   ("/dev/ttyACM1", Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so_leader/zihao_leader_arm.json"),
}
# 各关节的"行程中位"参考描述
MIDDLE_HINTS = {
    "shoulder_pan":  "底座水平转到行程中间（大致朝前）",
    "shoulder_lift": "大臂抬到行程中间（大致水平）",
    "elbow_flex":    "肘部弯到行程中间（约半弯）",
    "wrist_flex":    "腕部上下摆到中间",
    "wrist_roll":    "腕部滚转拧到中间",
    "gripper":       "夹爪开到行程中间（半开）",
}


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ARMS or sys.argv[2] not in NAMES:
        print(__doc__)
        sys.exit(1)
    arm, motor = sys.argv[1], sys.argv[2]
    port, cal_file = ARMS[arm]

    data = json.loads(cal_file.read_text())
    cal_all = {m: c for m, c in data.get("calibration", data).items()}
    old = cal_all[motor]
    print(f"== {arm} / {motor} 当前标定 ==")
    print(f"   id={old['id']} drive_mode={old['drive_mode']} homing={old['homing_offset']} range=[{old['range_min']}, {old['range_max']}]")

    print(f"\n连接 {port} …")
    cal = {m: MotorCalibration(id=c["id"], drive_mode=c["drive_mode"], homing_offset=c["homing_offset"],
                               range_min=c["range_min"], range_max=c["range_max"])
           for m, c in cal_all.items()}
    modes = {m: (MotorNormMode.RANGE_0_100 if m == "gripper" else MotorNormMode.DEGREES) for m in NAMES}
    bus = SerialMotorsBus(port=port, motors={m: Motor(i + 1, "sts3215", modes[m]) for i, m in enumerate(NAMES)}, calibration=cal)
    bus.connect()
    print("连接成功（无力矩，全程可徒手摆动）。\n")

    print(f"┌ 第 1 步：把 [{motor}] 摆到行程中位 —— {MIDDLE_HINTS[motor]}")
    print(f"└ 其他关节不用管。摆好后按回车 > ", end="")
    input()
    offsets = bus.set_half_turn_homings([motor])
    new_homing = offsets[motor]
    print(f"   → 新 homing_offset = {new_homing}\n")

    if motor == "wrist_roll":
        print("wrist_roll 是整圈关节，按官方流程不扫行程，保留旧 range。")
        new_min, new_max = old["range_min"], old["range_max"]
    else:
        print(f"┌ 第 2 步：只缓慢扫 [{motor}] 的全行程（来回转满），其他关节不动")
        print(f"└ 扫完后按回车结束 > ")
        mins, maxes = bus.record_ranges_of_motion([motor])
        new_min, new_max = int(mins[motor]), int(maxes[motor])
        span = new_max - new_min
        if span >= 3900 and motor != "wrist_roll":
            print(f"   ⚠️ 警告: 范围 [{new_min},{new_max}] 跨度 {span} 接近整圈 4096，")
            print(f"      可能扫行程时越过了编码器 0/4095 边界，建议重扫慢一点、别越过尽头。")
    bus.disconnect()

    # 更新这一个电机，其余原样保留
    cal_all[motor] = {"id": old["id"], "drive_mode": old["drive_mode"],
                      "homing_offset": int(new_homing), "range_min": new_min, "range_max": new_max}
    shutil.copy(cal_file, cal_file.with_suffix(".json.bak"))
    cal_file.write_text(json.dumps(cal_all, indent=4))
    print(f"\n✅ 已写回 {cal_file}")
    print(f"   （旧文件备份在 {cal_file.with_suffix('.json.bak')}）")
    print("\n== 新旧对比 ==")
    print(f"   {motor:<15} homing: {old['homing_offset']} → {new_homing}")
    print(f"   {motor:<15} range : [{old['range_min']}, {old['range_max']}] → [{new_min}, {new_max}]")
    print("\n提醒: 若另一臂配对使用，同关节也建议同姿势标一次，保证两臂 0° 对应。")


if __name__ == "__main__":
    main()
