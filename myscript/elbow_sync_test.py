"""
主从臂一致性自测脚本（肘部重点）
用法: cd /mnt/nvme01/workplace/lerobot && PYTHONPATH= uv run python myscript/elbow_sync_test.py
按屏幕提示逐步操作即可，数据自动保存为 CSV（存到本目录），最后打印统计结果。
随时 Ctrl+C 中断：会安全释放力矩并保存已录数据。
"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus
from lerobot.motors.motors_bus import MotorCalibration

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FOLLOWER = ("/dev/ttyACM0", "/home/x32/.cache/huggingface/lerobot/calibration/robots/so_follower/zihao_follower_arm.json")
LEADER = ("/dev/ttyACM1", "/home/x32/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/zihao_leader_arm.json")


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


def countdown(sec):
    for i in range(sec, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)


def record_phase(name, dur, samples):
    """录 dur 秒：主臂→从臂镜像 + 采样，每秒打印一次两臂角度"""
    print(f"\n▶ [{name}] 现在开始，持续 {dur} 秒：")
    t0 = time.perf_counter()
    last_print = -1.0
    while True:
        t = time.perf_counter() - t0
        if t >= dur:
            break
        leader_deg = {m: leader.read("Present_Position", m) for m in NAMES}
        follower.sync_write("Goal_Position", leader_deg)
        present = follower.sync_read("Present_Position")
        samples.append({"phase": name, "t": round(t, 2),
                        **{f"l_{m}": round(leader_deg[m], 2) for m in NAMES},
                        **{f"f_{m}": round(present[m], 2) for m in NAMES}})
        if int(t) > last_print:
            last_print = int(t)
            print("  " + " | ".join(
                f"{m.split('_')[0][:4]}:{leader_deg[m]:6.1f}/{present[m]:6.1f}" for m in NAMES), flush=True)
        time.sleep(0.03)


def summarize(samples):
    print("\n===== 统计：主臂目标 vs 从臂实际（度；gripper 为 0-100）=====")
    phases = list(dict.fromkeys(s["phase"] for s in samples))
    for phase in phases:
        sub = [s for s in samples if s["phase"] == phase]
        print(f"\n[{phase}]  {len(sub)} 个采样")
        for m in NAMES:
            devs = [abs(s[f"l_{m}"] - s[f"f_{m}"]) for s in sub]
            print(f"  {m:<15} 平均偏差={sum(devs)/len(devs):6.2f}  最大偏差={max(devs):6.2f}")


print("=" * 60)
print("主从臂一致性自测（重点：肘部 elbow_flex）")
print("=" * 60)
print("连接两臂中…")
follower = load_bus(*FOLLOWER)
leader = load_bus(*LEADER)
print("连接成功。")

samples = []
try:
    print("""
┌─────────────────────────────────────────────────────────┐
│ 第 1 步：把主臂摆到和从臂当前姿势大致一样的位置          │
│ （从臂现在是松的；上力矩后会瞬间锁到主臂的姿势）         │
└─────────────────────────────────────────────────────────┘""")
    input("摆好后按回车 > ")
    print("3 秒后从臂上力矩并跟随主臂：")
    countdown(3)
    follower.enable_torque()

    record_phase("1_静止基线_别动", 5, samples)

    print("""
┌─────────────────────────────────────────────────────────┐
│ 第 2 步：只把主臂的肘部弯到大约一半行程                  │
│ 其他关节保持不动                                        │
└─────────────────────────────────────────────────────────┘""")
    input("摆好姿势后按回车，我将记录 4 秒 > ")
    record_phase("2_肘部半弯", 4, samples)

    print("""
┌─────────────────────────────────────────────────────────┐
│ 第 3 步：主臂肘部弯到最深（完全折叠），保持              │
└─────────────────────────────────────────────────────────┘""")
    input("摆好姿势后按回车，我将记录 4 秒 > ")
    record_phase("3_肘部全弯", 4, samples)

    print("""
┌─────────────────────────────────────────────────────────┐
│ 第 4 步：主臂肘部完全伸直，保持                         │
└─────────────────────────────────────────────────────────┘""")
    input("摆好姿势后按回车，我将记录 4 秒 > ")
    record_phase("4_肘部伸直", 4, samples)

    print("""
┌─────────────────────────────────────────────────────────┐
│ 第 5 步：随意动所有关节，模拟正常遥操作 10 秒            │
└─────────────────────────────────────────────────────────┘""")
    input("按回车开始 > ")
    record_phase("5_自由动作", 10, samples)

except KeyboardInterrupt:
    print("\n⚠️ 已中断")

finally:
    try:
        follower.disable_torque()
        print("\n✅ 从臂力矩已释放")
    except Exception as e:
        print(f"\n释放力矩出错（可能本来就未上力矩）: {e}")

    if samples:
        out = Path(__file__).resolve().parent / f"arm_sync_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
            w.writeheader()
            w.writerows(samples)
        print(f"📁 数据已保存: {out}  （{len(samples)} 个采样）")
        summarize(samples)

    follower.disconnect()
    leader.disconnect()
    print("\n完成。把上面的统计结果发给 Claude，或告知 CSV 路径由我来分析。")
