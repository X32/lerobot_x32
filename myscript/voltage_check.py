"""只读：两臂电压/温度检查（不上力矩）
运行: cd /mnt/nvme01/workplace/lerobot && PYTHONPATH= uv run python myscript/voltage_check.py
（电压单位 0.1V，如 118 = 11.8V；从臂应在 ~11.8-12V，主臂 ~5V 或 12V 都正常）
"""
from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

for label, port in (("主臂(ACM1)", "/dev/ttyACM1"), ("从臂(ACM0)", "/dev/ttyACM0")):
    bus = SerialMotorsBus(
        port=port,
        motors={m: Motor(i + 1, "sts3215", MotorNormMode.DEGREES) for i, m in enumerate(NAMES)},
    )
    bus.connect()
    v = bus.sync_read("Present_Voltage", normalize=False)
    t = bus.sync_read("Present_Temperature", normalize=False)
    print(f"\n== {label} ==")
    for m in NAMES:
        print(f"  {m:<15} 电压={v.get(m)}  温度={t.get(m)}°C")
    bus.disconnect()
