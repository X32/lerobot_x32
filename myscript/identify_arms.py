"""身份确认：不上力矩，看哪个口的 1 号电机在被拧动（拧主臂底座）
运行: cd /mnt/nvme01/workplace/lerobot && PYTHONPATH= uv run python myscript/identify_arms.py
"""
import time
from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

buses = {}
for port in ("/dev/ttyACM0", "/dev/ttyACM1"):
    bus = SerialMotorsBus(
        port=port,
        motors={"m1": Motor(1, "sts3215", MotorNormMode.DEGREES)},
    )
    bus.connect()
    buses[port] = bus

print("⏳ 3 秒后开始采样 6 秒，请拧主臂底座…")
time.sleep(3)
samples = {p: [] for p in buses}
t0 = time.perf_counter()
last_print = 0.0
while time.perf_counter() - t0 < 6:
    for p, b in buses.items():
        samples[p].append(b.read("Present_Position", "m1", normalize=False))
    now = time.perf_counter() - t0
    if now - last_print >= 2.0:
        last_print = now
        print(f"[{now:4.1f}s] " + " | ".join(
            f"{p[-5:]}={samples[p][-1]:7.1f}" for p in buses))
    time.sleep(0.05)

print("\n== 判定 ==")
for p, s in samples.items():
    span = max(s) - min(s)
    print(f"{p}: 变化幅度 {span:6.1f} 刻度 {'← 在动（这是主臂）' if span > 10 else ''}")

for b in buses.values():
    b.disconnect()
