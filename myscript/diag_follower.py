"""SO-101 从臂总线诊断（只读 + 短暂力矩测试）
运行: cd /mnt/nvme01/workplace/lerobot && PYTHONPATH= uv run python myscript/diag_follower.py
"""
import time
from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

PORT = "/dev/ttyACM0"
bus = SerialMotorsBus(
    port=PORT,
    motors={
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    },
)
bus.connect()
names = list(bus.motors)
print("已连接:", names)

print("\n== 1. 通信稳定性：30 轮 sync_read(Present_Position) ==")
fails = {}
ok = 0
t0 = time.perf_counter()
for _ in range(30):
    try:
        pos = bus.sync_read("Present_Position", normalize=False)
        ok += 1
        for n in names:
            if pos is None or n not in pos or pos[n] is None:
                fails[n] = fails.get(n, 0) + 1
    except Exception as e:
        msg = str(e)
        hit = False
        for n in names:
            if n in msg:
                fails[n] = fails.get(n, 0) + 1
                hit = True
                break
        if not hit:
            fails["_全局"] = fails.get("_全局", 0) + 1
    time.sleep(0.005)
print(f"成功 {ok}/30 轮, 耗时 {time.perf_counter() - t0:.2f}s")
print("各电机失败次数:", fails if fails else "无")

print("\n== 2. 电压/温度/负载（无力矩） ==")
for reg in ("Present_Voltage", "Present_Temperature", "Present_Load"):
    try:
        vals = bus.sync_read(reg, normalize=False)
        print(f"{reg}: " + ", ".join(f"{n}={vals.get(n)}" for n in names))
    except Exception as e:
        print(f"{reg}: 读取失败 -> {type(e).__name__}: {e}")

print("\n⏳ 3 秒后开始力矩测试…")
time.sleep(3)
print("== 3. 逐个电机使能力矩 0.6s 再释放 ==")
for n in names:
    try:
        bus.enable_torque([n], num_retry=0)
        time.sleep(0.6)
        bus.disable_torque([n], num_retry=0)
        print(f"{n}: OK")
    except Exception as e:
        print(f"{n}: 失败 -> {type(e).__name__}: {e}")
        try:
            bus.disable_torque(num_retry=0)
        except Exception:
            pass

print("\n== 4. 全部电机同时使能（复现报错场景） ==")
try:
    bus.enable_torque(names, num_retry=0)
    time.sleep(1.0)
    try:
        v = bus.sync_read("Present_Voltage", normalize=False)
        print("带载电压:", {k: v.get(k) for k in names})
    except Exception as e:
        print("带载电压读取失败:", e)
    bus.disable_torque(names, num_retry=0)
    print("全部使能: OK")
except Exception as e:
    print(f"失败 -> {type(e).__name__}: {e}")
finally:
    try:
        bus.disable_torque(num_retry=0)
    except Exception:
        pass

bus.disconnect()
print("\n诊断结束")
