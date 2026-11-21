"""探测两个串口各自的电机应答（只 ping，不上力矩）
运行: cd /mnt/nvme01/workplace/lerobot && PYTHONPATH= uv run python myscript/probe_ports.py
"""
from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

for port in ("/dev/ttyACM0", "/dev/ttyACM1"):
    bus = SerialMotorsBus(
        port=port,
        motors={n: Motor(i, "sts3215", MotorNormMode.DEGREES)
                for i, n in enumerate(["a", "b", "c", "d", "e", "f"], start=1)},
    )
    try:
        bus.connect()
        print(f"{port}: 连接成功，6 个电机全部应答")
        bus.disconnect()
    except RuntimeError as e:
        lines = str(e).splitlines()
        print(f"{port}: 失败 -> " + " | ".join(lines[-2:]))
