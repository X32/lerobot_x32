"""
电机 ID 顺序自测脚本（无力矩，纯手拧 + 读数）
用法: cd /mnt/nvme01/workplace/lerobot && PYTHONPATH= uv run python myscript/verify_motor_order.py /dev/ttyACM0
      （参数是串口；从臂 ttyACM0，主臂 ttyACM1，两台都可以各测一遍）
跟着提示逐个关节拧动即可，每个关节自动判 ✓/✗。
"""
import sys
import time

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

# (电机名, 期望ID, 怎么拧这个关节)
JOINTS = [
    ("shoulder_pan",  1, "水平旋转底座（抓着整条臂左右扭）"),
    ("shoulder_lift", 2, "抬放整条大臂（前后大幅摆动）"),
    ("elbow_flex",    3, "弯曲/伸直肘部"),
    ("wrist_flex",    4, "上下摆腕（小臂前端上下点动）"),
    ("wrist_roll",    5, "拧转腕部滚转（绕小臂轴线拧）"),
    ("gripper",       6, "开合夹爪"),
]

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
print(f"连接 {port} …（期望电机 ID = 1~6）")
bus = SerialMotorsBus(
    port=port,
    motors={n: Motor(i, "sts3215", MotorNormMode.DEGREES)
            for i, (n, _, _) in enumerate(JOINTS, start=1)},
)
try:
    bus.connect()
except RuntimeError as e:
    print("\n❌ 连接握手失败——说明链上没有凑齐 ID 1~6（setup-motors 有电机没写上或写重号）。")
    print("错误详情：")
    print(e)
    sys.exit(1)

print("连接成功（ID 1~6 齐全）。")
print("=" * 60)
print("接下来 6 个关节挨个检测。每个关节：按回车 → 拧 4 秒 → 看判定。")
print("只拧提示的那个关节，其他关节扶稳别动。")
print("=" * 60)

results = []
for name, expect_id, how in JOINTS:
    print(f"\n┌─ 检测 [{name}]（期望 ID={expect_id}）")
    print(f"│  动作：{how}")
    input("└─ 摆好后按回车，开始 4 秒检测 > ")
    spans = {n: 0.0 for n, _, _ in JOINTS}
    first = bus.sync_read("Present_Position", normalize=False)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 4.0:
        cur = bus.sync_read("Present_Position", normalize=False)
        for n in spans:
            spans[n] = max(spans[n], abs(cur[n] - first[n]))
        time.sleep(0.03)
    detected = max(spans, key=spans.get)
    det_id = dict((n, i) for i, (n, _, _) in enumerate(JOINTS, start=1))[detected]
    ok = detected == name
    results.append((name, expect_id, detected, det_id, spans[name], ok))
    mark = "✅ 正确" if ok else f"❌ 不对！动的其实是 ID={det_id} ({detected})"
    print(f"   → 判定: {mark}   (变化幅度: " + ", ".join(f"{n}={s:.0f}" for n, s in spans.items() if s > 5) + ")")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
bad = [r for r in results if not r[5]]
for name, expect_id, detected, det_id, span, ok in results:
    print(f"  {'✅' if ok else '❌'} {name:<15} 期望ID={expect_id} 实际响应ID={det_id}")
if not bad:
    print("\n🎉 全部正确：ID 顺序、电机命名、标定对象都对得上。")
else:
    print(f"\n⚠️ 有 {len(bad)} 个不匹配。修复方法：")
    print("   重跑该臂的 lerobot-setup-motors（每次只把控制板连到提示的那一个电机），")
    print("   完成后重新 lerobot-calibrate（标定也依赖 ID 顺序，必须一起重做）。")

bus.disconnect()
print("\n完成")
