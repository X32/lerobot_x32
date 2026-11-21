# SO-101 本机自测脚本说明

> 本机（SO-101 双臂）专用诊断/自测脚本集合，2026-11-18 整理。
> 操作主手册见 [`docs/遥操作与采集操作手册.md`](../docs/遥操作与采集操作手册.md)；本文只讲这些脚本。

## 运行前的三条通用规则

1. **统一运行方式**——都在仓库根下执行，`PYTHONPATH=` 前缀必带（屏蔽系统 ROS2 的 Python 环境污染，否则会报 lark 等莫名 import 错）：

   ```bash
   cd /mnt/nvme01/workplace/lerobot        # /home/x32/Desktop/nvme01 是同一个目录的快捷方式
   PYTHONPATH= uv run python myscript/<脚本> [参数]
   ```

2. **串口假设**——脚本默认 从臂=`/dev/ttyACM0`、主臂=`/dev/ttyACM1`。重新插拔 USB 后编号可能对调，先跑 `identify_arms.py` 确认（见下文"串口对调"）。
3. **上力矩的脚本要先摆姿势**——`compare_arms.py` / `elbow_sync_test.py` / `diag_follower.py` 会让**从臂上力矩**。开始前把主臂摆到和从臂大致一样的姿势（有 3 秒缓冲），避免从臂瞬间猛跳。

## 脚本清单

| 脚本 | 作用 | 是否上力矩 | 备注 |
|---|---|---|---|
| `probe_ports.py` | 探测两个串口各有几个电机应答 | ❌ | 插拔/接线排查第一步 |
| `identify_arms.py` | 串口身份识别：拧主臂底座，看哪个口读数在动 | ❌ | 3 秒后采样 6 秒 |
| `voltage_check.py` | 两臂电压/温度读数 | ❌ | 电压单位 0.1V（118=11.8V） |
| `verify_motor_order.py <串口>` | 电机 ID 顺序自测：逐关节手拧，自动判 ✓/✗ | ❌ | **两台臂各测一遍** |
| `calibrate_single_motor.py <臂> <电机>` | 单独校准一个电机的零点+行程（其余 5 个不动） | ❌ | 用法见脚本头注释；wrist_roll 自动跳过扫行程；自动备份 `.bak` |
| `release_check.py` | 释放从臂力矩 + 打印两臂当前姿势 | ❌（只释放） | 脚本异常中断后先跑这个 |
| `compare_arms.py` | 20 秒主从跟随 + 各关节偏差统计 | ✅ 从臂 | 自动结束并释放 |
| `elbow_sync_test.py` | 主从一致性分步自测（肘部重点，5 步引导） | ✅ 从臂 | 数据存本目录 `arm_sync_*.csv` |
| `diag_follower.py` | 从臂总线全面诊断：通信 30 轮/电压/温度/逐电机+全体力矩 | ✅ 从臂（第 3、4 部分） | 3 秒缓冲 |

所有脚本 Ctrl+C 都安全退出（自动释放力矩、保存已录数据）。

## 推荐工作流

```
刚插好线/怀疑接线       probe_ports.py → identify_arms.py → voltage_check.py
怀疑 ID/标定对象错位     verify_motor_order.py（两臂各一遍）→ 有 ✗ 按提示重跑 setup-motors + calibrate
排查关节不一致           verify_motor_order.py → elbow_sync_test.py → 把统计结果/CSV 拿来分析
脚本中断后臂锁着         release_check.py
偶发力矩超时            直接重跑；反复超时再 diag_follower.py
```

## 串口对调问题（插拔后 ttyACM 变了怎么办）

Linux 按插入顺序分配 `ttyACM0/1`，**重新插拔后可能对调**。板上 USB 序列号永远不变：

| 臂 | USB 序列号 | 稳定路径（永不变化） |
|---|---|---|
| 从臂 | `5CBC431912` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5CBC431912-if00` |
| 主臂 | `5CBC430736` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5CBC430736-if00` |

- 查当前对应：`ls -l /dev/serial/by-id/`（`->` 后面就是当前的 ttyACM 编号）
- 拿不准哪个是主臂：跑 `identify_arms.py`，拧主臂底座，**在动的那个口就是主臂**
- 一劳永逸：lerobot 命令里直接用上面的 by-id 路径填 `--robot.port` / `--teleop.port`

## 安全红线

- **主臂永远不上力矩**：正常遥操作/采集/本目录所有脚本都只把主臂当传感器读，不会给它上力矩。12V 电源误插主臂本身无害，但此状态下若运行"给主臂上力矩"的非常规脚本，电机会以 2 倍力矩猛甩——别写也别跑这种脚本。
- **从臂正常供电是 12V 电源适配器**（必须插）；主臂只插 USB。12V 若被挪去插主臂，从臂电机就欠电了，先恢复再排查别的。
- 标定文件在 `~/.cache/huggingface/lerobot/calibration/{robots/so_follower,teleoperators/so_leader}/`，脚本读的就是它们；重标定最简单的方式是连接时在提示处输入 **`c`**（详见主手册 §1.3）。
