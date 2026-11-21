# LeRobot 架构总览

> 本文是 `src/lerobot/` 的架构地图与模块职责速查，面向开发阅读；用户向操作指南见 [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)。
> 基于仓库 v0.7.1（2026-11 整理）。

## 分层架构

```
                        用户入口（CLI 命令）
┌────────────────────────────────────────────────────────────────┐
│ scripts/   lerobot-teleoperate · record · train · eval ·       │
│            replay · rollout · policy-server · env-server …     │
└────────┬──────────────────────────────────────────┬────────────┘
         │ draccus 解析 CLI → 配置对象                 │
         ▼                                          ▼
┌─────────────────┐                      ┌────────────────────┐
│    configs/     │  配置数据类            │     processor/     │
│ TrainPipeline   │  @register_subclass  │  ProcessorStep 注册 │
│ PreTrainedConfig│  多态注册表            │  观察/动作变换链     │
└────────┬────────┘                      └────────────────────┘
         │ 组装 ↓ 三大核心子系统
┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│    datasets/    │   │     policies/    │   │   envs/ + sim/   │
│ LeRobotDataset  │   │ PreTrainedPolicy │   │ EnvConfig 只存选项│
│ 分episode采样    │   │ = nn.Module+Hub  │   │ make_env() 工厂   │
│ parquet+mp4解码  │   │ factory 懒加载    │   │ → sim/integrations│
└────────▲────────┘   └────────┬─────────┘   └────────▲─────────┘
         │                     │ 推理动作               │ gym 接口
┌────────┴─────────────────────┴──────────────────────┴─────────┐
│                       硬件抽象层（HAL）                          │
│  robots/ 机器人本体      motors/ 电机总线（feetech/dynamixel…）  │
│  teleoperators/ 遥操作端  cameras/ 相机                          │
└────────────────────────────────────────────────────────────────┘
```

核心设计两条主线：

1. **一切皆配置类** —— draccus 注册表多态（`@register_subclass("name")`），CLI 参数直接解析成配置对象，再由工厂构造出策略/环境/机器人。
2. **HAL 隔离硬件** —— 上层从不直接碰串口/相机，全部走 `robots/` / `motors/` / `cameras/` / `teleoperators/` 抽象。

## 四条主线数据流

```
① 采集   teleoperators/ ──► robots/ ──lerobot-record──► datasets/（+processor 归一化）
② 训练   datasets/ ──► processor/ ──lerobot-train──► policies/（→ Hub 上传）
③ 评估   policies/ ──动作──► envs/+sim/ ──观测──► policies/（lerobot-eval 循环）
④ 部署   policies/ ──► rollout/ 控制循环 ──► inference/ 引擎 ──► robots/ 实机执行
```

## 子系统：RL 分布式训练栈（`rl/`，mypy 跳过）

```
┌──────────────── rl/  actor-learner 分离 ────────────────┐
│                                                        │
│  actor.py（环境采样rollout）──transitions 上行──► buffer.py │
│       ▲                                └─► learner.py   │
│  gym_manipulator/ 自建仿真环境              梯度更新/SAC类算法│
└──────────┬───────────────────────────────┬─────────────┘
           │ gRPC（transport/*_pb2）         │ 远程环境托管
           ▼                               ▼
      transport/（gRPC+Zenoh 通信）    env_server/（环境服务）
```

## 子系统：推理/服务栈

```
policies/（训练好的策略）
   ├─ 本地实机：inference/（调度·延迟控制）→ rollout/（controller/planner）→ robots/
   ├─ 远程推理：remote_inference/（协议·编解码）↔ lerobot-policy-server
   └─ 数据侧：  streaming/（episode 缓存/远程 mp4 读取）、env_server/、jobs/
```

## 支撑模块速查

| 模块 | 作用 |
|------|------|
| `utils/` `common/` | 设备/导入工具、训练通用逻辑、trackio 记录 |
| `optim/` | 优化器/调度器/EMA/梯度缩放的工厂封装 |
| `distributed/` | 多卡并行维度、checkpoint、accelerate 封装 |
| `rewards/` | 奖励模型（classifier、robometer、sarm、topreward） |
| `model/` | 运动学计算 |
| `transforms/` | 旧版数据变换（deprecated，被 processor/ 取代） |
| `annotations/` `data_processing/` `jobs/` | 标注管线、sarm 预处理、数据集管理任务 |
| `templates/` | Hub 模型卡模板 |
| `lerobot_types.py` `configs/types.py` | 核心类型别名与特征定义 |

## 相关文档

- [`AGENTS.md`](./AGENTS.md) — AI/开发快速指南（命令、架构要点、约定）
- [`docs/采集模块架构.md`](./docs/采集模块架构.md) — 采集环节展开：主循环、processor 链、数据 schema、v3 数据集格式
- [`docs/遥操作与采集操作手册.md`](./docs/遥操作与采集操作手册.md) — 本机实测命令手册：遥操作/单双相机采集、参数表、键盘控制、故障排查
- [`docs/模型选型指南.md`](./docs/模型选型指南.md) — ACT/SmolVLA/π0/LingBot 对比、选型决策树、边缘硬件速查、SO-101 落地路线
- [`AGENT_GUIDE.md`](./AGENT_GUIDE.md) — 用户向：SO-101 安装、录制、选策略、训练、评估
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — 测试分层（base/dataset/hardware/viz/gpu）与贡献流程
