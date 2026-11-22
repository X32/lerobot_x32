#!/usr/bin/env bash
# ACT 训练：zihao/so101_task1（51 条「拿起桌上的积木放进盒子」）
# 用法: cd /mnt/nvme01/workplace/lerobot && bash myscript/train_act_task1.sh
# ⚠️ 必须用 uv run --no-sync：venv 里的 torch 2.8.0+cu126 是为 1080 Ti 特意装的，
#    裸 uv run 会按 lockfile 同步回 cu132（驱动不认，CUDA 不可用）
# 预计: ~0.3s/步 × 100000 步 ≈ 8.5 小时；每 20000 步存一个 checkpoint
# 进度: tail -f outputs/train_act.log（手册: docs/训练手册.md §1.4）
# 中断: pkill -f lerobot-train（重跑本脚本加 --resume=true 从最后 checkpoint 续训）
set -e
cd /mnt/nvme01/workplace/lerobot

PYTHONPATH= nohup uv run --no-sync lerobot-train \
    --dataset.repo_id=zihao/so101_task1 \
    --dataset.root=/mnt/nvme_data/huggingface_cache/lerobot/zihao/so101_task1 \
    --policy.type=act --policy.push_to_hub=false \
    --output_dir=outputs/train/act_so101_task1 \
    --steps=100000 --batch_size=8 --save_freq=20000 --log_freq=200 \
    --num_workers=4 \
    > outputs/train_act.log 2>&1 &

echo "训练已后台启动 PID=$!"
echo "看进度: tail -f /mnt/nvme01/workplace/lerobot/outputs/train_act.log"
