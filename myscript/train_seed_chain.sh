#!/usr/bin/env bash
# 最终配置 seed 重复跑：seed2000 → seed3000 串行（seed1000 复用 50k 跑的 10k/20k/30k
# checkpoint，恒定 lr 下与 30k 独立跑逐样本一致，无需重训）。
# 配置 = 41 条训练 + 10 条验证（episodes 置尾）+ 30k 步（s41 曲线证 20k 后无收益，留余量）。
# 断点安全：已有 checkpoint 自动 --resume=true，断电重跑本脚本即续。
# 用法: cd /mnt/nvme01/workplace/lerobot && PYTHONPATH= nohup bash myscript/train_seed_chain.sh > outputs/seed_chain.log 2>&1 &
set -u
cd /mnt/nvme01/workplace/lerobot

VAL="0,5,10,15,20,25,30,35,40,45"
TRAIN41="1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24,26,27,28,29,31,32,33,34,36,37,38,39,41,42,43,44,46,47,48,49,50"

run_one() {
    local seed=$1
    local dir="outputs/train/act_task1_s41_seed${seed}"
    local extra=""
    [ -d "$dir/checkpoints" ] && extra="--resume=true"
    echo "[$(date '+%F %T')] === 启动 seed${seed} (dir=$dir, resume=${extra:-false}) ==="
    PYTHONPATH= uv run --no-sync lerobot-train \
        --dataset.repo_id=zihao/so101_task1 \
        --dataset.root=/mnt/nvme_data/huggingface_cache/lerobot/zihao/so101_task1 \
        --dataset.episodes="[${TRAIN41},${VAL}]" \
        --dataset.eval_split=0.196 \
        --policy.type=act --policy.push_to_hub=false \
        --output_dir=$dir \
        --steps=30000 --batch_size=8 --save_freq=10000 --log_freq=200 \
        --eval_steps=2500 --max_eval_samples=2048 \
        --num_workers=4 --seed=${seed} $extra \
        > "outputs/train_act_s41_seed${seed}.log" 2>&1
    echo "[$(date '+%F %T')] === seed${seed} 退出 rc=$? ==="
}

run_one 2000
run_one 3000
echo "[$(date '+%F %T')] === seed 双跑全部结束 ==="
