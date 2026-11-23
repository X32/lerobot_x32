#!/usr/bin/env bash
# 学习曲线三连跑：s41 → s25 → s10 串行（GPU 串行，一个跑完自动发下一个）。
# 断点安全：每个 run 若已有 checkpoint 目录则自动加 --resume=true，
#   断电/误关机后重新执行本脚本即可从最后 checkpoint 继续，不会从头重训。
# 用法: cd /mnt/nvme01/workplace/lerobot && PYTHONPATH= nohup bash myscript/train_curve_chain.sh > outputs/chain.log 2>&1 &
# 进度: tr '\r' '\n' < outputs/train_act_s41_seed1000.log | grep step: | tail -1
# 验证集列表（10 条，≡0 mod 5，横跨两个采集批次）固定放 episodes 最后——
#   lerobot-train 的 eval_split 取"最后 N 条"，顺序即语义（docs/训练科学化方案.md §2）。
set -u
cd /mnt/nvme01/workplace/lerobot

VAL="0,5,10,15,20,25,30,35,40,45"
TRAIN41="1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24,26,27,28,29,31,32,33,34,36,37,38,39,41,42,43,44,46,47,48,49,50"
TRAIN10="1,6,11,16,21,26,31,36,41,46"
TRAIN25="1,6,11,16,21,26,31,36,41,46,2,3,7,8,12,13,17,18,22,23,27,28,32,33,37"

run_one() {
    local name=$1 eps=$2 esplit=$3
    local dir="outputs/train/act_task1_${name}"
    local extra=""
    [ -d "$dir/checkpoints" ] && extra="--resume=true"
    echo "[$(date '+%F %T')] === 启动 ${name} (output_dir=$dir, resume=${extra:-false}) ==="
    PYTHONPATH= uv run --no-sync lerobot-train \
        --dataset.repo_id=zihao/so101_task1 \
        --dataset.root=/mnt/nvme_data/huggingface_cache/lerobot/zihao/so101_task1 \
        --dataset.episodes="[${eps},${VAL}]" \
        --dataset.eval_split=$esplit \
        --policy.type=act --policy.push_to_hub=false \
        --output_dir=$dir \
        --steps=50000 --batch_size=8 --save_freq=10000 --log_freq=200 \
        --eval_steps=2500 --max_eval_samples=2048 \
        --num_workers=4 --seed=1000 $extra \
        > "outputs/train_act_${name}.log" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] === ${name} 退出 rc=$rc ==="
}

run_one s41_seed1000 "$TRAIN41" 0.196   # 41 训 + 10 验, N=51
run_one s25_seed1000 "$TRAIN25" 0.285   # 25 训 + 10 验, N=35
run_one s10_seed1000 "$TRAIN10" 0.500   # 10 训 + 10 验, N=20

echo "[$(date '+%F %T')] === 三连跑全部结束 ==="
