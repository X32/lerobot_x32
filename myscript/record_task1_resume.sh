#!/usr/bin/env bash
# 续录 so101_task1：本次再录 38 条（已有 12 条，50 - 12 = 38，录完正好 50 条）
# ⚠️ resume 模式下 num_episodes 是"本次新录几条"，不是总条数——已录条数变了要改这里
# 用法: cd /mnt/nvme01/workplace/lerobot && bash myscript/record_task1_resume.sh
# 键盘: n 或 → = 保存进下一条 | r 或 ← = 重录本条 | q 或 Esc = 收工
set -e
cd /mnt/nvme01/workplace/lerobot
uv run lerobot-record \
    --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=zihao_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, third_person: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
    --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=zihao_leader_arm \
    --dataset.repo_id=zihao/so101_task1 \
    --dataset.single_task="拿起桌上的积木放进盒子" \
    --dataset.num_episodes=38 --dataset.episode_time_s=30 --dataset.reset_time_s=10 \
    --dataset.push_to_hub=false \
    --resume=true \
    --dataset.root=/mnt/nvme_data/huggingface_cache/lerobot/zihao/so101_task1
