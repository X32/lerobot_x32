#!/usr/bin/env bash
# 交叉备份：数据集/标定 → nvme01，仓库快照 → nvme_data（互为异盘副本）
# 用法: cd /mnt/nvme01/workplace/lerobot && bash myscript/backup.sh
# 建议: 每个采集日/重要节点跑一次；备份保留最近 3 份，旧的自动清
set -euo pipefail
cd /mnt/nvme01/workplace/lerobot

DATE=$(date +%Y%m%d_%H%M)
DATA_SRC=/mnt/nvme_data/huggingface_cache/lerobot
DATA_DST=/mnt/nvme01/lerobot_backups        # 数据集+标定 备份区（在另一块盘）
REPO_DST=/mnt/nvme_data/lerobot_repo_backups # 仓库快照 备份区（在另一块盘）
KEEP=3

mkdir -p "$DATA_DST" "$REPO_DST"

echo "[1/2] 备份训练数据 + 标定文件 → $DATA_DST"
tar -czf "$DATA_DST/data_${DATE}.tar.gz" -C "$DATA_SRC" \
    zihao/so101_task1 calibration

echo "[2/2] 备份仓库快照（源码+文档+myscript，排除 .venv/.git）→ $REPO_DST"
tar --exclude='.venv' --exclude='.git' --exclude='*.egg-info' --exclude='__pycache__' \
    -czf "$REPO_DST/repo_${DATE}.tar.gz" -C /mnt/nvme01/workplace lerobot

echo "清理旧备份（各保留最近 $KEEP 份）"
ls -1t "$DATA_DST"/data_*.tar.gz  2>/dev/null | tail -n +$((KEEP+1)) | xargs -r rm --
ls -1t "$REPO_DST"/repo_*.tar.gz  2>/dev/null | tail -n +$((KEEP+1)) | xargs -r rm --

echo
echo "== 完成，现有备份 =="
ls -lht "$DATA_DST" | head -4
ls -lht "$REPO_DST" | head -4
