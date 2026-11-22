#!/usr/bin/env python
"""离线验证集评估：对指定 checkpoint 在 so101_task1_val 上计算 action L1 / eval loss。

用途：checkpoint 选型（按验证 L1，不按训练 loss）。可评估任何 ACT checkpoint，
包括 lerobot-train 训出的 outputs/train/*/checkpoints/<step>/pretrained_model。

用法（在仓库根，注意 --no-sync 与 PYTHONPATH= 纪律）：
    PYTHONPATH= uv run --no-sync python myscript/eval_val_l1.py \
        --ckpt outputs/train/act_so101_task1/checkpoints/100000/pretrained_model

    # 只跑前 64 个 batch（快速比较）：
    PYTHONPATH= uv run --no-sync python myscript/eval_val_l1.py --ckpt <路径> --max-batches 64

指标口径：
- eval_loss   : 策略自身损失（ACT = L1 + 10×KL，归一化空间）
- action_l1   : 归一化动作空间的平均绝对误差（与本 batch 内 action 维度平均）
- 跨运行比较注意：不同数据量训练的 run 归一化统计不同（41/51 条算出的 mean/std
  略有差异），action_l1 跨 run 只做粗对比；最终排序以真机成功率为准。
"""

import argparse
import logging
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.act.modeling_act import ACTPolicy

VAL_REPO_ID = "zihao/so101_task1_val"
VAL_ROOT = "/mnt/nvme_data/huggingface_cache/lerobot/zihao/so101_task1_split/val"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pretrained_model 目录路径")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = 跑满整个验证集")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("加载策略: %s", ckpt)
    policy = ACTPolicy.from_pretrained(ckpt)
    policy.to(device)
    policy.eval()

    logger.info("加载验证集: %s (%s)", VAL_REPO_ID, VAL_ROOT)
    # 两步构造：先建数据集（拿 delta_timestamps 需要的 meta），再按策略配置重开
    probe = LeRobotDataset(VAL_REPO_ID, root=VAL_ROOT)
    delta_ts = resolve_delta_timestamps(policy.config, probe.meta, {})
    probe = None  # 释放，避免两个句柄
    dataset = LeRobotDataset(VAL_REPO_ID, root=VAL_ROOT, delta_timestamps=delta_ts)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # 从 checkpoint 目录加载配套的 preprocessor（含训练时的归一化统计）
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, _postprocessor = make_pre_post_processors(policy.config, pretrained_path=ckpt)
    preprocessor.to(device) if hasattr(preprocessor, "to") else None

    camera_keys = dataset.meta.camera_keys
    sum_loss, sum_l1, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            for cam_key in camera_keys:
                if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
            batch = {k: v.to(device) for k, v in batch.items()}
            batch = preprocessor(batch)
            loss, output = policy(batch)
            pred = output["action"] if isinstance(output, dict) else output.action
            target = batch["action"]
            l1 = torch.abs(pred - target).mean().item()
            sum_loss += loss.item()
            sum_l1 += l1
            n_batches += 1
            if args.max_batches and n_batches >= args.max_batches:
                break

    logger.info("=" * 60)
    logger.info("checkpoint : %s", ckpt)
    logger.info("batches    : %d (batch_size=%d)", n_batches, args.batch_size)
    logger.info("eval_loss  : %.4f  (策略自身损失, 归一化空间)", sum_loss / max(n_batches, 1))
    logger.info("action_l1  : %.4f  (归一化动作空间, 选型主指标)", sum_l1 / max(n_batches, 1))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
