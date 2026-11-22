#!/usr/bin/env python
"""State-only BC 基线：仅用关节状态（无图像）的 MLP 行为克隆，预测与 ACT 相同的
100 步动作 chunk（对齐对比协议：同数据、同 chunk、同 L1 损失）。

作用：给 ACT 一个"图像到底值多少"的基线。若 ACT 成功率不显著高于本基线，
说明视觉没被利用（或数据不够），这比任何 loss 都有说服力。

用法（仓库根）：
    PYTHONPATH= nohup uv run --no-sync python myscript/train_bc_state.py \
        --steps 20000 --output-dir outputs/train/bc_state_task1 \
        > outputs/train_bc_state.log 2>&1 &

数据：zihao/so101_task1_train（41 条，无图像请求 → DataLoader 不解码视频，很快）
验证：zihao/so101_task1_val（10 条），每 eval_freq 步算一次 val L1，存 best.pt。
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

TRAIN_REPO_ID = "zihao/so101_task1_train"
TRAIN_ROOT = "/mnt/nvme_data/huggingface_cache/lerobot/zihao/so101_task1_split/train"
VAL_REPO_ID = "zihao/so101_task1_val"
VAL_ROOT = "/mnt/nvme_data/huggingface_cache/lerobot/zihao/so101_task1_split/val"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def make_loader(repo_id: str, root: str, chunk: int, batch_size: int, shuffle: bool, workers: int):
    delta_ts = {
        "observation.state": [0],
        "action": list(range(chunk)),
    }
    ds = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_ts)
    return ds, torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=shuffle
    )


class StateBC(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, chunk: int, hidden: int = 512):
        super().__init__()
        self.action_dim = action_dim
        self.chunk = chunk
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, action_dim * chunk),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state: (B, state_dim) → (B, chunk, action_dim)
        out = self.net(state)
        return out.view(-1, self.chunk, self.action_dim)


def batch_to_tensors(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    state = batch["observation.state"]  # (B, state_dim)（n_obs_steps=1，无时间维）
    if state.dim() == 3:
        state = state[:, 0] if state.shape[1] == 1 else state.flatten(1)
    action = batch["action"]  # (B, chunk, action_dim)
    if action.dim() == 4:
        action = action[:, 0]
    return state.to(device), action.to(device)


@torch.no_grad()
def evaluate(model, loader, device, state_norm, action_norm) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        state, action = batch_to_tensors(batch, device)
        state_n = (state - state_norm[0]) / state_norm[1]
        pred = model(state_n) * action_norm[1] + action_norm[0]
        total += torch.abs(pred - action).mean().item()
        n += 1
    model.train()
    return total / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--chunk", type=int, default=100, help="与 ACT chunk_size 对齐")
    ap.add_argument("--eval-freq", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--output-dir", default="outputs/train/bc_state_task1")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("加载数据（无图像请求，不解码视频）...")
    train_ds, train_loader = make_loader(TRAIN_REPO_ID, TRAIN_ROOT, args.chunk, args.batch_size, True, args.num_workers)
    val_ds, val_loader = make_loader(VAL_REPO_ID, VAL_ROOT, args.chunk, args.batch_size, False, args.num_workers)
    state_dim = train_ds.meta.features["observation.state"]["shape"][0]
    action_dim = train_ds.meta.features["action"]["shape"][0]
    logger.info("train %d eps / val %d eps, state_dim=%d action_dim=%d chunk=%d",
                train_ds.meta.total_episodes, val_ds.meta.total_episodes, state_dim, action_dim, args.chunk)

    # MEAN_STD 归一化统计（来自训练集，与 ACT 协议一致）
    stats = train_ds.meta.stats
    state_mean = torch.tensor(stats["observation.state"]["mean"], dtype=torch.float32, device=device)
    state_std = torch.tensor(stats["observation.state"]["std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    action_mean = torch.tensor(stats["action"]["mean"], dtype=torch.float32, device=device)
    action_std = torch.tensor(stats["action"]["std"], dtype=torch.float32, device=device).clamp_min(1e-6)

    model = StateBC(state_dim, action_dim, args.chunk).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("模型参数量: %d (%.2fM)", n_params, n_params / 1e6)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_step = float("inf"), 0
    history = []
    t0 = time.time()
    step = 0
    logger.info("开始训练: %d 步", args.steps)
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            state, action = batch_to_tensors(batch, device)
            state_n = (state - state_mean) / state_std
            action_n = (action - action_mean) / action_std
            pred = model(state_n)
            loss = torch.abs(pred - action_n).mean()  # L1，与 ACT 主损失同族
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0:
                logger.info("step:%d loss:%.4f lr:%.1e", step, loss.item(), args.lr)
            if step % args.eval_freq == 0:
                val_l1 = evaluate(model, val_loader, device, (state_mean, state_std), (action_mean, action_std))
                history.append({"step": step, "val_l1_raw": val_l1})
                logger.info("step:%d val_l1(raw units):%.4f", step, val_l1)
                if val_l1 < best_val:
                    best_val, best_step = val_l1, step
                    torch.save({"model": model.state_dict(), "step": step, "val_l1_raw": val_l1,
                                "state_mean": state_mean.cpu(), "state_std": state_std.cpu(),
                                "action_mean": action_mean.cpu(), "action_std": action_std.cpu(),
                                "chunk": args.chunk}, out_dir / "best.pt")
                if step % (args.eval_freq * 5) == 0:
                    with open(out_dir / "metrics.json", "w") as f:
                        json.dump({"history": history, "best_val_l1_raw": best_val,
                                   "best_step": best_step, "args": vars(args)}, f, indent=2)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"history": history, "best_val_l1_raw": best_val, "best_step": best_step,
                   "args": vars(args)}, f, indent=2)
    logger.info("完成: best val_l1(raw)=%.4f @ step %d, 用时 %.1f 分钟, 产物 %s",
                best_val, best_step, (time.time() - t0) / 60, out_dir / "best.pt")


if __name__ == "__main__":
    main()
