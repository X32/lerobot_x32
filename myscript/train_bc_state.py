#!/usr/bin/env python
"""State-only BC 基线：仅用关节状态（无图像）的 MLP 行为克隆，预测与 ACT 相同的
100 步动作 chunk（对齐对比协议：同数据、同 chunk、同 masked-L1 损失）。

作用：回答"不用视觉能不能做"。若 ACT 真机成功率不显著高于本基线，说明这套
41 条数据上视觉没有带来收益。注意它**不能**单独证明"视觉没被利用"——架构、
chunking、VAE 都与 ACT 不同；视觉是否被使用，另由推理期图像置零消融回答。

同口径参照：训练与评估时额外计算"持久性基线"（100 步全预测当前关节状态）的
masked L1，作为"模型到底学到了什么"的下界参照。

用法（仓库根）：
    PYTHONPATH= nohup uv run --no-sync python myscript/train_bc_state.py \
        --steps 20000 --output-dir outputs/train/bc_state_task1 \
        > outputs/train_bc_state.log 2>&1 &

数据：zihao/so101_task1_train（41 条）。本脚本会从 meta.features 移除视频特征，
     DataLoader 不做视频解码（实测 0.004s/批 vs 0.24s/批）。
验证：zihao/so101_task1_val（10 条），每 eval_freq 步算一次 val L1，存 best.pt。
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset

TRAIN_REPO_ID = "zihao/so101_task1_train"
TRAIN_ROOT = "/mnt/nvme_data/huggingface_cache/lerobot/zihao/so101_task1_split/train"
VAL_REPO_ID = "zihao/so101_task1_val"
VAL_ROOT = "/mnt/nvme_data/huggingface_cache/lerobot/zihao/so101_task1_split/val"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def make_loader(
    repo_id: str,
    root: str,
    chunk: int,
    batch_size: int,
    shuffle: bool,
    workers: int,
    drop_video: bool = True,
):
    # delta_timestamps 的单位是「秒」，内部再 × fps 转帧；传帧号会让窗口跨 30 倍时长。
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta_ts = {
        "observation.state": [0],
        "action": [i / meta.fps for i in range(chunk)],
    }
    ds = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_ts)
    if drop_video:
        # 本基线不使用图像。LeRobotDataset 默认会解码所有视频特征（实测 0.24s/批 → 0.004s/批），
        # 移除 video/image 特征后 video_keys 为空，DataLoader 不再碰解码器。
        for key in list(ds.meta.features):
            if ds.meta.features[key]["dtype"] in ("video", "image"):
                ds.meta.features.pop(key)
    reader = getattr(ds, "reader", None)
    got = reader.delta_indices.get("action") if reader is not None and reader.delta_indices else None
    if got is not None and list(got) != list(range(chunk)):
        raise RuntimeError(f"动作窗口帧索引异常（应为 0..{chunk - 1}）：{list(got)[:6]}...")
    return ds, torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=shuffle
    )


def masked_l1(pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor | None) -> torch.Tensor:
    """与 ACT 主损失同一口径：尾部 padding 位置不计入。"""
    err = (pred - target).abs()
    if is_pad is None:
        return err.mean()
    valid = (~is_pad).unsqueeze(-1)
    return (err * valid).sum() / (valid.sum() * err.shape[-1]).clamp_min(1)


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


def batch_to_tensors(batch: dict, device: torch.device):
    state = batch["observation.state"]  # (B, state_dim)（n_obs_steps=1，无时间维）
    if state.dim() == 3:
        state = state[:, 0] if state.shape[1] == 1 else state.flatten(1)
    action = batch["action"]  # (B, chunk, action_dim)
    if action.dim() == 4:
        action = action[:, 0]
    is_pad = batch.get("action_is_pad")
    if is_pad is not None:
        is_pad = is_pad.to(device)
    return state.to(device), action.to(device), is_pad


@torch.no_grad()
def evaluate(model, loader, device, state_norm, action_norm) -> tuple[float, float]:
    """返回 (归一化空间的 masked L1, 同口径的持久性参照 L1)。"""
    model.eval()
    total, total_persist, n = 0.0, 0.0, 0
    for batch in loader:
        state, action, is_pad = batch_to_tensors(batch, device)
        state_n = (state - state_norm[0]) / state_norm[1]
        pred = model(state_n) * action_norm[1] + action_norm[0]
        total += masked_l1(pred, action, is_pad).item()
        persist = state.unsqueeze(1).expand(-1, action.shape[1], -1)
        total_persist += masked_l1(persist, action, is_pad).item()
        n += 1
    model.train()
    return total / max(n, 1), total_persist / max(n, 1)


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
            state, action, is_pad = batch_to_tensors(batch, device)
            state_n = (state - state_mean) / state_std
            action_n = (action - action_mean) / action_std
            pred = model(state_n)
            loss = masked_l1(pred, action_n, is_pad)  # L1，与 ACT 主损失同口径（忽略 padding）
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0:
                logger.info("step:%d loss:%.4f lr:%.1e", step, loss.item(), args.lr)
            if step % args.eval_freq == 0:
                val_l1, val_persist = evaluate(
                    model, val_loader, device, (state_mean, state_std), (action_mean, action_std)
                )
                history.append({"step": step, "val_l1_raw": val_l1, "val_persist_raw": val_persist})
                logger.info("step:%d val_l1(raw units):%.4f | 持久性参照(vs val_l1):%.4f", step, val_l1, val_persist)
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
