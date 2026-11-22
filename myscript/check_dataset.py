"""
数据集质量批次检查（对应采集清单 §五）
用法: cd /mnt/nvme01/workplace/lerobot && PYTHONPATH= uv run python myscript/check_dataset.py zihao/so101_task1_20261121_165448
检查: 每条帧数/时长、时间戳断档（掉帧）、各关节活动量、视频文件完整性
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HF_HOME = Path.home() / ".cache/huggingface/lerobot"
FPS_TOL = 0.25      # 相邻帧间隔超过 1/fps*(1+容差) 视为断档
JOINT_MIN_SPREAD = 2.0  # 全数据集某关节活动量低于此(度)视为可疑

JOINTS = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    repo = sys.argv[1].lstrip("/")
    root = HF_HOME / repo
    if not (root / "meta/info.json").exists():
        print(f"❌ 找不到 {root}")
        sys.exit(1)

    info = json.load(open(root / "meta/info.json"))
    fps = info["fps"]
    total = info["total_episodes"]
    print(f"== {repo} ==")
    print(f"fps={fps}  total_episodes={total}  size={sum(p.stat().st_size for p in root.rglob('*') if p.is_file())/1e6:.0f}MB\n")

    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    state = np.stack(df["observation.state"].to_numpy())
    action = np.stack(df["action"].to_numpy())
    df["state_spread"] = 0.0

    print(f"{'ep':>3} {'帧数':>5} {'时长s':>6} {'最大帧间隔s':>9}  各关节活动量(度: pan/lift/elbow/wflex/wroll/grip)")
    warns = []
    for ep, idx in df.groupby("episode_index").groups.items():
        sub = df.loc[idx]
        n = len(sub)
        ts = sub["timestamp"].to_numpy()
        gaps = np.diff(ts)
        max_gap = gaps.max() if len(gaps) else 0.0
        spread = state[idx].max(axis=0) - state[idx].min(axis=0)
        sp = " ".join(f"{s:5.1f}" for s in spread)
        flag = ""
        w = None
        w = None
        if n < fps * 2:
            flag, w = " ⚠️太短", f"ep{ep} 只有 {n} 帧（{n/fps:.1f}s），疑似半截"
        if max_gap > (1 / fps) * (1 + FPS_TOL):
            flag, w = flag + " ⚠️断档", f"ep{ep} 最大帧间隔 {max_gap:.3f}s（应 ~{1/fps:.3f}s）"
        if (spread < JOINT_MIN_SPREAD).all():
            flag, w = flag + " ⚠️全静止", f"ep{ep} 全程所有关节零活动（{n/fps:.1f}s），疑似废条/误触发，建议删除"
        print(f"{ep:>3} {n:>5} {n/fps:>6.1f} {max_gap:>9.3f}  {sp}{flag}")
        if w:
            warns.append(w)

    print("\n== 全数据集关节活动量（度）==")
    ds = state.max(axis=0) - state.min(axis=0)
    for j, s in zip(JOINTS, ds):
        mark = " ⚠️ 基本没动，查电机/标定" if s < JOINT_MIN_SPREAD else ""
        print(f"  {j:<6} {s:7.1f}{mark}")
    if (ds < JOINT_MIN_SPREAD).any():
        warns.append("存在几乎不动的关节")

    print("\n== 视频文件 ==")
    # 交叉核对：episodes 元数据里引用的每个 (相机, file_index) 视频文件都必须存在
    # （2026-11-21 实测丢过 third_person/file-002.mp4，元数据指向不存在的文件——
    #   只遍历现存文件查完整性会漏判这类缺失，必须反向核对）
    ep_files = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    if ep_files:
        ep_df = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
        for col in [c for c in ep_df.columns if c.startswith("videos/") and c.endswith("/file_index")]:
            feature = col[: -len("/file_index")]  # e.g. videos/observation.images.front
            cam = feature.split(".")[-1]
            for fidx in sorted(ep_df[col].unique()):
                expect = root / feature / f"chunk-{int(ep_df[ep_df[col]==fidx][feature+'/chunk_index'].iloc[0]):03d}" / f"file-{int(fidx):03d}.mp4"
                if not expect.exists():
                    print(f"  ❌ 缺文件（元数据引用但不存在）: {expect.relative_to(root)}")
                    warns.append(f"视频缺失: {cam}/file-{int(fidx):03d}.mp4（覆盖 ep{int(ep_df[ep_df[col]==fidx]['episode_index'].min())}~{int(ep_df[ep_df[col]==fidx]['episode_index'].max())}）")
    for v in sorted((root / "videos").rglob("*.mp4")):
        try:
            dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                  "-of", "csv=p=0", str(v)], capture_output=True, text=True).stdout.strip()
            print(f"  {v.relative_to(root)}  {v.stat().st_size/1e6:5.1f}MB  {float(dur):6.1f}s")
        except Exception as e:
            print(f"  {v.relative_to(root)}  ⚠️ ffprobe 失败: {e}")
            warns.append(f"视频损坏: {v.name}")

    print("\n== 结论 ==")
    if warns:
        for w in warns:
            print(f"  ⚠️ {w}")
    else:
        print(f"  ✅ {total} 条全部健康：无断档、无半截、各关节活动正常。可继续采集。")


if __name__ == "__main__":
    main()
