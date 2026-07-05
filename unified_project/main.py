"""统一训练入口。

读 config -> 建 dataset(在线)/encoder/loss/optim -> 训练循环 + 周期 eval + ckpt。
切换编码器只改 CONFIG["encoder"]["type"], 数据零改动。

Run (冒烟):
    python -m unified_project.main --iters 50 --eval_every 25 --eval_batches 2
Run (全量, 默认读 config):
    python -m unified_project.main
可用 --encoder 覆盖 config 里的 encoder.type (regression_cnn/classification_cnn/resfreq/rope)。
"""
import os
# 本机 OpenMP/MKL 冲突: ResFreq 的 Conv2d 多线程会段错误。必须在 import torch 前设置
# (环境问题, 非代码 bug; CNN/RoPE 的 1D/attention 不受影响, 但统一置单线程最稳)。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import CONFIG, grid_size, signal_dim as _signal_dim
from dataset import UnifiedBTTDataset, CachedUnifiedDataset
from win_encoder import build_encoder, encoder_forward, is_regression
from loss import build_loss
from eval import evaluate_all_snr, evaluate_at_snr


def make_train_loader(cfg, seed):
    tcfg = cfg["training"]
    bs = tcfg["batch_size"]
    nw = int(tcfg.get("num_workers", 0))
    # 混训 SNR: 每样本抽 U{train_snr_choices} dB (对齐原文); None/[] -> 无噪训练
    train_snr = tcfg.get("train_snr_choices") or None
    # 多 worker 并行现采 (数据生成是瓶颈); persistent 避免每 epoch 重启 worker。
    dl_kw = dict(num_workers=nw)
    if nw > 0:
        dl_kw.update(persistent_workers=True, prefetch_factor=4)
    if tcfg["cache"]["enabled"]:
        ds = CachedUnifiedDataset(cfg, tcfg["cache"]["cache_size"], seed=seed, snr_db=train_snr)
        return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, **dl_kw), True
    # 在线: 大 size 近似无限流。__getitem__ 用 seed+idx 定种子, 多 worker 下每 idx
    # 只被处理一次 -> 无重复/无泄漏。
    ds = UnifiedBTTDataset(cfg, size=bs * tcfg["iters"], seed=seed, snr_db=train_snr)
    return DataLoader(ds, batch_size=bs, shuffle=False, **dl_kw), False


def format_snr_row(per_snr, key="recall@5"):
    return " | ".join(
        f"{int(s)}dB {key.split('@')[-1] if '@' in key else key}={m.get(key, float('nan')):.3f}"
        for s, m in per_snr.items())


def train_one(cfg, etype, device):
    """训练单个编码器 (cfg 已定死 encoder.type=etype)。返回最终 per_snr。"""
    cfg = copy.deepcopy(cfg)
    cfg["encoder"]["type"] = etype
    torch.manual_seed(cfg["seed"])
    print(f"\n========== 训练 {etype} ==========")
    print(f"[cfg] encoder={etype} G={grid_size(cfg)} signal_dim={_signal_dim(cfg)} "
          f"regression={is_regression(cfg)} device={device}")

    model = build_encoder(cfg).to(device)
    fr_size = getattr(model, "fr_size", None)
    loss_fn = build_loss(cfg, fr_size=fr_size, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M fr_size={fr_size}")

    loader, cached = make_train_loader(cfg, cfg["seed"])
    out_dir = Path(cfg["checkpoint"]["dir"]) / etype
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    iters = cfg["training"]["iters"]
    eval_every = cfg["eval"]["eval_every"]
    grad_clip = cfg["training"]["grad_clip"]
    mon_snr = cfg["eval"]["monitor_snr_db"]
    mon_L = cfg["eval"]["monitor_recall_L"]
    train_log = []      # 训练期: 每评测点 train_loss / val_loss / R@monitor_snr
    t0 = time.time()
    it = 0
    data_iter = iter(loader)
    while it < iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        it += 1
        model.train()
        output = encoder_forward(model, batch, cfg, device)
        loss = loss_fn(output, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if it == 1 or it == iters or it % eval_every == 0:
            # 训练期监控: 只在单档 SNR 评测 (含 val_loss + R@monitor_L)
            mon = evaluate_at_snr(model, cfg, device, mon_snr, loss_fn=loss_fn)
            train_loss = float(loss.item())
            val_loss = mon.get("val_loss", float("nan"))
            recall = mon.get(f"recall@{mon_L}", float("nan"))
            row = {"iter": it, "train_loss": train_loss, "val_loss": val_loss,
                   f"recall@{mon_L}_{int(mon_snr)}dB": recall,
                   "elapsed_s": time.time() - t0}
            train_log.append(row)
            print(f"[it {it:>6}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"R{mon_L}@{int(mon_snr)}dB={recall:.3f}")
            (out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2), encoding="utf-8")
            torch.save({"model_state_dict": model.state_dict(), "iter": it, "cfg": cfg},
                       out_dir / "latest.pt")

    # 训练结束: 全 SNR 分档评估 -> metrics.json
    print(f"[{etype}] 训练完成, 跑全 SNR 分档评估...")
    per_snr = evaluate_all_snr(model, cfg, device, loss_fn=loss_fn)
    metrics = {"encoder": etype, "iters": iters,
               "train_log": train_log, "per_snr": per_snr}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"encoder": etype, "output_dir": str(out_dir),
                      "final_per_snr": per_snr}, indent=2, default=str))
    return per_snr


def resolve_encoder_list(cfg, cli_encoder):
    """决定这次要训练哪些编码器。

    --encoder 优先 (只训它); 否则用 config encoder.train 列表; 列表空则回退 type。
    """
    if cli_encoder:
        return [cli_encoder]
    train_list = cfg["encoder"].get("train") or []
    if train_list:
        return list(train_list)
    return [cfg["encoder"]["type"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=str, default=None,
                    help="只训这一个编码器 (覆盖 config encoder.train 列表)")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=None)
    ap.add_argument("--eval_batches", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    cli = ap.parse_args()

    cfg = copy.deepcopy(CONFIG)
    if cli.iters:        cfg["training"]["iters"] = cli.iters
    if cli.eval_every:   cfg["eval"]["eval_every"] = cli.eval_every
    if cli.eval_batches: cfg["eval"]["eval_batches"] = cli.eval_batches
    if cli.device:       cfg["training"]["device"] = cli.device
    if cli.seed is not None: cfg["seed"] = cli.seed

    device = torch.device(cfg["training"]["device"])
    # 本机 OpenMP/MKL 冲突下 Conv2d(ResFreq) 会段错误 (环境问题非代码 bug)。
    # 单线程 + 关闭 mkldnn 后端可根治; CNN/RoPE 不受影响。
    if device.type == "cpu":
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled = False

    encoders = resolve_encoder_list(cfg, cli.encoder)
    print(f"[plan] 本次训练编码器: {encoders}")

    summary = {}
    for etype in encoders:
        summary[etype] = train_one(cfg, etype, device)

    if len(encoders) > 1:
        print("\n========== 汇总 (final R@5) ==========")
        for etype, per_snr in summary.items():
            row = " | ".join(f"{int(s)}dB={m.get('recall@5', float('nan')):.3f}"
                             for s, m in per_snr.items())
            print(f"  {etype:>20}: {row}")


if __name__ == "__main__":
    main()
