"""训练报告统一工具 (六阶段共用): config 落盘 / train_log.json / TensorBoard / 选模型 / metrics.json。

统一口径 (2026-07-28 定):
  一、log        每次评测打印 train_loss + val_loss; loss 多项时打印总值和每一项。
                 loss 用 200 步**滑动均值** (不打瞬时值)。
                 metric: 不训方差头 -> recall@K@20dB + allK@20dB;
                         训方差头   -> 再加 sigma_corr + sigma_ratio(σ/err)。
                 train_log.json **随训练更新** (每个评测点落盘), loss/metric 同步写 TB。
  二、metrics    训练结束写 metrics.json, 六阶段**都做分档评估** (eval_snr_db 每档);
                 训了方差头的阶段, 分档里再加 sigma_corr / sigma_ratio。
  三、config     训练开始把 config 复制到结果目录 (config.json)。
  四、选模型     阶段 1/3/5 按 val_recall@K@20dB 最大; 阶段 2/4/6 按 val_loss 最小。
  五、双 recall  阶段 3-6 报两路: sum@K 与模型输出 model@K。
                 ⚠ sum **只进 log, 不参与选模型**。

用法 (典型):
    rep = TrainReporter(out_dir, cfg, stage="time_axis", select="recall",
                        train_sigma=False, tb_name="axial_timeaxis")
    ...
    rep.log_step(it, train_parts, val_parts, metrics, lrs)   # 每个评测点
    rep.finish(per_snr)                                      # 结束
"""
import json
import os
import time

import numpy as np
import torch

from ckpt_root import TB_ROOT, no_clobber


def dump_config(out_dir, cfg):
    """[要求三] 训练开始把 config 落到结果目录, 产物目录自解释。"""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, default=str, ensure_ascii=False)


class RunningMean:
    """[要求一] loss 滑动均值 (窗口 200)。瞬时值 std 可达均值 50%, 只报滑动值。"""

    def __init__(self, window=200):
        self.window = int(window)
        self._buf = {}

    def update(self, parts):
        """parts: {name: float} -> 返回当前各项滑动均值 {name: float}。"""
        for k, v in parts.items():
            b = self._buf.setdefault(k, [])
            b.append(float(v))
            if len(b) > self.window:
                b.pop(0)
        return self.value()

    def value(self):
        return {k: float(np.mean(v)) for k, v in self._buf.items() if v}


def sigma_calibration(sigmas, errs):
    """σ 校准两指标: corr(σ,|err|) 与 median(σ/|err|)。

    corr   排序能力 (σ 能否分辨误差大小)
    ratio  数值吻合 (→1 才是真校准; corr 高不代表 ratio 对)
    """
    s = np.asarray(sigmas, dtype=np.float64)
    e = np.asarray(errs, dtype=np.float64)
    ok = (s > 0) & np.isfinite(s) & np.isfinite(e)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    corr = float(np.corrcoef(s[ok], e[ok])[0, 1])
    ratio = float(np.median(s[ok] / np.clip(e[ok], 1e-6, None)))
    return corr, ratio


class TrainReporter:
    """六阶段统一的 log / ckpt / metrics 报告器。

    select: "recall" -> 按 val_recall 最大选 (阶段 1/3/5)
            "loss"   -> 按 val_loss 最小选   (阶段 2/4/6)
    train_sigma: 该阶段是否训练方差头 (决定是否报 sigma_corr/sigma_ratio)。
    """

    def __init__(self, out_dir, cfg, stage, select, train_sigma,
                 tb_name=None, tb_dir=None, recall_L=None, monitor_snr_db=20.0):
        assert select in ("recall", "loss"), f"select 只能是 recall/loss, 得到 {select!r}"
        # ⚠ 不覆盖旧结果: 目录已有产物就加时间戳另起 (见 ckpt_root.no_clobber)
        self.out_dir = no_clobber(str(out_dir))
        self.cfg = cfg
        self.stage = stage
        self.select = select
        self.train_sigma = bool(train_sigma)
        self.recall_L = recall_L
        self.monitor_snr_db = float(monitor_snr_db)

        os.makedirs(self.out_dir, exist_ok=True)
        dump_config(self.out_dir, cfg)                      # [要求三]

        self.train_log = []
        self.best_recall = -float("inf")
        self.best_val = float("inf")
        self.t0 = time.time()

        # [要求一] TB: loss 与 metric 都落盘
        self.tb = None
        _dir = tb_dir or cfg.get("tb_dir") or TB_ROOT
        if tb_name:
            tb_name = os.path.basename(no_clobber(os.path.join(_dir, tb_name)))
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(os.path.join(_dir, tb_name))
                print(f"[tb] {os.path.join(_dir, tb_name)}")
            except Exception as exc:                        # noqa: BLE001
                print(f"[tb] 未启用 ({exc})")

    # ---------- 每个评测点 ----------
    def log_step(self, it, train_parts, val_parts, metrics, lrs, ckpt=None):
        """写一行 train_log + TB + 存 last/best。

        train_parts / val_parts: {"total": x, "anchor": ..., "gnll": ...} (滑动均值)
        metrics: {"recall": float, "allk": float,
                  可选 "sum_recall","sum_allk","sigma_corr","sigma_ratio"}
        lrs: {name: lr}
        ckpt: 要保存的 dict (None 则不存盘)
        返回 (is_best, row)
        """
        row = {"iter": int(it), "elapsed_s": time.time() - self.t0}
        # 单分项 (== total) 不重复落盘, 与打印口径一致
        keep = set(self._loss_keys(train_parts, val_parts)) | {"total"}
        for k, v in train_parts.items():
            if k in keep:
                row[f"train_{k}"] = float(v)
        for k, v in val_parts.items():
            if k in keep:
                row[f"val_{k}"] = float(v)
        for k, v in metrics.items():
            row[k] = float(v)
        row["lr"] = {k: float(v) for k, v in lrs.items()}
        self.train_log.append(row)

        # [要求一] train_log.json 随训练更新
        with open(os.path.join(self.out_dir, "train_log.json"), "w", encoding="utf-8") as fh:
            json.dump(self.train_log, fh, indent=2, ensure_ascii=False)

        if self.tb is not None:
            tr_t, va_t = train_parts.get("total"), val_parts.get("total")
            if tr_t is not None and va_t is not None:
                self.tb.add_scalars("loss/train_vs_val",
                                    {"train": float(tr_t), "val": float(va_t)}, it)
            for k, v in train_parts.items():
                if k in keep:
                    self.tb.add_scalar(f"loss_train/{k}", float(v), it)
            for k, v in val_parts.items():
                if k in keep:
                    self.tb.add_scalar(f"loss_val/{k}", float(v), it)
            for k, v in metrics.items():
                self.tb.add_scalar(f"metric/{k}", float(v), it)
            for k, v in lrs.items():
                self.tb.add_scalar(f"lr/{k}", float(v), it)
            self.tb.flush()

        # [要求四] 选模型: 1/3/5 按 recall 最大; 2/4/6 按 val_loss 最小
        if self.select == "recall":
            score, best = metrics.get("recall", float("nan")), self.best_recall
        else:
            score, best = -float(val_parts.get("total", float("inf"))), -self.best_val
        is_best = score == score and score > best        # NaN-safe
        if is_best:
            if self.select == "recall":
                self.best_recall = float(score)
            else:
                self.best_val = float(val_parts.get("total"))

        if ckpt is not None:
            torch.save(ckpt, os.path.join(self.out_dir, "last.pt"))
            if is_best:
                torch.save(ckpt, os.path.join(self.out_dir, "best.pt"))
        return is_best, row

    # ---------- 打印 ----------
    #  [要求一] loss 的**每一项**都成对列出 (train/val 各一列), 不把分项挤到第二行。
    #  列布局: it | total(tr/va) | <每个分项>(tr/va) ... | model R@K/allK | sum R@K/allK
    #          [ σcorr σ/err ] | lr
    #  ⚠ 分项名要等第一次 log 才知道 (随阶段的 loss 而变) ⇒ 表头惰性打印。
    _LOSS_W = 17          # 每个 "train/val" 单元格宽度

    def _loss_keys(self, train_parts, val_parts):
        """分项名 (不含 total), train/val 并集, 保持 train 的出现顺序。

        ⚠ [要求一] "loss 中有不止一项才打印每一项": 只有一个分项时它必然等于 total
        (如纯 GNLL 的校准阶段, l_reg == total), 再单列一遍是重复 ⇒ 返回空列表。
        """
        keys = [k for k in train_parts if k != "total"]
        for k in val_parts:
            if k != "total" and k not in keys:
                keys.append(k)
        return keys if len(keys) > 1 else []

    def header(self, loss_keys=None):
        """打印表头。loss_keys 为 None 时推迟到第一行数据 (那时才知道分项名)。"""
        if loss_keys is None:
            self._header_done = False
            return
        cols = [f"{'it':>6}", f"{'total tr/val':>{self._LOSS_W}}"]
        for k in loss_keys:
            cols.append(f"{k + ' tr/val':>{self._LOSS_W}}")
        cols.append(f"{'model R/allK':>13}")
        cols.append(f"{'sum R/allK':>13}")
        if self.train_sigma:
            cols += [f"{'σcorr':>7}", f"{'σ/err':>7}"]
        cols.append(f"{'lr':>9}")
        line = " | ".join(cols)
        print("\n" + line)
        print("-" * len(line))
        self._header_done = True

    def print_row(self, it, train_parts, val_parts, metrics, lr):
        keys = self._loss_keys(train_parts, val_parts)
        if not getattr(self, "_header_done", False):
            self.header(keys)

        def _pair(d1, d2, k):
            a, b = d1.get(k, float("nan")), d2.get(k, float("nan"))
            return f"{a:.4f}/{b:.4f}".rjust(self._LOSS_W)

        seg = [f"{it:>6}", _pair(train_parts, val_parts, "total")]
        for k in keys:                                   # 每一项 loss 单独成列
            seg.append(_pair(train_parts, val_parts, k))
        seg.append(f"{metrics.get('recall', float('nan')):>6.3f}/"
                   f"{metrics.get('allk', float('nan')):.3f}")
        seg.append(f"{metrics.get('sum_recall', float('nan')):>6.3f}/"
                   f"{metrics.get('sum_allk', float('nan')):.3f}")
        if self.train_sigma:
            seg.append(f"{metrics.get('sigma_corr', float('nan')):>7.3f}")
            seg.append(f"{metrics.get('sigma_ratio', float('nan')):>7.3f}")
        seg.append(f"{lr:>9.2e}")
        print(" | ".join(seg), flush=True)

    # ---------- 结束 ----------
    def finish(self, per_snr, extra=None):
        """[要求二] 写 metrics.json (含分档评估)。per_snr: {snr: {metric: value}}。"""
        crit = ("val_recall@K@%gdB" % self.monitor_snr_db) if self.select == "recall" \
            else "val_loss"
        metrics = {
            "stage": self.stage,
            "best_criterion": crit,
            "best_recall": (self.best_recall if np.isfinite(self.best_recall) else None),
            "best_val_loss": (self.best_val if np.isfinite(self.best_val) else None),
            "train_sigma": self.train_sigma,
            "monitor_snr_db": self.monitor_snr_db,
            "recall_L": self.recall_L,
            "per_snr": per_snr,                 # 分档评估 (六阶段都做)
            "train_log": self.train_log,
        }
        if extra:
            metrics.update(extra)
        with open(os.path.join(self.out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=str, ensure_ascii=False)

        if self.tb is not None:
            for snr, m in per_snr.items():
                for k, v in m.items():
                    try:
                        self.tb.add_scalar(f"final_{k}/vs_snr", float(v), int(float(snr)))
                    except (TypeError, ValueError):
                        continue
            self.tb.close()

        print(f"\n[{self.stage}] 完成 {time.time()-self.t0:.0f}s | 判据={crit} | "
              f"best_recall={self.best_recall:.4f} best_val_loss={self.best_val:.4f}")
        print(f"  -> {self.out_dir}/best.pt, metrics.json (含 {len(per_snr)} 档分档评估)")
        return metrics
