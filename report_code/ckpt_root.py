"""ckpt / artifacts 根目录 —— 新旧 SNR 口径的模型必须分开存放。

背景: 2026-08-30 起数据生成的 SNR 口径从**最弱分量**改成**总功率**
  (sigma_n^2 = sum_k alpha_k^2 * 10^(-SNR/10), 见 config.py 的 train_snr_choices 注释)。
  数值区间未变 (train_snr_choices 仍 {0,50}) => 同一个 SNR 数字对应的噪声强得多,
  两套口径下训出来的模型**不可互换**, 但 ckpt 目录结构与文件名完全相同
  (axial_sigcalib/best.pt 等), 混在一起事后无从分辨 => 用目录隔离。

    artifacts/          = 2026-08-30 之前的**旧口径** (最弱分量) 模型, 只读存档
    artifacts_total/    = 新口径 (总功率) 模型, 当前默认写入

⚠ 目录不搬迁: 旧 ckpt 原地不动, 因为它是全部旧结果 (sweeps_weakest/) 的来源,
  搬走会让那批结果无法复现。新训练写到新目录, 两边并存。

用法:
    默认读/写新口径目录; 要回用旧口径 ckpt, 设 CKPT_DIR=artifacts。
        CKPT_DIR=artifacts python eval/<script>.py
⚠ 旧 ckpt 配新口径数据 (或反之) 等于严重外推, 结果无意义 —— 两者必须配套使用。
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# CKPT_DIR 可给相对名 (相对 report_code/) 或绝对路径。
CKPT_DIR = os.environ.get("CKPT_DIR", "artifacts_total")
ARTIFACTS = (CKPT_DIR if os.path.isabs(CKPT_DIR)
             else os.path.join(_HERE, CKPT_DIR))

# 旧口径目录 (只读, 供对照)。
ARTIFACTS_LEGACY = os.path.join(_HERE, "artifacts")


def ckpt_path(*parts):
    """拼 artifacts 根下的路径。"""
    return os.path.join(ARTIFACTS, *parts)

# TensorBoard 根目录: 放 report_code/tf-logs (远端即数据盘 /root/autodl-tmp 下);
#   原 /root/tf-logs 在系统盘 (30G 已满 91%)。TB_DIR 环境变量可覆盖。
TB_ROOT = os.environ.get("TB_DIR", os.path.join(_HERE, "tf-logs"))


def no_clobber(path):
    """⚠ 不覆盖旧结果: path 已存在 (文件, 或非空目录) 就加时间戳后缀另起一份。

    目录 foo/ -> foo_20260925_183000/; 文件 a.json -> a_20260925_183000.json。
    """
    path = os.path.normpath(path)
    taken = (os.path.isfile(path)
             or (os.path.isdir(path) and os.listdir(path)))
    if not taken:
        return path
    import time
    ts = time.strftime("%Y%m%d_%H%M%S")
    root, ext = (os.path.splitext(path) if os.path.isfile(path) else (path, ""))
    new = f"{root}_{ts}{ext}"
    print(f"[no_clobber] {path} 已存在, 改写到 {new}")
    return new
