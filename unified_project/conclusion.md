# 统一编码器对比结论

四种局部编码器（regression_cnn / classification_cnn / resfreq / rope）共用同一份
非均匀 BTT 数据，mixed-train / separated-eval 协议。

---

## 实验一：5 探头 / 100Hz 转速 / K=8（paper5，开梯度裁剪）

结果目录：`artifacts/`（旧正式版）

recall@8 分档 + 20dB freq MAE（K=8，取 8 峰）：

| 编码器 | -5dB | 0dB | 5dB | 10dB | 20dB | 30dB | 20dB MAE |
|---|---|---|---|---|---|---|---|
| regression_cnn | 0.405 | 0.608 | 0.765 | 0.860 | 0.903 | 0.907 | 2.88Hz |
| classification_cnn | 0.368 | 0.568 | 0.711 | 0.794 | 0.833 | 0.837 | 4.22Hz |
| rope | 0.335 | 0.509 | 0.619 | 0.663 | 0.682 | 0.684 | 10.26Hz |
| resfreq | 0.156 | 0.220 | 0.282 | 0.325 | 0.344 | 0.347 | 11.80Hz |

要点：
- resfreq 垫底，因为圈数太少（16 转短窗、协方差图 D=block_revs×n_probes 很小），
  却套了很深的 2D ResNet（通道翻倍到 512），过参数化 + 输入信息不足。ckpt 57MB
  ≈ 其他三家 6–9×。
- 训练开了梯度裁剪（grad_clip=1.0）。

---

## 实验二：8 探头 / 30Hz 转速 / K=6（real8，grad_clip=0）

结果目录：`artifatcts_8pro_30hzFr_6fre/`

新增两个更能区分能力的指标：
- **all-K@L（样本级全命中率）**：一个样本的全部真频都被 top-L 覆盖才记 1（比分量级 recall 严格得多）。
- **median 误差**：MAE 是均值、被 ±k·f_rot 混叠重尾主导；median 反映"选对的峰"的真实定位精度。

评测 L 取 **6/8/10**（K=6，只有 6 个真频；R@6 = 恰好取 6 峰的召回，allK@6 = 6 峰全中）。

20dB 档核心对比：

| 编码器 | R@6 | **allK@6** | R@8 | **allK@8** | mae | **median** |
|---|---|---|---|---|---|---|
| regression_cnn | **0.858** | **0.329** | 0.899 | 0.500 | 5.15Hz | **0.14Hz** |
| classification_cnn | 0.805 | 0.172 | 0.848 | 0.302 | 7.54Hz | 0.62Hz |
| rope | 0.664 | 0.021 | 0.712 | 0.072 | 12.92Hz | 0.40Hz |
| resfreq | 0.694 | 0.050 | 0.731 | 0.104 | 7.23Hz | 0.92Hz |

完整分档（R@6/8/10 | allK@6/8/10 | mae / median）：

```
regression_cnn
  -5dB  R6=0.437 R8=0.480 R10=0.515 | allK6=0.000 allK8=0.002 allK10=0.005  mae=18.23 med=7.08Hz
   0dB  R6=0.633 R8=0.681 R10=0.715 | allK6=0.016 allK8=0.051 allK10=0.088  mae=12.25 med=0.32Hz
   5dB  R6=0.766 R8=0.815 R10=0.839 | allK6=0.112 allK8=0.227 allK10=0.296  mae=7.85  med=0.19Hz
  10dB  R6=0.833 R8=0.876 R10=0.898 | allK6=0.250 allK8=0.405 allK10=0.495  mae=5.73  med=0.16Hz
  20dB  R6=0.858 R8=0.899 R10=0.917 | allK6=0.329 allK8=0.500 allK10=0.581  mae=5.15  med=0.14Hz
  30dB  R6=0.859 R8=0.898 R10=0.916 | allK6=0.341 allK8=0.499 allK10=0.579  mae=5.06  med=0.14Hz

classification_cnn
  -5dB  R6=0.413 R8=0.464 R10=0.504 | allK6=0.000 allK8=0.001 allK10=0.006  mae=20.92 med=8.27Hz
   0dB  R6=0.598 R8=0.649 R10=0.685 | allK6=0.007 allK8=0.027 allK10=0.051  mae=14.98 med=0.86Hz
   5dB  R6=0.723 R8=0.772 R10=0.805 | allK6=0.058 allK8=0.131 allK10=0.204  mae=10.40 med=0.68Hz
  10dB  R6=0.784 R8=0.832 R10=0.860 | allK6=0.127 allK8=0.252 allK10=0.348  mae=8.32  med=0.63Hz
  20dB  R6=0.805 R8=0.848 R10=0.875 | allK6=0.172 allK8=0.302 allK10=0.401  mae=7.54  med=0.62Hz
  30dB  R6=0.807 R8=0.850 R10=0.875 | allK6=0.173 allK8=0.310 allK10=0.399  mae=7.57  med=0.62Hz

resfreq
  -5dB  R6=0.309 R8=0.344 R10=0.368 | allK6=0.000 allK8=0.000 allK10=0.000  mae=19.78 med=6.87Hz
   0dB  R6=0.485 R8=0.519 R10=0.539 | allK6=0.000 allK8=0.005 allK10=0.010  mae=13.83 med=2.18Hz
   5dB  R6=0.609 R8=0.643 R10=0.665 | allK6=0.008 allK8=0.028 allK10=0.043  mae=9.90  med=1.22Hz
  10dB  R6=0.671 R8=0.704 R10=0.724 | allK6=0.031 allK8=0.073 allK10=0.098  mae=7.74  med=0.98Hz
  20dB  R6=0.694 R8=0.731 R10=0.750 | allK6=0.050 allK8=0.104 allK10=0.139  mae=7.23  med=0.92Hz
  30dB  R6=0.696 R8=0.731 R10=0.752 | allK6=0.044 allK8=0.104 allK10=0.141  mae=7.15  med=0.91Hz

rope
  -5dB  R6=0.338 R8=0.381 R10=0.415 | allK6=0.000 allK8=0.000 allK10=0.000  mae=23.18 med=11.71Hz
   0dB  R6=0.509 R8=0.555 R10=0.588 | allK6=0.000 allK8=0.006 allK10=0.014  mae=17.49 med=1.53Hz
   5dB  R6=0.615 R8=0.660 R10=0.691 | allK6=0.011 allK8=0.032 allK10=0.056  mae=14.39 med=0.54Hz
  10dB  R6=0.650 R8=0.697 R10=0.728 | allK6=0.018 allK8=0.054 allK10=0.091  mae=13.37 med=0.43Hz
  20dB  R6=0.664 R8=0.712 R10=0.743 | allK6=0.021 allK8=0.072 allK10=0.115  mae=12.92 med=0.40Hz
  30dB  R6=0.664 R8=0.713 R10=0.744 | allK6=0.021 allK8=0.072 allK10=0.118  mae=12.77 med=0.39Hz
```

### 结论

1. **freq_mae 三家看似"差不多"（5–13Hz）是假象**。median 全部只有 0.1–0.9Hz，
   说明一半以上样本其实定位到亚 Hz（远优于 2Hz 网格）；MAE 被 ±k·f_rot=30Hz 的
   混叠重尾拉高并饱和到同一平台。要看真实精度看 median，不看 mae。

2. **MAE 比实验一变大（regression 2.88→5.15Hz）是转速 100→30Hz 的预期后果**，
   不是 bug、不是 SNR。转速降 3.3× → BTT 旁瓣间距 f_rot 从 100→30Hz，混叠更密、
   更易骗过 top-K 选峰 → 重尾更重。recall 几乎没变（强峰照样选对），变差的是被
   混叠污染那部分的定位 = RMSE 重尾。压尾要靠物理精修（粗选 basin + 窄带 LS），
   换网络无用。

3. **all-K 把网络能力差异放大**。分量级 recall@6 挤在 0.66–0.86，但样本级全中率
   allK@6（6 峰全中）拉开成 0.329 / 0.172 / 0.050 / 0.021（regression / classification
   / resfreq / rope）—— 即便放宽到取 8 峰（allK@8）也只有 0.50 / 0.30 / 0.10 / 0.07。
   regression_cnn 恰取 6 峰时约 1/3 样本 6 频全中，resfreq/rope 仅 ~5%/2%。这才是真实排序。

4. **rope 特点：选对就准（median 0.40Hz，第二好），但常漏分量（allK 最低）**，
   坐实"rope 只做粗选、精度靠物理精修"的定位。

5. **排序稳定**：regression_cnn ≫ classification_cnn > resfreq ≈ rope；
   resfreq 参数最多（57MB）却全面垫底，同实验一原因（短窗小协方差 + 过深 2D ResNet）。
