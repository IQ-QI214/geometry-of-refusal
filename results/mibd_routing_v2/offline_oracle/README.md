# 离线路由标定 round-0 结果（CPU，无 GPU）

> 2026-06-30 · 配套 `experiments/mibd_routing_v2/eval/` 与路由核心 `routing/`
> 运行：`python3 -m experiments.mibd_routing_v2.eval.run_offline_oracle --npz <hidden_states.npz> --out <report.json>`

## 一、做了什么

把已落盘的真实 VLM 隐状态接进纯 numpy 路由核心，跑通完整离线链路：
`probe_bank（每层探针）→ aggregate_risk_score → gate 阈值扫描`，并算出**跨载体迁移矩阵**。
两个模型各产出一份 JSON 报告：

- `results/mibd_routing_v2/offline_oracle/qwen3_vl_8b_loci10.json`
- `results/mibd_routing_v2/offline_oracle/internvl3_8b_loci8.json`

## 二、核心结果（一个明确的负结果）

| 指标 | Qwen3-VL-8B | InternVL3-8B |
|---|---|---|
| 同载体 oracle AUC（均值） | 1.0 | 1.0 |
| 跨载体 oracle AUC（均值） | 1.0 | 1.0 |
| 跨载体迁移损失 | 0.0 | 0.0 |

**所有层、所有载体、甚至跨载体迁移，AUC 全为 1.0。** 当前配对数据集**没有暴露**
CaRoB 赖以成立的"按载体重编码 ⇒ 固定单层读取失效"信号——固定单层、oracle 选层、
跨载体迁移三者完全无差异。

## 三、归因（方法学，重要）

数据集卡显示 `safe controls` 是 **空白占位图**（`generated_blank_placeholder`，200/200）。
因此探针实际区分的是"**空白图 vs 含内容图**"这一最底层视觉特征，而非有害语义——
这解释了为何**第 0 层**就已 AUC=1.0。换言之：当前 harmful/harmless 的对照在视觉输入级别
即可平凡线性可分，路由选层在此数据上无可学习空间。

## 四、结论与下一步

1. 路由核心 + 离线评测**工程链路已验证可用**（54 个 CPU 测试全绿），可直接复用于更难数据。
2. **要让"动态选层"有意义，必须先修数据对照**：用 format-matched 的中性载体
   （已有 `data/neutral_carrier_renderer.py`：typographic/figstep 中性图）替换空白占位图，
   让 harmful/harmless 在**同等视觉复杂度**下仅在语义上不同。这是 GPU 重抽隐状态前
   就该敲定的设计。
3. 重抽隐状态后，重跑本脚本即可得到非平凡的跨载体迁移矩阵——那才是 motivation 配图的真实证据。

> 边界：本结果纯 CPU、纯 numpy、不跑模型；matmul 的 macOS Accelerate 警告为虚警
> （与 einsum 一致到 1e-14），已在 `probe_bank.py` 用 `np.errstate` 屏蔽。
