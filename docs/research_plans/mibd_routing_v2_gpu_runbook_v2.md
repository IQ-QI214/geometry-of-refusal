# MIBD Routing V2 — GPU 阶段 Runbook v2（数据修复 + 路由训练）

> 2026-06-30 · 接 `mibd_routing_v2_offline_gpu_runbook.md`（v1）
> 解决 v1 遗留的 `generated_blank_placeholder` 混淆：v1 的 safe 控制是**空白占位图**，
> 导致探针在**第 0 层**就 AUC=1.0（分的是"空白 vs 含内容"而非有害语义）。
> 本 v2 先用 format-matched 中性载体重建数据并**重抽隐状态**，再训练路由。
>
> 验证依据：`experiments/mibd_routing_v2/eval/check_saturation.py` 对 v1 的
> `probe_summary.json` 判定 `saturated=true`（exit 1）。本流程跑完后必须判定 `OK`。

---

## 阶段 0：修数据（GPU 前，CPU 即可，但放这里保证顺序）

### 0.1 生成 format-matched 中性 safe 池（替换空白占位图）

```bash
python -m experiments.mibd_routing_v2.run_build_neutral_safe_pool \
  --output-dir data/mibd_routing_v2/benign_safe_images \
  --per-category 8 \
  --carriers typographic,figstep
```

> 产物：`data/mibd_routing_v2/benign_safe_images/<01..13>/neutral_<carrier>_<k>.png`
> 白底渲染文本、与风险载体**同视觉格式**，但内容为中性日常指令（见 `neutral_carrier_renderer.NEUTRAL_PHRASES`）。

### 0.2 重建配对数据集，指向中性池（关键：带 `--safe-image-dir`）

```bash
python -m experiments.mibd_routing_v2.run_build_phase2a_dataset \
  --mmsafety-dir <OFFLINE_MM_SAFEBENCH_DIR> \
  --safe-image-dir data/mibd_routing_v2/benign_safe_images \
  --output-dir results/mibd_routing_v2/paired_dataset/phase2a_matched_v3 \
  --num-pairs 200 \
  --seed 20260604
```

检查：`build_report.md` 的 `safe image modes` 必须是 `category_matched`（**不再**是 `generated_blank_placeholder`）。

---

## 阶段 1：重抽隐状态（GPU）

沿用 v1 的 `run_phase2b_extract_probe`，仅把 `--dataset` 换成 v3、`--output-dir` 换成新目录：

```bash
# Qwen3-VL-8B
conda run -n qwen3-vl python -m experiments.mibd_routing_v2.run_phase2b_extract_probe \
  --model qwen3_vl_8b \
  --dataset results/mibd_routing_v2/paired_dataset/phase2a_matched_v3/paired_dataset.jsonl \
  --output-dir results/mibd_routing_v2/sensor_probe/qwen3_vl_8b_v3 \
  --device cuda:0 \
  --layers 0,4,8,12,16,20,24,28,32,35 \
  --positions=-1

# InternVL3-8B
conda run -n rdo python -m experiments.mibd_routing_v2.run_phase2b_extract_probe \
  --model internvl3_8b \
  --dataset results/mibd_routing_v2/paired_dataset/phase2a_matched_v3/paired_dataset.jsonl \
  --output-dir results/mibd_routing_v2/sensor_probe/internvl3_8b_v3 \
  --device cuda:0 \
  --layers 0,4,8,12,16,20,24,27 \
  --positions=-1
```

---

## 阶段 2：验证混淆已破除（CPU，Go/No-Go 闸门）

```bash
python -m experiments.mibd_routing_v2.eval.check_saturation \
  --probe-summary results/mibd_routing_v2/sensor_probe/qwen3_vl_8b_v3/probe_summary.json
# 期望 verdict=OK，exit 0；若仍 SATURATED 说明数据没修对，停止

python -m experiments.mibd_routing_v2.eval.run_offline_oracle \
  --npz results/mibd_routing_v2/sensor_probe/qwen3_vl_8b_v3/hidden_states.npz \
  --out results/mibd_routing_v2/offline_oracle/qwen3_vl_8b_v3.json
# 期望: cross_carrier_transfer_drop > 0（跨载体迁移开始失效 = 路由有意义的信号）
```

**Go 条件**：`check_saturation` 判 OK **且** 早层 AUC 明显低于中后层。
只有满足才进入阶段 3 训练；否则路由会再次训练在平凡信号上。

---

## 阶段 3：路由核心 torch 训练（GPU）

torch 模块：`experiments/mibd_routing_v2/routing/torch_modules.py`
（`CarrierRouter` / `ThresholdGate` / `LowRankBridge` + 防坍缩损失，已与 numpy 参考对齐，
见 `tests/test_torch_parity.py`，GPU 上 `pytest` 该文件应全过）。

最小训练循环骨架（伪代码，按 round-1 矩阵 T2/T3）：

```python
import torch
from experiments.mibd_routing_v2.routing import torch_modules as tm

cfg = tm.TorchRouterConfig(hidden_dim=H, num_layers=L, num_carriers=C, mode="soft")
router = tm.CarrierRouter(cfg).cuda()
gate   = tm.ThresholdGate(tau=0.0, alpha=4.0, learn_threshold=True).cuda()
bridge = tm.LowRankBridge(hidden_dim=H, rank=4, alpha=8.0).cuda()
opt = torch.optim.AdamW([*router.parameters(), *gate.parameters(), *bridge.parameters()], lr=1e-3)

bias = torch.zeros(L).cuda()          # Loss-Free 偏置
target = torch.full((L,), 1.0/L).cuda()
for step, batch in enumerate(loader):
    feats, carrier_ids = batch.feats.cuda(), batch.carrier_ids.cuda()
    probe_scores = batch.per_layer_probe_scores.cuda()   # (n, L)
    T = tm.anneal_temperature(step, 2.0, 0.5, total_steps)
    out = router(feats, carrier_ids, temperature=T)
    risk = tm.aggregate_risk_score(out["layer_probs"], probe_scores)
    g = gate(risk)

    # C2 蒸馏（teacher = 纯文本会拒答时的 gate 隐状态） + 防坍缩
    student_gate = bridge(batch.gate_hidden.cuda(), g)
    loss = (
        tm.distillation_loss_c2(student_gate, batch.teacher_gate_hidden.cuda())
        + 0.01 * tm.router_z_loss(out["layer_logits"])
        + 0.01 * tm.switch_load_balancing_loss(out["layer_probs"], out["selected"])
        + 0.1  * tm.utility_anchor_penalty(bridge.delta(batch.gate_hidden.cuda()), g)
    )
    opt.zero_grad(); loss.backward(); opt.step()

    # Loss-Free 偏置在 step 间更新（无梯度）
    load = out["selected"].mean(dim=0).detach()
    bias = tm.update_loss_free_bias(bias, load, target, lr=1e-3)
```

> 训练所需的 `per_layer_probe_scores` / `gate_hidden` / `teacher_gate_hidden`
> 由阶段 1 的隐状态 + 反事实纯文本前向产生（teacher 抽取脚本属下一批 GPU 代码，
> 本 runbook 暂留接口）。

---

## 边界与状态

| 部分 | 状态 |
|---|---|
| 中性载体渲染 / 数据重建 / 隐状态重抽脚本 | ✅ 已存在，本 runbook 串好 |
| 饱和度闸门 `check_saturation` | ✅ 新增，CPU 可跑，已对 v1 判 saturated |
| 路由核心 torch 移植 `torch_modules.py` | ✅ 新增，parity 测试待 GPU 跑 |
| teacher 抽取 / 完整训练脚本 / 评测竞品 | ⏳ 下一批 GPU 代码 |
