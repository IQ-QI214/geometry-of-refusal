# Phase 2 进展交接 (2026-03-30)

## 已完成

### Exp 2A: Confound Resolution — 完成，结论明确

**结果**: `results/exp_2a_results.json`, `results/exp_2a_controlled_directions.pt`

**核心数据**:
- Controlled min similarity = **0.231** (< 0.40 阈值)
- Controlled mean similarity = **0.413**
- 对比 Exp B uncontrolled min = 0.018 → 改善 +0.213

**决策**: **DYNAMIC** — 即使消除了 content difference confound，refusal direction 仍然在 generation 过程中显著旋转。Exp B 原始结果中约一半的漂移来自 confound，但剩余的漂移是真实的。

**对后续的 implication**: Exp 2C 应使用 timestep-specific directions 或 SVD cone basis (从 `exp_2a_controlled_directions.pt` 构建)，而非单一静态 direction。

---

### Exp 2B: Ablation Attack — 部分完成，需修复后重跑

**状态**: 跑完 `baseline_text` (58 prompts)，在 `blank_image` config 第 34 个 prompt 处被 **OOM kill**。

**已获得的 baseline_text 数据** (从 log 中可提取):
- 58 prompts: bypass rate ≈ 22.4%, full harmful ≈ 17.2%
- 大部分 prompt 被直接 refuse

**OOM 原因**: 2A 和 2B 并行在同一节点跑，两个 LLaVA 实例同时占用 GPU。加上 2B 有 7 个 config × 58 prompts = 406 次 generation。

**修复方案**:
1. `generate_with_config()` 中添加 `torch.cuda.empty_cache()` 在每次 generation 后
2. 不要并行跑 2A 和 2B（顺序执行或用不同 GPU 节点）
3. 考虑 `--quick` 模式先用 8 个 test prompts 验证 hook 可行性
4. `pad_token_id` 警告：在 generate 调用中显式传入 `pad_token_id=processor.tokenizer.eos_token_id`

---

## 代码位置

```
experiments/phase2/
├── common/                          # 共享模块（已完成）
│   ├── llava_utils.py, hook_utils.py, eval_utils.py, direction_utils.py
├── exp_2a/exp_2a_confound_resolution.py   # ✅ 已跑通
├── exp_2b/exp_2b_ablation_attack.py       # ⚠️ 需修复 OOM
├── exp_2c/                                # 待跑（依赖 2B 的 target completions）
│   ├── exp_2c_visual_perturbation.py
│   └── perturbation_utils.py
├── exp_2d/exp_2d_ablation_study.py        # 待跑（依赖 2C）
```

## 下一步行动

1. **修 Exp 2B OOM 问题** → 加 `torch.cuda.empty_cache()`，添加 `pad_token_id`，先 `--quick` 跑通
2. **跑完 Exp 2B** → 得到 ablation attack 的 ASR 数据 + harmful completions
3. **跑 Exp 2C** → 核心 PGD 视觉扰动（direction_mode 已自动设为 "dynamic"）
4. **写 Phase 2 分析报告** → `analysis/phase2_report.md`

## 关键文件指引

| 文件 | 说明 |
|---|---|
| `plan-markdown/gapc-research-framework` | 研究框架总文档 |
| `plan-markdown/gapc-pilot-exp` | Pilot 实验操作指南 |
| `analysis/pilot_experiments_report.md` | Pilot 实验分析报告 |
| `base_instructions.md` | 基础指令（环境、规范） |
| `results/exp_2a_results.json` | Exp 2A 结果 (decision: dynamic) |
| `results/exp_2a_controlled_directions.pt` | 5 个 position-specific directions |
