# Repro Arditi-Wollschläger

**阶段**: `repro-arditi-wollschlager-2026-04-19`
**目标**: 在纯 LLM 上复现 DIM + RDO/Cone，验证 pipeline 正确性。

## 快速运行

```bash
# 1. GPU smoke test (Qwen only, 32 prompts)
CUDA_VISIBLE_DEVICES=0 conda run -n rdo \
    python experiments/repro_arditi_wollschlager/smoke_test.py \
    | tee experiments/repro_arditi_wollschlager/logs/smoke_test.log

# 2. DIM 全量 (Gate 1 通过后)
bash experiments/repro_arditi_wollschlager/run_dim.sh

# 3. RDO k=1 (Gate 2 通过后)
bash experiments/repro_arditi_wollschlager/run_rdo.sh

# 4. Cone k=2→5
bash experiments/repro_arditi_wollschlager/run_cone.sh

# 5. 全量评估
conda run -n rdo python experiments/repro_arditi_wollschlager/run_evaluate.py \
    | tee experiments/repro_arditi_wollschlager/logs/eval.log
```

## 文档
- 进度: PROGRESS.md
- 接手: HANDOFF.md
- 结果: FINDINGS.md
- 设计: docs/superpowers/specs/2026-04-19-repro-arditi-wollschlager-design.md
