# Handoff — repro-arditi-wollschlager

## 状态
见 PROGRESS.md

## 关键路径
- 设计 spec: docs/superpowers/specs/2026-04-19-repro-arditi-wollschlager-design.md
- 实现 plan: docs/superpowers/plans/2026-04-19-repro-arditi-wollschlager.md
- DIM 产出: results/repro_arditi_wollschlager/{Qwen2.5-7B-Instruct,Llama-3.1-8B-Instruct}/
- RDO/Cone 产出: results/repro_arditi_wollschlager/rdo/{...}/
- 日志: experiments/repro_arditi_wollschlager/logs/

## Gate 状态
- [ ] Gate 1: T6 smoke test 通过 → qi 确认 → 启动 T7
- [ ] Gate 2: T7 DIM 全量完成 → qi 确认 → 启动 T8/T9
- [ ] Gate 3: T10 评估完成 → qi 审核 → T12 结论

## Kill 条件
- K1: T6 smoke test 连续 3 次崩溃 → 停止报告
- K2: T7 双模型 ablation ASR 都不升 → 停止报告
