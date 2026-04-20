# PROGRESS — repro-arditi-wollschlager

Format per task:
### T<N> <name> — done <YYYY-MM-DD>
- 做了什么:
- 得到什么:
- 保存在哪:

---

### T0 目录结构 — done 2026-04-19
- 做了什么: 建立实验目录和文档框架
- 得到什么: 目录树 + 空文档
- 保存在哪: experiments/repro_arditi_wollschlager/

### T1 Fix qwen_model.py Qwen2.5 架构 — done 2026-04-20
- 做了什么: 重写 orthogonalize_qwen_weights + act_add_qwen_weights，改用 model.model.embed_tokens / model.model.layers
- 得到什么: unit test 全通过
- 保存在哪: refusal_direction/pipeline/model_utils/qwen_model.py
