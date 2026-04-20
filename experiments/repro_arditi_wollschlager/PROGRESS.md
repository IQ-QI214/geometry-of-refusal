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

### T2 Fix llama3_model.py 模板引号 — done 2026-04-20
- 做了什么: 删除 LLAMA3_CHAT_TEMPLATE 和 LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM 开头多余的 "
- 得到什么: unit test 全通过，格式化输出以 <|begin_of_text|> 开头
- 保存在哪: refusal_direction/pipeline/model_utils/llama3_model.py

### T3 Smart-quote prefixes — done 2026-04-20
- 做了什么: 在 _test_prefixes_jailbreakbench 末尾追加 4 条 U+2019 变体
- 得到什么: unit test 全通过，curly quote 拒绝响应被正确识别
- 保存在哪: refusal_direction/pipeline/submodules/evaluate_jailbreak.py

### T4 common/ 评估模块 — done 2026-04-20
- 做了什么: 写 model_paths.py / eval_judges.py / stealth_analysis.py，unit test 全通过
- 得到什么: keyword/LG3/SR judge 接口就绪，SRR+concordance 计算就绪
- 保存在哪: experiments/repro_arditi_wollschlager/common/

### T5 smoke_test.py — done 2026-04-19
- 做了什么: 写 smoke_test.py（32-sample DIM pipeline 独立脚本，不依赖 jailbreakbench）
- 得到什么: 语法检查通过，待 GPU 执行
- 保存在哪: experiments/repro_arditi_wollschlager/smoke_test.py

### T5b Patch rdo.py offline mode — done 2026-04-20
- 做了什么: 将 train_refusal_vector、train_refusal_cone 和 train_independent_vector 的 mode="online" 改为 os.getenv("WANDB_MODE","online")
- 得到什么: WANDB_MODE=offline 在 GPU 节点可生效；语法检查通过；三处 wandb.init 调用已修复
- 保存在哪: rdo.py (三处 wandb.init 调用，行号: 995, 1074, 1435)

### T7 run_dim 脚本 — done 2026-04-20
- 做了什么: 写 run_dim.py（完整 DIM pipeline）和 run_dim.sh（双模型并行，Qwen GPU0 / Llama GPU1）
- 得到什么: 语法检查通过，待 GPU 执行
- 保存在哪: experiments/repro_arditi_wollschlager/run_dim.py + run_dim.sh

### T8+T9 RDO+Cone 脚本 — done 2026-04-20
- 做了什么: 写 run_rdo.sh / run_rdo_evaluate.py / run_cone.sh，语法检查通过
- 得到什么: 3 个脚本就绪，待 GPU 执行
- 保存在哪: experiments/repro_arditi_wollschlager/

### T10+T11 评估脚本 — done 2026-04-20
- 做了什么: 写 run_evaluate.py（Keyword+LG3+SR）和 compute_summary.py（SRR+表格），语法检查通过
- 得到什么: 评估基础设施就绪，待 GPU 运行后执行
- 保存在哪: experiments/repro_arditi_wollschlager/run_evaluate.py + compute_summary.py

