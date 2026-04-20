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

