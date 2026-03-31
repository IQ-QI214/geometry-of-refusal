Here is Claude's plan:                                                                                         
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Phase 3 Exp 3C: Cross-Model Narrow Waist Ablation Attack                                                       
                                                                                                                
 Context                                                                                                        

 Exp 3A 已完成 4 个 VLM 的 refusal direction 提取和 amplitude/narrow waist 分析，发现两类行为模式：
 - Group A (LLaVA, InstructBLIP): amplitude reversal 存在，CLIP 系 ViT
 - Group B (Qwen2.5-VL, InternVL2): 全层均匀压制，自研 ViT

 Exp 3C 核心问题：narrow waist 单层 ablation 优于全层 ablation 的发现（Phase 2: 89.7% >
 74.1%）是否跨模型通用？Group A vs Group B 的 ablation 效果是否不同？

 Scope

 - 优先 LLaVA + Qwen2.5-VL（各代表一个 Group）
 - InternVL2 + InstructBLIP 作为补充验证
 - 不含 PGD（那是 Exp 2C/2D 的内容），仅 direction ablation during generation

 数据来源

 - exp_3a_directions.pt 提供每个模型每层的 v_text 和 v_mm 方向
 - TEST_PROMPTS（8 个 harmful prompts，Phase 2 定义）用于评估

 文件结构

 experiments/phase3/
 ├── common/
 │   ├── model_adapters.py   # 修改: 添加 generate_mm() 和 generate_text() 方法
 │   └── (其他不变)
 ├── exp_3c_ablation_attack.py  # 新文件: 主实验脚本
 └── run_3c_all.sh              # 新文件: 4 GPU 并行

 results/phase3/{model}/
     └── exp_3c_results.json    # 输出

 Implementation

 修改 1: common/model_adapters.py — 添加 generate 方法

 在基类 ModelAdapter 中添加：

 def generate_mm(self, mm_inputs, max_new_tokens=200, **kwargs) -> str:
     """MM 模式生成文本。返回 decoded response（不含 prompt）。"""

 def generate_text(self, text_inputs, max_new_tokens=200, **kwargs) -> str:
     """Text-only 模式生成文本。"""

 def decode_new_tokens(self, gen_ids, input_len) -> str:
     """解码新生成的 tokens。"""

 每个子类重写 generate_mm / generate_text，处理模型差异：

 - LLaVA: model.generate(**mm_inputs, ...) / model.language_model.generate(input_ids=...)
 - Qwen2.5-VL: model.generate(**mm_inputs, ...) / model.generate(**text_inputs, ...)（无图像就是 text-only）
 - InternVL2: 手动构造 inputs_embeds（同 forward_mm），然后 model.language_model.generate(inputs_embeds=...) /
 model.language_model.generate(input_ids=...)
 - InstructBLIP: model.generate(**mm_inputs, ...) / model.language_model.generate(input_ids=...)

 新文件: exp_3c_ablation_attack.py

 通用 ablation context manager

 泛化 Phase 2 的 ablation_context，不依赖特定模型结构：

 @contextlib.contextmanager
 def ablation_context(adapter, direction, target_layers=None):
     """
     在 adapter.get_llm_layers() 的指定层上注册 pre-hook，
     移除 hidden state 在 direction 上的投影。
     """
     llm_layers = adapter.get_llm_layers()
     n_layers = len(llm_layers)
     if target_layers is None:
         target_layers = list(range(n_layers))

     handles = []
     for idx in target_layers:
         layer = llm_layers[idx]
         # pre-hook: 从 layer input 移除 direction 投影
         h = layer.register_forward_pre_hook(make_ablation_hook(direction))
         handles.append(h)
         # output-hook on self_attn and mlp (如果存在)
         if hasattr(layer, 'self_attn'):
             h = layer.self_attn.register_forward_hook(make_ablation_output_hook(direction))
             handles.append(h)
         if hasattr(layer, 'mlp'):
             h = layer.mlp.register_forward_hook(make_ablation_output_hook(direction))
             handles.append(h)
     yield
     for h in handles:
         h.remove()

 实验配置

 每个模型 5 种配置：

 ┌───────────────────┬───────┬──────────┬──────────────────────┬───────────┐
 │      Config       │ Image │ Ablation │    Target Layers     │ Direction │
 ├───────────────────┼───────┼──────────┼──────────────────────┼───────────┤
 │ baseline_text     │ None  │ None     │ -                    │ -         │
 ├───────────────────┼───────┼──────────┼──────────────────────┼───────────┤
 │ baseline_mm       │ blank │ None     │ -                    │ -         │
 ├───────────────────┼───────┼──────────┼──────────────────────┼───────────┤
 │ ablation_nw_vmm   │ blank │ Yes      │ [narrow_waist_layer] │ v_mm      │
 ├───────────────────┼───────┼──────────┼──────────────────────┼───────────┤
 │ ablation_all_vmm  │ blank │ Yes      │ all layers           │ v_mm      │
 ├───────────────────┼───────┼──────────┼──────────────────────┼───────────┤
 │ ablation_nw_vtext │ blank │ Yes      │ [narrow_waist_layer] │ v_text    │
 └───────────────────┴───────┴──────────┴──────────────────────┴───────────┘

 NW layer 和 direction 从 exp_3a_directions.pt 加载。

 每个模型的 NW layer（来自 3A）：
 - LLaVA: 16, Qwen2.5-VL: 24, InternVL2: 28, InstructBLIP: 20

 主流程

 1. 加载模型 + adapter
 2. 加载 exp_3a_directions.pt → 获取 NW layer + v_text/v_mm at NW
 3. 对 TEST_PROMPTS (8 个) × 5 configs:
    a. 注册/不注册 ablation hooks
    b. 调用 adapter.generate_mm() 或 generate_text()
    c. evaluate_response() 检测 refusal/self-correction
 4. compute_attack_metrics() 汇总
 5. 保存 exp_3c_results.json

 输出格式

 {
   "model": "llava_7b",
   "narrow_waist_layer": 16,
   "configs": {
     "baseline_text": {
       "metrics": {...},
       "responses": [{"prompt": "...", "response": "...", "initial_bypass": true, ...}]
     },
     ...
   }
 }

 新文件: run_3c_all.sh

 与 run_3a_all.sh 相同模式：4 GPU 并行，Qwen2.5-VL 使用 qwen3-vl 环境。

 关键复用

 ┌───────────────────────────────────────┬───────────────────────────────────────────────┐
 │                 来源                  │                   复用内容                    │
 ├───────────────────────────────────────┼───────────────────────────────────────────────┤
 │ phase2/common/eval_utils.py           │ evaluate_response(), compute_attack_metrics() │
 ├───────────────────────────────────────┼───────────────────────────────────────────────┤
 │ phase2/common/llava_utils.py          │ TEST_PROMPTS (8 个)                           │
 ├───────────────────────────────────────┼───────────────────────────────────────────────┤
 │ phase2/common/hook_utils.py           │ ablation hook 逻辑（重写为模型无关版本）      │
 ├───────────────────────────────────────┼───────────────────────────────────────────────┤
 │ phase3/common/model_adapters.py       │ adapter 基础设施（扩展 generate 方法）        │
 ├───────────────────────────────────────┼───────────────────────────────────────────────┤
 │ results/phase3/*/exp_3a_directions.pt │ refusal directions + NW layer                 │
 └───────────────────────────────────────┴───────────────────────────────────────────────┘

 Verification

 1. LLaVA sanity check: 3C 结果应接近 Phase 2 Exp 2B:
   - baseline_mm bypass ≈ 60%, ablation_nw_vmm full_harmful ≈ 89%, ablation_all_vmm ≈ 74%
   - ablation_nw > ablation_all 必须成立
 2. Direction 加载: 打印 v_mm.shape, v_mm.norm() 确认方向有效
 3. Generation 质量: 每个 config 第一个 prompt 打印完整 response 便于人工检查
 4. Qwen2.5-VL 环境: 必须用 qwen3-vl 环境（transformers >= 4.52）

 运行优先级

 1. LLaVA (GPU 0) — sanity check，与 Phase 2 对比
 2. Qwen2.5-VL (GPU 1) — Group B 代表，最关键的新数据
 3. InternVL2 (GPU 2) — Group B 交叉验证
 4. InstructBLIP (GPU 3) — cos 偏低，ablation 效果可能差

 用户确认

 - Generate 方法：在 model_adapters.py 中添加（扩展现有 adapter）
 - 测试数据：先用 8 个 TEST_PROMPTS，快速出结果

 预期科学问题

 - 如果 Group B 的 NW ablation > all-layer ablation: narrow waist 是通用现象 (strong claim)
 - 如果 Group B 的 NW ablation ≈ all-layer ablation: narrow waist 与 amplitude reversal 相关，仅在 Group A 有效
 - v_text vs v_mm: 如果 v_text 在 MM 设定下也有效，说明 direction 确实跨模态稳定