Phase 3 Implementation Plan — Exp 3A Only (First Pass)                                                  
                                                                                                         
 Context                                                                                                 
                                                                                                         
 Phase 1/2 在 LLaVA-1.5-7B 上发现 Amplitude Reversal (浅层压制、深层放大) 和 Narrow Waist (layer 16)。
 Phase 3 目标：在 4 个架构多样的 VLM 上验证 generality。
 本轮 scope: 只实现 Exp 3A (Amplitude Reversal + Narrow Waist 验证)，看结果后再决定 3B/3C。

 Pre-requisite

 qi 手动下载到 /inspire/hdd/global_user/wenming-253108090054/models/:
 - Qwen/Qwen2.5-VL-7B-Instruct
 - OpenGVLab/InternVL2-8B
 - Salesforce/instructblip-vicuna-7b

 File Structure (最小化)

 experiments/phase3/
 ├── common/
 │   ├── __init__.py
 │   ├── model_configs.py      # 4 模型配置 + 统一加载接口
 │   └── model_adapters.py     # 每个模型的输入准备 + hidden state 提取适配
 ├── exp_3a_amplitude_reversal.py   # Exp 3A 主脚本 (self-contained)
 └── run_3a_all.sh                  # 4 GPU 并行

 results/phase3/{llava_7b,qwen2vl_7b,internvl2_8b,instructblip_7b}/
     ├── exp_3a_results.json
     └── exp_3a_directions.pt       # 保存 directions 供后续 3B/3C 复用

 Implementation

 File 1: common/model_configs.py

 MODEL_CONFIGS dict，每个模型包含:
 - model_path: 本地路径 (models/ 或 hub 缓存)
 - total_layers, probe_layers (相对深度 ~25%/40%/50%/64%/86%)
 - blank_image_size, model_class
 - hidden_dim (用于 sanity check)

 load_model_by_name(name, device) 函数:
 - LLaVA: LlavaForConditionalGeneration, AutoProcessor, 从 hub 缓存加载
 - Qwen2.5-VL: Qwen2_5_VLForConditionalGeneration, AutoProcessor, 从 models/ 加载
 - InternVL2: AutoModel(trust_remote_code=True), AutoTokenizer, 从 models/ 加载
 - InstructBLIP: InstructBlipForConditionalGeneration, InstructBlipProcessor, 从 models/ 加载
 - 所有: local_files_only=True, torch_dtype=torch.bfloat16, device_map=device

 File 2: common/model_adapters.py

 基类 ModelAdapter + 4 个子类，每个实现:

 class ModelAdapter:
     def prepare_mm_inputs(self, prompt, image) -> dict    # 多模态输入
     def prepare_text_inputs(self, prompt) -> dict          # 纯文本输入
     def get_llm_layers(self) -> nn.ModuleList              # LLM backbone layers
     def get_visual_token_count(self, inputs) -> int        # visual token 数
     def get_last_token_hidden(self, hidden_states, layer_idx) -> Tensor  # 最后 text token

 关键适配差异:
 - LLaVA: text-only 通过 model.language_model() forward; mm 通过 model() forward; visual=576
 - Qwen2.5-VL: text-only 不拆分 backbone, 直接用无图像 chat template forward; layers 在
 model.model.layers; visual 动态
 - InternVL2: text-only 通过 model.language_model() forward; layers 在
 model.language_model.model.layers; visual=256; trust_remote_code
 - InstructBLIP: text-only 通过 model.language_model() forward; layers 在
 model.language_model.model.layers; visual=32

 File 3: exp_3a_amplitude_reversal.py

 主逻辑 (基于补充文档 Part 2.3):
 1. 加载模型 + adapter
 2. 12 对 paired prompts × 2 conditions (text/mm) × all probe_layers
 3. Hook-based extraction: 一次 forward 捕获所有 probe_layers 的 hidden states
 4. 计算 mean-difference direction per layer per condition
 5. 指标: cos(v_text, v_mm), norm_text, norm_mm, norm_ratio
 6. 确定 narrow_waist (cos 最高层), amplitude_reversal (浅层 ratio<1 且深层 ratio>1)

 CLI: python exp_3a_amplitude_reversal.py --model llava_7b --device cuda:0

 输出:
 - exp_3a_results.json: 所有层的 metrics + narrow_waist + reversal 判断
 - exp_3a_directions.pt: 每层的 v_text, v_mm (normalized) 供 3B/3C 使用

 File 4: run_3a_all.sh

 # 4 GPU 并行
 CUDA_VISIBLE_DEVICES=0 python exp_3a... --model llava_7b --device cuda:0 &
 CUDA_VISIBLE_DEVICES=1 python exp_3a... --model qwen2vl_7b --device cuda:0 &
 CUDA_VISIBLE_DEVICES=2 python exp_3a... --model internvl2_8b --device cuda:0 &
 CUDA_VISIBLE_DEVICES=3 python exp_3a... --model instructblip_7b --device cuda:0 &
 wait

 注意: CUDA_VISIBLE_DEVICES=X + --device cuda:0 (映射后只可见一张卡)

 Critical Reuse

 - experiments/phase2/common/llava_utils.py: PAIRED_PROMPTS (12对), TEST_PROMPTS
 - results/exp_a_directions.pt: 可选加载验证 LLaVA 结果一致性
 - experiments/phase2/common/hook_utils.py: ablation hook pattern (3C 时复用)

 Verification

 1. LLaVA-7B sanity check: 3A 结果应与 Pilot Exp A 一致:
   - Layer 16 cos ≈ 0.918, narrow_waist = 16
   - Layer 12 norm_ratio ≈ 0.78 (< 1), Layer 20 norm_ratio ≈ 1.15 (> 1)
 2. Shape check: 每个模型首次 forward 打印 hidden_states shape 和 visual_token_count
 3. Direction norm check: v_text.norm() 和 v_mm.norm() 应 > 0 (若接近 0 说明 prompt 区分度不足)