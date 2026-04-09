# Category A 实验运行指南

> 在 GPU 节点执行。所有命令从项目根目录运行。
> 项目根目录: `/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal`

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
conda activate rdo
```

---

## 0. 初始化：创建结果目录

```bash
mkdir -p results/category_a/llava_7b
mkdir -p results/category_a/llava_13b
mkdir -p results/category_a/qwen2vl_7b
mkdir -p results/category_a/qwen2vl_32b
mkdir -p results/category_a/internvl2_8b
mkdir -p results/phase3/llava_13b
mkdir -p results/phase3/qwen2vl_32b
```

---

## 1. Smoke Test（先用 n=10 验证整个流程，再跑正式实验）

### 1.1 A1 Smoke Test（LLaVA-7B，10 prompts，baseline_mm only）

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "
import sys
sys.path.insert(0, 'experiments/category_a')
import common.data_utils as du
_orig = du.load_saladbench_test
du.load_saladbench_test = lambda: _orig()[:10]
du.load_dataset = lambda name: du.load_saladbench_test()
from exp_a1_dsa_validation import run_a1
run_a1('llava_7b', 'cuda:0', 'saladbench', max_new_tokens=50)
"
```

验证输出：
```bash
python -c "
import json
with open('results/category_a/llava_7b/a1_baseline_mm_saladbench.json') as f:
    d = json.load(f)
print('n_prompts:', d['n_prompts'])
print('FHCR_kw:', d['metrics_kw']['full_harmful_completion_rate'])
print('response keys:', list(d['responses'][0].keys()))
print('response is full text (not truncated):', len(d['responses'][0]['response']) > 0)
"
```

期望输出：`n_prompts: 10`，`response keys` 包含 `response`，无报错。

### 1.2 A3 Smoke Test（LLaVA-7B，10 prompts）

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "
import sys
sys.path.insert(0, 'experiments/category_a')
import common.data_utils as du
_orig = du.load_saladbench_test
du.load_saladbench_test = lambda: _orig()[:10]
from exp_a3_norm_prediction import run_a3
run_a3('llava_7b', 'cuda:0', max_new_tokens=50)
"
```

验证输出：
```bash
python -c "
import json
with open('results/category_a/llava_7b/a3_norm_prediction.json') as f:
    d = json.load(f)
print('n_prompts:', d['n_prompts'])
seq0 = d['sequences'][0]
print('norms length:', len(seq0['norms']))
print('norms[:5]:', seq0['norms'][:5])
print('eval SC:', seq0['eval']['self_correction_found'])
"
```

期望输出：`norms length > 0`，norms 为浮点数列表，无报错。

### 1.3 清理 Smoke Test 结果

```bash
rm -f results/category_a/llava_7b/a1_*_saladbench.json
rm -f results/category_a/llava_7b/a1_progress_saladbench.json
rm -f results/category_a/llava_7b/a3_norm_prediction.json
```

**Smoke test 通过后，继续正式实验。**

---

## 2. Phase 0：提取新模型 Directions（LLaVA-13B 下载后执行）

两个模型并行，各占一张 GPU：

```bash
# GPU 0: LLaVA-13B（~26GB，需先下载 llava-hf/llava-1.5-13b-hf）
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup python experiments/phase3/exp_3a_amplitude_reversal.py \
    --model llava_13b --device cuda:0 \
    > results/phase3/llava_13b/exp_3a.log 2>&1 &
echo "Phase 0 LLaVA-13B PID: $!"

# GPU 3: Qwen-32B（~64GB，公共目录已有）
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup python experiments/phase3/exp_3a_amplitude_reversal.py \
    --model qwen2vl_32b --device cuda:3 \
    > results/phase3/qwen2vl_32b/exp_3a.log 2>&1 &
echo "Phase 0 Qwen-32B PID: $!"
```

监控进度：
```bash
tail -f results/phase3/llava_13b/exp_3a.log
tail -f results/phase3/qwen2vl_32b/exp_3a.log
```

验证完成：
```bash
ls results/phase3/llava_13b/exp_3a_directions.pt
ls results/phase3/qwen2vl_32b/exp_3a_directions.pt
```

---

## 3. Phase 1：A1 正式实验（5 models，572 prompts）+ A3 同步

### 3.1 A1 生成（4 GPUs 并行）

```bash
bash experiments/category_a/run_a1_gen.sh
```

内部会并行启动：
- GPU 0: LLaVA-7B A1
- GPU 1: LLaVA-13B A1
- GPU 2: Qwen-7B A1 → InternVL2-8B A1（串行）
- GPU 3: Qwen-32B A1

监控：
```bash
tail -f results/category_a/llava_7b/a1_gen.log
tail -f results/category_a/qwen2vl_32b/a1_gen.log
```

### 3.2 A3 Norm Prediction（LLaVA-7B 完成后，或在 LLaVA-7B A1 运行时用另一张卡）

```bash
bash experiments/category_a/run_a3.sh
```

如果 GPU 0 被 LLaVA-7B A1 占用，可改为其他空闲卡：
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup python experiments/category_a/exp_a3_norm_prediction.py \
    --model llava_7b --device cuda:1 \
    > results/category_a/llava_7b/a3.log 2>&1 &
```

---

## 4. Phase 2：A2 因果验证（A1 完成后）

```bash
bash experiments/category_a/run_a2.sh
```

内部并行：
- GPU 0: LLaVA-7B A2
- GPU 2: Qwen-7B A2
- GPU 3: InternVL2-8B A2

监控：
```bash
tail -f results/category_a/llava_7b/a2.log
tail -f results/category_a/qwen2vl_7b/a2.log
tail -f results/category_a/internvl2_8b/a2.log
```

---

## 5. Phase 3：Qwen3Guard 评估（A1 完成后，切换 transformers>=4.51 环境）

```bash
conda activate <你的 transformers451 环境名>
bash experiments/category_a/run_a1_judge.sh
```

或手动逐模型运行（方便调试）：
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python experiments/category_a/exp_a1_judge.py \
    --model llava_7b --judge qwen3guard --dataset saladbench --device cuda:0
```

---

## 6. 验证最终结果

A1 完成后检查每个模型的结果：
```bash
python -c "
import json, glob
for path in sorted(glob.glob('results/category_a/*/a1_baseline_mm_saladbench.json')):
    with open(path) as f:
        d = json.load(f)
    m = d['metrics_kw']
    print(f'{d[\"model\"]}: IBR={m[\"initial_bypass_rate\"]:.3f} SCR={m[\"self_correction_rate_overall\"]:.3f} FHCR_kw={m[\"full_harmful_completion_rate\"]:.3f}')
"
```

A3 完成后检查 AUROC：
```bash
python -c "
import json
with open('results/category_a/llava_7b/a3_norm_prediction.json') as f:
    d = json.load(f)
a = d['analysis']
print('AUROC max_norm:', a.get('auroc_max_norm'))
print('AUROC mean_norm:', a.get('auroc_mean_norm'))
print('Spike precedes SC:', a.get('spike_precedes_sc_rate'))
print('n_sc:', a.get('n_sc'), '/', a.get('n_sequences'))
"
```

A2 完成后检查 SCR 三组对比：
```bash
python -c "
import json
for model in ['llava_7b', 'qwen2vl_7b', 'internvl2_8b']:
    path = f'results/category_a/{model}/a2_causality.json'
    with open(path) as f:
        d = json.load(f)
    print(f'\n=== {model} (strategy={d[\"ablation_strategy\"]}) ===')
    for gname, gdata in d['groups'].items():
        m = gdata['metrics']
        print(f'  {gname:<25} SCR={m[\"self_correction_rate_overall\"]:.3f}  FHCR={m[\"full_harmful_completion_rate\"]:.3f}')
"
```

---

## 7. 常见问题

**A3 报错 `past_key_values` 或 `language_model`**：
把完整报错贴给 Claude，A3 的手动 token-by-token 生成是最容易出架构兼容问题的地方。

**A2 报错 A1 results not found**：
确认 A1 对应 config 已完成。LLaVA 需要 `a1_ablation_nw_vmm_saladbench.json`，Qwen 需要 `a1_ablation_all_vmm_saladbench.json`。

**OOM（显存不足）**：
- LLaVA-7B/Qwen-7B/InternVL2: 应该单卡 16-20GB，H100 足够
- Qwen-32B: 需要约 64GB，H100 80GB 勉强可行，如果 OOM 试 `--device cuda:3` 并确保该卡空闲

**断点续跑 A1**：
如果 A1 中途中断，加 `--resume` 参数重启即可，会从上次中断处继续。
