# ARA-OPD-VLM 诊断阶段 — 会话交接文档

**更新时间**：2026-04-27
**当前状态**：CPU 端代码全部完成，待 qi 在 GPU 容器 + 网络环境执行剩余任务

---

## 一、已完成的工作

| 任务 | 内容 | 状态 |
|---|---|---|
| Task 4 | Qwen3-VL model adapter（`qwen3_vlm_model.py` + `model_factory.py` 注册） | ✅ 完成，commit `bc6025d` |
| Task 8 | `compute_alignment.py`（余弦矩阵计算，已用 PCD 数据验证） | ✅ 完成 |
| Task 9 | `projector_causal_test.py`（projector 因果测试，forward pass bug 已修复） | ✅ 完成 |
| Task 10 | `aggregate_diag.py`（汇总脚本，生成 target_modules.json） | ✅ 完成 |

compute_alignment.py 验证结果（Qwen2.5-VL，与 PCD 一致）：
- c1（LLM vs V-text）= 0.671
- c2（V-text vs V-blank）= 0.804
- c3（LLM vs V-blank）= 0.492
- 级联预测 c1×c2 = 0.539，误差 -0.047 → **两效应接近级联**

---

## 二、待执行任务（qi 手动操作）

### Task 1 & 2：venv 重建 + heretic probe（GPU 容器）

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal

# Task 1：重建 venv
rm -rf .venv_gemma_probe
bash install_offline.sh
# 验证：ls .venv_gemma_probe/lib/python3.12/site-packages/ | wc -l  → 应 > 50
CUDA_VISIBLE_DEVICES=0 .venv_gemma_probe/bin/python experiments/ara_sapp/smoke_test.py
# 期望最后一行：READY FOR FULL RUN.

# Task 2：启动 heretic probe（nohup，约 2h）
mkdir -p results/ara_opd_vlm_0427/cross_modal_geometry_diag/heretic_probe
nohup .venv_gemma_probe/bin/python \
    experiments/ara_sapp/exp_gemma4_heretic_probe.py all --n 50 \
    --output results/ara_opd_vlm_0427/cross_modal_geometry_diag/heretic_probe/heretic_probe_n50.json \
    > results/ara_opd_vlm_0427/cross_modal_geometry_diag/heretic_probe/probe.log 2>&1 &
echo $! > results/ara_opd_vlm_0427/cross_modal_geometry_diag/heretic_probe/probe.pid
```

### Task 3：下载 Qwen3-VL-8B-Instruct（有网络的环境）

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir /inspire/hdd/global_user/wenming-253108090054/models/Qwen3-VL-8B-Instruct \
    --local-dir-use-symlinks False
```

완료 후 확인：
```bash
python3 -c "
import json
cfg = json.load(open('/inspire/hdd/global_user/wenming-253108090054/models/Qwen3-VL-8B-Instruct/config.json'))
print('architectures:', cfg.get('architectures'))
print('num_hidden_layers:', cfg.get('num_hidden_layers') or cfg.get('text_config', {}).get('num_hidden_layers'))
"
```

**중요**: `architectures`가 `Qwen2_5_VLForConditionalGeneration`이 아닌 새 클래스명이면, `qwen3_vlm_model.py`의 import를 수정해야 합니다. (현재 `Qwen2_5_VLForConditionalGeneration` 가정으로 작성됨)

### Task 4 추가 확인：adapter 冒烟测试（GPU 컨테이너, 모델 다운로드 후）

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
conda run -n qwen3-vl python3 -c "
import sys
sys.path.insert(0, 'refusal_direction')
from pipeline.model_utils.model_factory import construct_model_base
m = construct_model_base(
    '/inspire/hdd/global_user/wenming-253108090054/models/Qwen3-VL-8B-Instruct',
    model_name='qwen3vl_8b'
)
print('model_type:', type(m).__name__)
print('num_layers:', len(m._get_model_block_modules()))
print('PASS')
"
```

### Task 5, 6, 7：Qwen3-VL Sweep → Ablate → Evaluate（GPU，순차 실행）

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal

# Task 5: sweep (3 조건, 조건당 약 30-60분)
bash experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/launch_qwen3vl_sweep.sh
# 완료 확인:
for COND in V-text V-blank V-noise; do
    F="results/ara_opd_vlm_0427/cross_modal_geometry_diag/qwen3vl/$COND/sweep/best_layer.json"
    [ -f "$F" ] && echo "$COND: $(cat $F)" || echo "$COND: NOT DONE"
done

# Task 6: ablate (sweep 완료 후)
bash experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/launch_qwen3vl_ablate.sh
# 완료 확인:
for COND in V-text V-blank V-noise; do
    F="results/ara_opd_vlm_0427/cross_modal_geometry_diag/qwen3vl/$COND/dim_responses.json"
    [ -f "$F" ] && echo "$COND OK" || echo "$COND: NOT DONE"
done

# Task 7: evaluate (ablate 완료 후)
bash experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/launch_qwen3vl_evaluate.sh
# 완료 확인:
for COND in V-text V-blank V-noise; do
    F="results/ara_opd_vlm_0427/cross_modal_geometry_diag/qwen3vl/$COND/dim_eval.json"
    [ -f "$F" ] && python3 -c "
import json; d=json.load(open('$F'))
print('$COND: asr_kw=', d.get('asr_kw'), 'asr_lg3=', d.get('asr_lg3'))
" || echo "$COND: NOT DONE"
done
```

### Task 8 재실행（sweep 완료 후, CPU에서）

Qwen3-VL sweep이 완료되면 compute_alignment.py를 다시 실행하면 Qwen3-VL의 c2도 나옵니다:

```bash
python3 experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/compute_alignment.py
```

### Task 9：Projector 因果测试（GPU, ablate 완료 후）

```bash
bash experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/launch_projector_test.sh
# 결과 확인:
cat results/ara_opd_vlm_0427/cross_modal_geometry_diag/projector_causal_test.json
```

### Task 10：최종 汇总（모든 GPU 작업 완료 후, CPU에서）

```bash
python3 experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/aggregate_diag.py
# 결과: target_modules.json + HANDOFF.md(이 파일) 갱신됨
```

---

## 三、关键文件位置

| 파일 | 경로 |
|---|---|
| spec | `docs/superpowers/specs/2026-04-27-ara-opd-vlm-design.md` |
| plan | `docs/superpowers/plans/2026-04-27-ara-opd-vlm-diag.md` |
| Qwen3-VL adapter | `refusal_direction/pipeline/model_utils/qwen3_vlm_model.py` |
| compute_alignment | `experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/compute_alignment.py` |
| projector_causal_test | `experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/projector_causal_test.py` |
| aggregate_diag | `experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/aggregate_diag.py` |
| 현재 alignment 결과 | `results/ara_opd_vlm_0427/cross_modal_geometry_diag/cross_modal_alignment.json` |

---

## 四、已知注意事项

1. **Qwen3-VL HF 클래스명**: `qwen3_vlm_model.py`는 `Qwen2_5_VLForConditionalGeneration`으로 작성됨. 다운로드 후 config.json `architectures` 필드 확인 필수. 새 클래스면 import 수정 필요.

2. **venv 미설치 상태**: `.venv_gemma_probe/`는 디렉토리만 존재, 패키지 미설치. Task 1의 `rm -rf + bash install_offline.sh` 필수.

3. **Gemma-3 c2**: PCD 데이터 기준 c2(V-text vs V-blank) = -0.006. 이는 Gemma가 V-blank 조건에서 best layer가 1로 점프(PCD findings §8.1)했기 때문으로, 서로 다른 층의 방향을 비교하는 것. 분석 시 주의.

4. **sweep 3개 조건 동시 실행**: GPU 메모리 충분한지 확인. 모자라면 순차 실행으로 변경.

---

## 五、診断完了 후 다음 단계

모든 GPU 작업 완료 → `aggregate_diag.py` 실행 → `target_modules.json` 생성 완료 시:
- **ARA 실험 plan 작성** (ara_vlm/ 서브디렉토리)
- target_modules.json의 `ara_target_modules_decision` 값에 따라 ARA 공격 모듈 범위 결정
