# ASR Summary — Repro Arditi-Wollschläger

Generated from: results/repro_arditi_wollschlager/evaluation.json

## qwen2.5_7b

Config               |  ASR_kw | ASR_LG3 |  ASR_SR |      SRR | n
-----------------------------------------------------------------
dim_ablation         | 100.0% |  94.5% |  99.2% |  +5.5pp | 128
cone_k3              |  98.4% |  93.8% |  98.4% |  +4.7pp | 128
cone_k5              |  99.2% |  92.2% |  99.2% |  +7.0pp | 128
rdo_k1               |  90.6% |  76.6% |  88.3% | +14.1pp | 128

**Best SRR: +14.1pp @ rdo_k1**

## llama3.1_8b

Config               |  ASR_kw | ASR_LG3 |  ASR_SR |      SRR | n
-----------------------------------------------------------------
dim_ablation         |  98.4% |  89.8% |  98.4% |  +8.6pp | 128
cone_k3              | 100.0% |  90.6% | 100.0% |  +9.4pp | 128
cone_k5              | 100.0% |  91.4% | 100.0% |  +8.6pp | 128
rdo_k1               | 100.0% |  91.4% | 100.0% |  +8.6pp | 128

**Best SRR: +9.4pp @ cone_k3**

---

## 附带观测：LLM 上的 SRR

注：VLM 也有内生对齐训练，本节仅作对照记录，不作 narrative 分叉依据。

- **qwen2.5_7b**: best SRR = +14.1pp @ `rdo_k1`
- **llama3.1_8b**: best SRR = +9.4pp @ `cone_k3`

