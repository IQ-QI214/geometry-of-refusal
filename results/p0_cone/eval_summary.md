# P0 Cone Ablation — Evaluation Summary

| Model | Method | k | ASR_kw | ASR_sr | mean_sr | ASR_q3g | ASR_lg | SRR(q3g) | SRR(lg) |
|-------|--------|---|--------|--------|---------|---------|--------|----------|---------|
| llava_7b | dim | 1 | 0.539 | 0.141 | 0.086 | 0.125 | 0.320 | N/A | N/A |
| llava_7b | dim | 3 | 0.469 | 0.062 | 0.043 | 0.047 | 0.281 | N/A | N/A |
| llava_7b | dim | 5 | 0.641 | 0.070 | 0.044 | 0.055 | 0.352 | N/A | N/A |
| llava_7b | rdo | 1 | 0.602 | 0.086 | 0.057 | 0.039 | 0.398 | N/A | N/A |
| llava_7b | rdo | 3 | 0.664 | 0.117 | 0.069 | 0.070 | 0.438 | N/A | N/A |
| llava_7b | rdo | 5 | 0.734 | 0.117 | 0.079 | 0.062 | 0.477 | N/A | N/A |
| qwen2vl_7b | dim | 1 | 0.609 | 0.148 | 0.056 | 0.047 | 0.062 | N/A | N/A |
| qwen2vl_7b | dim | 3 | 1.000 | 0.164 | 0.051 | 0.078 | 0.133 | N/A | N/A |
| qwen2vl_7b | dim | 5 | 1.000 | 0.133 | 0.045 | 0.227 | 0.500 | N/A | N/A |
| qwen2vl_7b | rdo | 1 | 0.547 | 0.234 | 0.110 | 0.062 | 0.055 | N/A | N/A |
| qwen2vl_7b | rdo | 3 | 0.641 | 0.188 | 0.078 | 0.039 | 0.039 | N/A | N/A |
| qwen2vl_7b | rdo | 5 | 0.695 | 0.227 | 0.112 | 0.070 | 0.094 | N/A | N/A |

**Columns**: ASR_kw=keyword ASR, ASR_sr=StrongREJECT ASR, ASR_q3g=Qwen3Guard ASR, ASR_lg=LlamaGuard ASR, SRR=Stealth Refusal Rate (ASR_kw - ASR_judge). N/A=not computed or unavailable.
