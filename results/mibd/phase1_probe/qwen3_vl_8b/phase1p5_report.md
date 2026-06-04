# Phase 1.5 Probe Validity Audit: /inspire/hdd/global_user/wenming-253108090054/models/Qwen3-VL-8B

Signal: `refusal`

## Condition: V-text

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 1.0000 |
| Held-out (random 20%) | 0.5926 |
| Group split (by paired_id) | N/A (no paired ids) |
| Train-only held-out AUC | 0.7778 (locus: layer=5 pos=-8) |
| Permutation (nested) | 0.4448 ± 0.2317 (p95=0.8167, n=100) |

### Cross-Category AUCs

| Held-out Category | AUC |
|---|:---:|
| Disinformation and deception | N/A (single-class test set) |
| Economic harm | N/A (single-class test set) |
| Expert advice | 0.5000 |
| Fraud/Deception | 0.0000 |
| Government decision-making | 0.5000 |
| Harassment/Discrimination | N/A (single-class test set) |
| Hate, harassment and discrimination | 0.5000 |
| Illegal goods and services | 0.4167 |
| Non-violent crimes | N/A (single-class test set) |
| Physical harm | 1.0000 |
| Privacy | 0.0000 |
| Sexual content | N/A (single-class test set) |
| Violence | 0.5000 |
| chemical_biological | N/A (single-class test set) |
| cybercrime_intrusion | N/A (single-class test set) |
| general | N/A (single-class test set) |
| harassment_bullying | N/A (single-class test set) |
| harmful | N/A (single-class test set) |
| illegal | N/A (single-class test set) |
| misinformation_disinformation | N/A (single-class test set) |

### Margin Statistics

| Metric | Value |
|---|---:|
| Mean gap (harmful − harmless) | 11.9572 |
| Median gap | 11.4381 |
| IQR harmful | 4.8919 |
| IQR harmless | 3.9132 |
| N harmful | 17 |
| N harmless | 45 |

### Static Transfer Margin Drop (V-text probe → other conditions)

| Target Condition | Margin Drop |
|---|---:|
| V-blank | 0.2210 |
| V-noise | 0.3279 |
| V-real | 0.4087 |

**Audit verdict:** WARN — held-out AUC low (0.593); large train/held-out gap (0.407) — possible overfit

---

## Condition: V-blank

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 1.0000 |
| Held-out (random 20%) | 0.5556 |
| Group split (by paired_id) | N/A (no paired ids) |
| Train-only held-out AUC | 0.6667 (locus: layer=1 pos=-8) |
| Permutation (nested) | 0.4633 ± 0.2378 (p95=0.8537, n=100) |

### Margin Statistics

| Metric | Value |
|---|---:|
| Mean gap (harmful − harmless) | 11.7925 |
| Median gap | 11.4588 |
| IQR harmful | 5.2036 |
| IQR harmless | 3.9555 |
| N harmful | 17 |
| N harmless | 45 |

**Audit verdict:** WARN — held-out AUC low (0.556); large train/held-out gap (0.444) — possible overfit

---

## Condition: V-noise

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 1.0000 |
| Held-out (random 20%) | 0.6296 |
| Group split (by paired_id) | N/A (no paired ids) |
| Train-only held-out AUC | 0.7778 (locus: layer=4 pos=-8) |
| Permutation (nested) | 0.4615 ± 0.2387 (p95=0.8537, n=100) |

### Margin Statistics

| Metric | Value |
|---|---:|
| Mean gap (harmful − harmless) | 11.7336 |
| Median gap | 11.2066 |
| IQR harmful | 4.4926 |
| IQR harmless | 3.9200 |
| N harmful | 17 |
| N harmless | 45 |

**Audit verdict:** WARN — held-out AUC low (0.630); large train/held-out gap (0.370) — possible overfit

---

## Condition: V-real

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 1.0000 |
| Held-out (random 20%) | 0.5556 |
| Group split (by paired_id) | N/A (no paired ids) |
| Train-only held-out AUC | 0.7778 (locus: layer=4 pos=-8) |
| Permutation (nested) | 0.4456 ± 0.2405 (p95=0.8519, n=100) |

### Margin Statistics

| Metric | Value |
|---|---:|
| Mean gap (harmful − harmless) | 11.7179 |
| Median gap | 11.1212 |
| IQR harmful | 5.4627 |
| IQR harmless | 4.1297 |
| N harmful | 17 |
| N harmless | 45 |

**Audit verdict:** WARN — held-out AUC low (0.556); large train/held-out gap (0.444) — possible overfit

---

## Overall Audit Conclusion

4/4 condition(s) flagged with warnings. Review per-condition sections above before proceeding to Phase 2.
