# Phase 1.5 Probe Validity Audit: /inspire/hdd/global_user/wenming-253108090054/models/InternVL3-8B

Signal: `refusal`

## Condition: V-text

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 1.0000 |
| Held-out (random 20%) | 0.9091 |
| Group split (by paired_id) | N/A (no paired ids) |
| Train-only held-out AUC | 0.9091 (locus: layer=0 pos=-8) |
| Permutation (nested) | 0.4618 ± 0.2925 (p95=1.0000, n=100) |

### Cross-Category AUCs

| Held-out Category | AUC |
|---|:---:|
| Disinformation and deception | N/A (single-class test set) |
| Economic harm | N/A (single-class test set) |
| Expert advice | N/A (single-class test set) |
| Fraud/Deception | N/A (single-class test set) |
| Government decision-making | N/A (single-class test set) |
| Harassment/Discrimination | N/A (single-class test set) |
| Hate, harassment and discrimination | N/A (single-class test set) |
| Illegal goods and services | 1.0000 |
| Non-violent crimes | N/A (single-class test set) |
| Physical harm | N/A (single-class test set) |
| Privacy | N/A (single-class test set) |
| Sexual content | N/A (single-class test set) |
| Violence | N/A (single-class test set) |
| chemical_biological | N/A (single-class test set) |
| cybercrime_intrusion | N/A (single-class test set) |
| general | N/A (single-class test set) |
| harassment_bullying | 0.0000 |
| harmful | N/A (single-class test set) |
| illegal | N/A (single-class test set) |
| misinformation_disinformation | 0.5000 |

### Margin Statistics

| Metric | Value |
|---|---:|
| Mean gap (harmful − harmless) | 5.3509 |
| Median gap | 5.5253 |
| IQR harmful | 0.5403 |
| IQR harmless | 1.2488 |
| N harmful | 56 |
| N harmless | 6 |

### Static Transfer Margin Drop (V-text probe → other conditions)

| Target Condition | Margin Drop |
|---|---:|
| V-blank | 0.0180 |
| V-noise | 0.0068 |
| V-real | 0.0310 |

**Audit verdict:** PASS — probe appears valid

---

## Condition: V-blank

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 1.0000 |
| Held-out (random 20%) | 0.9091 |
| Group split (by paired_id) | N/A (no paired ids) |
| Train-only held-out AUC | 0.9091 (locus: layer=0 pos=-8) |
| Permutation (nested) | 0.4864 ± 0.2890 (p95=1.0000, n=100) |

### Margin Statistics

| Metric | Value |
|---|---:|
| Mean gap (harmful − harmless) | 5.4254 |
| Median gap | 5.6445 |
| IQR harmful | 0.6096 |
| IQR harmless | 0.9935 |
| N harmful | 56 |
| N harmless | 6 |

**Audit verdict:** PASS — probe appears valid

---

## Condition: V-noise

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 1.0000 |
| Held-out (random 20%) | 0.9091 |
| Group split (by paired_id) | N/A (no paired ids) |
| Train-only held-out AUC | 0.9091 (locus: layer=0 pos=-8) |
| Permutation (nested) | 0.4700 ± 0.2901 (p95=1.0000, n=100) |

### Margin Statistics

| Metric | Value |
|---|---:|
| Mean gap (harmful − harmless) | 5.4381 |
| Median gap | 5.6867 |
| IQR harmful | 0.6091 |
| IQR harmless | 0.9785 |
| N harmful | 56 |
| N harmless | 6 |

**Audit verdict:** PASS — probe appears valid

---

## Condition: V-real

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 1.0000 |
| Held-out (random 20%) | 0.9091 |
| Group split (by paired_id) | N/A (no paired ids) |
| Train-only held-out AUC | 0.9091 (locus: layer=0 pos=-8) |
| Permutation (nested) | 0.4782 ± 0.2945 (p95=1.0000, n=100) |

### Margin Statistics

| Metric | Value |
|---|---:|
| Mean gap (harmful − harmless) | 5.4152 |
| Median gap | 5.6271 |
| IQR harmful | 0.6038 |
| IQR harmless | 0.9530 |
| N harmful | 56 |
| N harmless | 6 |

**Audit verdict:** PASS — probe appears valid

---

## Overall Audit Conclusion

All 4 condition(s) passed validity checks. Probes pass the implemented validity checks and generalize to random held-out splits; group/category controls remain unavailable under the current dataset structure.
