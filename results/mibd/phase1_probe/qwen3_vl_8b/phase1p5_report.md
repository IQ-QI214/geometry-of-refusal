# Phase 1.5 Probe Validity Audit: /inspire/hdd/global_user/wenming-253108090054/models/Qwen3-VL-8B

Signal: `harmfulness`

## Condition: V-text

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 1.0000 |
| Held-out (random 20%) | 1.0000 |
| Group split (by paired_id) | N/A (no paired ids) |
| Train-only held-out AUC | 1.0000 (locus: layer=16 pos=-1) |
| Permutation (nested) | 0.5169 ± 0.1705 (p95=0.7885, n=100) |

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
| Illegal goods and services | N/A (single-class test set) |
| Non-violent crimes | N/A (single-class test set) |
| Physical harm | N/A (single-class test set) |
| Privacy | N/A (single-class test set) |
| Sexual content | N/A (single-class test set) |
| Violence | N/A (single-class test set) |
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
| Mean gap (harmful − harmless) | 24.9020 |
| Median gap | 25.2606 |
| IQR harmful | 8.2605 |
| IQR harmless | 5.4961 |
| N harmful | 64 |
| N harmless | 64 |

### Static Transfer Margin Drop (V-text probe → other conditions)

| Target Condition | Margin Drop |
|---|---:|
| FigStep | 20.0866 |
| V-blank | 5.9694 |
| V-noise | 4.5113 |
| V-real | 5.8369 |

**Audit verdict:** PASS — probe appears valid

---

## Condition: V-blank

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 0.9995 |
| Held-out (random 20%) | 0.9941 |
| Group split (by paired_id) | N/A (no paired ids) |
| Train-only held-out AUC | 1.0000 (locus: layer=17 pos=-1) |
| Permutation (nested) | 0.5137 ± 0.1668 (p95=0.7988, n=100) |

### Margin Statistics

| Metric | Value |
|---|---:|
| Mean gap (harmful − harmless) | 21.4197 |
| Median gap | 21.9328 |
| IQR harmful | 11.2975 |
| IQR harmless | 4.1529 |
| N harmful | 64 |
| N harmless | 64 |

**Audit verdict:** PASS — probe appears valid

---

## Condition: V-noise

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 0.9995 |
| Held-out (random 20%) | 1.0000 |
| Group split (by paired_id) | N/A (no paired ids) |
| Train-only held-out AUC | 1.0000 (locus: layer=16 pos=-1) |
| Permutation (nested) | 0.5047 ± 0.1755 (p95=0.7932, n=100) |

### Margin Statistics

| Metric | Value |
|---|---:|
| Mean gap (harmful − harmless) | 22.3189 |
| Median gap | 22.5420 |
| IQR harmful | 8.5537 |
| IQR harmless | 4.7865 |
| N harmful | 64 |
| N harmless | 64 |

**Audit verdict:** PASS — probe appears valid

---

## Condition: V-real

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 0.9998 |
| Held-out (random 20%) | 1.0000 |
| Group split (by paired_id) | N/A (no paired ids) |
| Train-only held-out AUC | 1.0000 (locus: layer=17 pos=-1) |
| Permutation (nested) | 0.4914 ± 0.1640 (p95=0.7589, n=100) |

### Margin Statistics

| Metric | Value |
|---|---:|
| Mean gap (harmful − harmless) | 21.8823 |
| Median gap | 21.8169 |
| IQR harmful | 8.6950 |
| IQR harmless | 4.0616 |
| N harmful | 64 |
| N harmless | 64 |

**Audit verdict:** PASS — probe appears valid

---

## Condition: FigStep

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 1.0000 |
| Held-out (random 20%) | 1.0000 |
| Group split (by paired_id) | N/A (no paired ids) |
| Train-only held-out AUC | 1.0000 (locus: layer=3 pos=-1) |
| Permutation (nested) | 0.5154 ± 0.2044 (p95=0.8467, n=100) |

### Margin Statistics

| Metric | Value |
|---|---:|
| Mean gap (harmful − harmless) | 23.1043 |
| Median gap | 23.1291 |
| IQR harmful | 1.0188 |
| IQR harmless | 5.5550 |
| N harmful | 102 |
| N harmless | 64 |

**Audit verdict:** PASS — probe appears valid

---

## Overall Audit Conclusion

All 5 condition(s) passed validity checks. Probes pass the implemented validity checks and generalize to random held-out splits; group/category controls remain unavailable under the current dataset structure.
