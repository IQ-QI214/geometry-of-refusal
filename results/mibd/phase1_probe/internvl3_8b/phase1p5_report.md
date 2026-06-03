# Phase 1.5 Probe Validity Audit: /inspire/hdd/global_user/wenming-253108090054/models/InternVL3-8B

Signal: `harmfulness`

## Condition: V-text

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 1.0000 |
| Held-out (random 20%) | 1.0000 |
| Group split (by paired_id) | N/A (no paired ids) |
| Permutation (mean over 100) | 0.5422 |

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
| Mean gap (harmful − harmless) | 6.3247 |
| Median gap | 6.4955 |
| IQR harmful | 2.0643 |
| IQR harmless | 2.7206 |
| N harmful | 64 |
| N harmless | 64 |

### Static Transfer Margin Drop (V-text probe → other conditions)

| Target Condition | Margin Drop |
|---|---:|
| FigStep | 5.5133 |
| V-blank | 1.2269 |
| V-noise | 1.5213 |
| V-real | 2.5011 |

**Audit verdict:** PASS — probe appears valid

---

## Condition: V-blank

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 0.9998 |
| Held-out (random 20%) | 1.0000 |
| Group split (by paired_id) | N/A (no paired ids) |
| Permutation (mean over 100) | 0.6094 |

### Margin Statistics

| Metric | Value |
|---|---:|
| Mean gap (harmful − harmless) | 5.8085 |
| Median gap | 6.2395 |
| IQR harmful | 1.6075 |
| IQR harmless | 2.1088 |
| N harmful | 64 |
| N harmless | 64 |

**Audit verdict:** WARN — permutation AUC high (0.609) — possible artifact

---

## Condition: V-noise

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 0.9995 |
| Held-out (random 20%) | 0.9822 |
| Group split (by paired_id) | N/A (no paired ids) |
| Permutation (mean over 100) | 0.6079 |

### Margin Statistics

| Metric | Value |
|---|---:|
| Mean gap (harmful − harmless) | 5.7641 |
| Median gap | 6.1760 |
| IQR harmful | 1.4194 |
| IQR harmless | 2.2004 |
| N harmful | 64 |
| N harmless | 64 |

**Audit verdict:** WARN — permutation AUC high (0.608) — possible artifact

---

## Condition: V-real

### Split AUCs

| Split | AUC |
|---|:---:|
| Train (full) | 0.9990 |
| Held-out (random 20%) | 0.9941 |
| Group split (by paired_id) | N/A (no paired ids) |
| Permutation (mean over 100) | 0.5931 |

### Margin Statistics

| Metric | Value |
|---|---:|
| Mean gap (harmful − harmless) | 5.0061 |
| Median gap | 5.1774 |
| IQR harmful | 1.3709 |
| IQR harmless | 2.1752 |
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
| Permutation (mean over 100) | 0.4525 |

### Margin Statistics

| Metric | Value |
|---|---:|
| Mean gap (harmful − harmless) | 5.9540 |
| Median gap | 6.3601 |
| IQR harmful | 1.3720 |
| IQR harmless | 1.7491 |
| N harmful | 102 |
| N harmless | 64 |

**Audit verdict:** PASS — probe appears valid

---

## Overall Audit Conclusion

2/5 condition(s) flagged with warnings. Review per-condition sections above before proceeding to Phase 2.
