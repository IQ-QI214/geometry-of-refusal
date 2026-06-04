# Phase 1 Go/No-Go Report: /inspire/hdd/global_user/wenming-253108090054/models/gemma-3-4b-it

Signal: `harmfulness`
Decision: **STOP_OR_PIVOT**

| Condition | Layer | Token Pos | AUC |
|---|:---:|:---:|:---:|
| V-text | 29 | -1 | 1.000 |
| V-blank | 32 | -1 | 0.996 |
| V-noise | 32 | -1 | 0.998 |
| V-real | 32 | -3 | 0.998 |
| FigStep | 17 | -8 | 0.999 |

- Locus/direction shift: PASS
- Blank/noise equivalence: PASS
- Static transfer drop: FAIL
