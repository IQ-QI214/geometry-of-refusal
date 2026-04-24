# PCD — Handoff Document

**Last updated**: 2026-04-24 (Stage A + sweep 全部完成，进入 ablate 阶段)
**Last commit**: `6116e0e` — fix: select_direction val set too large

---

## Quick Start for New Session

**读此文件即可**，无需再读其他文档。

- Spec: `docs/superpowers/specs/2026-04-22-1500-pcd-pipeline-verification-design.md`
- Plan: `docs/superpowers/plans/2026-04-22-1500-pcd-pipeline-verification-implementation.md`

---

## 当前状态：Task 10 (sweep) ✅ 全部完成，下一步 Task 11 (ablate)

### 立即执行（新窗口第一件事）

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
bash experiments/pcd/check_status.sh       # 确认 sweep 全部 ✅
bash experiments/pcd/launch_ablate.sh      # 启动 6 条件 ablate（分两批）
```

ablate 完成后：
```bash
bash experiments/pcd/launch_evaluate.sh   # 启动 6 条件 evaluate
bash experiments/pcd/launch_aggregate.sh  # CPU 汇总矩阵（秒完成）
```

---

## Task 状态

| Task | 描述 | 状态 | 备注 |
|:-:|---|:-:|---|
| 1 | Download Gemma-3-4B + verify α | ✅ | α = PASS（weights等价） |
| 2 | Qwen VLM adapter image_mode + β | ✅ | β = PASS |
| 3 | Gemma-3 adapters + factory | ✅ | |
| 4 | Arditi templates + judge + tests | ✅ | 6/6 tests pass |
| 5 | Bootstrap stability utility | ✅ | 3/3 tests pass |
| 6 | Three experiment entry scripts | ✅ | |
| 7 | run_all.sh + docs | ✅ | |
| 8 | E2E smoke n=4 | ✅ | commit 9a9ee7d |
| 9 | H0 bootstrap (Qwen2.5-7B) | ✅ | cos=0.9984 PASS |
| **10** | **DIM layer sweep × 6 conditions** | **✅** | **见下方结果表** |
| 11 | DIM ablate + generate × 6 | ⏳ | 下一步 |
| 12 | RDO k=3（可选） | ⏳ | |
| 13 | 4-judge evaluation × 6 | ⏳ | |
| 14 | Aggregate matrix + findings | ⏳ | |

---

## Sweep 结果（Task 10）

| 条件 | Model | Best Layer | pos | filter | 说明 |
|---|---|:-:|:-:|:-:|---|
| V-text | Qwen2.5-VL | **17** | -5 | ✅ | 中层，匹配 P0 baseline |
| V-blank | Qwen2.5-VL | **15** | -5 | ✅ | 中层 |
| V-noise | Qwen2.5-VL | **16** | -5 | ✅ | 中层 |
| V-text | Gemma-3-4B | **29** | -5 | ✅ | 深层（reselect 修正） |
| V-blank | Gemma-3-4B | **1** | -5 | ✅ | 早层：最强 ablation 方向¹ |
| V-noise | Gemma-3-4B | **1** | -5 | ✅ | 同上¹ |

¹ Gemma V-blank/V-noise 选 L1 原因：排序逻辑为 `sorting_score=-refusal_score`，
L1 的 refusal=-18.3 是所有非剪枝层最低值，即 ablation 后模型拒绝最彻底消除。
与 V-text(L29) 形成层深对比，是 H3 假说的核心比较数据点。**结果有效，无需重跑。**

---

## Key Decisions (验证点)

| 验证点 | 结论 |
|---|---|
| **α** Gemma backbone weight equivalence | PASS — L ≡ V-text，只跑一个 text 条件 |
| **β** Qwen V-text (pixel_values=None) | PASS |
| **γ** Gemma adapter smoke | PASS（34层，VLM+text 都正常） |
| **H0** bootstrap cos (Qwen2.5-7B, L17, pos=-5) | **cos=0.9984 PASS** — 方向极稳定 |

---

## 环境（新 agent 必读）

| 类型 | 信息 |
|---|---|
| CPU session | Claude 在这里写代码，不能直接跑 GPU |
| GPU session | 4×H100，qi 手动执行，offline 实例 |
| 工作目录 | `/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/` |
| 运行目录 | `refusal_direction/`（`PYTHONPATH=.`） |
| Conda env | `qwen3-vl`（Qwen2.5-VL + Gemma-3，transformers 4.57.3） |
| Gemma 路径 | `/inspire/hdd/global_user/wenming-253108090054/models/gemma-3-4b-it` |
| Qwen VLM 路径 | `/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct` |

**关键注意**：
- 进程必须用 `nohup` 启动，否则 SSH 断开后会被 SIGHUP 杀死
- `launch_*.sh` 脚本已内置 nohup，直接用即可
- 确认无旧进程再启动新实验：`ps aux | grep exp_pcd | grep -v grep`

---

## 已知 Bug 及修复（本会话）

| Bug | 修复 | Commit |
|---|---|---|
| `Gemma3Config` 无 `num_hidden_layers` | `len(block_modules)` 替代 | `dbccef4` |
| Gemma images 格式错误（`[img]*N` → 应为 `[[img]]*N`） | 嵌套列表 | `4b9d9f6` |
| `local_files_only=True` 在新版 HF 报 ValidationError | 移除，依赖 env offline | `4b9d9f6` |
| `select_direction` val 集 3132 条→每条件 14h | `--select_n_val 128`，降到 35min | `6116e0e` |
| `select_direction` 无中间保存→被杀归零 | 每 source_pos 存 checkpoint.pt | `13cd019` |
| Gemma sweep fallback 选最差方向 | sort 改为降序 | `291f21f` |
| Named-pipe GPU pool 竞态（GPU30）| 改为静态批次分配 | `08f6fe2` |

---

## 启动脚本（一键操作）

```
experiments/pcd/
├── launch_sweep.sh      # 4 GPU nohup sweep（当前已完成，无需重跑）
├── launch_ablate.sh     # 6 条件 ablate（下一步）
├── launch_evaluate.sh   # 6 条件 evaluate
├── launch_aggregate.sh  # CPU 汇总矩阵
└── check_status.sh      # 一键状态检查 + 下一步提示
```

**进度检查命令**：
```bash
bash experiments/pcd/check_status.sh
# 或实时监控日志
tail -f results/pcd/qwen_family/V-text/ablate.log
```

---

## 背景进程状态

- 当前：无运行进程，所有 GPU 空闲
- 无未提交文件

---

*Created: 2026-04-22 | Updated: 2026-04-24 (sweep ✅, ablate 待启动)*
