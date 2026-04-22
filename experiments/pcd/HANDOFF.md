# PCD — Handoff for New Session

**Last updated**: 2026-04-22 (post-planning, pre-execution)
**Status**: Spec + Plan complete, execution not started.

---

## Quick Start for New Session

**Read these files in order before doing anything:**

1. **Spec**: `docs/superpowers/specs/2026-04-22-1500-pcd-pipeline-verification-design.md`
   - Understand the research question (H1/H2/H3 hypotheses) and condition matrix.
2. **Plan**: `docs/superpowers/plans/2026-04-22-1500-pcd-pipeline-verification-implementation.md`
   - 14 Tasks, TDD-style. Every GPU step has a "Hand to qi" command block.
3. **This file** (`experiments/pcd/HANDOFF.md`)
   - Current progress + any outcomes of verification points.

**Then:**
- Use skill `superpowers:subagent-driven-development` to execute the plan task-by-task.
- Start from Task 1 (Download Gemma-3-4B + verify α).

---

## Current State

- **No tasks started yet.**
- `experiments/pcd/` only contains this file; PROGRESS.md and other scaffolding are created in Task 7.
- No GPU runs performed, no models downloaded, no code changes beyond spec/plan files.

---

## Critical Constraints (read every session)

1. **CPU session (Claude) writes code; GPU session (qi, 4×H100, offline) runs experiments manually.**
   - Every GPU-requiring step has a "Hand to qi" block with complete commands in the plan.
2. **Workspace scope**: only `/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/` is writable. Shared `models/` is read-only unless qi explicitly permits.
3. **Verification points are mandatory branching**:
   - α (Task 1): Gemma-3 backbone equivalence
   - β (Task 2): Qwen V-text `pixel_values=None` support
   - H0 (Task 9): Direction stability bootstrap
   - Do NOT skip these; branching rules in the plan's Task sections.
4. **Conda envs**:
   - `rdo` — LLaVA, Gemma, general pipeline
   - `qwen3-vl` — Qwen2.5-VL, all guard models (Qwen3Guard, LlamaGuard-3, StrongReject)
5. **Address user as qi. Conversation in Chinese (Simplified). Code stays English.**

---

## Key Decisions (to be filled as tasks complete)

- **α verdict**: _pending Task 1_
- **β verdict**: _pending Task 2_
- **H0 verdict**: _pending Task 9_
- **Gemma-3-4B download path**: _to be asked from qi in Task 1 Step 1.2_
- **Final Qwen V-blank re-sweep best_layer**: _pending Task 10; determines whether to reuse P0 responses or regenerate_

---

## Environment State

- No background processes running.
- No intermediate files awaiting cleanup.
- `results/pcd/` does not yet exist.
- Gemma-3-4B-it not yet downloaded.

---

## Session Rotation Checkpoints (per plan's Appendix)

Suggest new session at these natural breakpoints:
1. After Task 8 (Stage A complete, E2E smoke passed) — before Stage B GPU runs
2. After Task 10 (all layer sweeps done) — decision point on best_layer for each condition
3. After Task 13 (all judges done) — before aggregate/findings

At each rotation point: update this HANDOFF.md and commit before telling qi to rotate.

---

## Git Reference

Most recent commit before handoff: _(filled by commit script below)_
