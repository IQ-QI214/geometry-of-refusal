# 研究方案文档索引（Refusal Vector / VLM Safety）

本目录存放本研究方向的**方案与执行框架文档**，目的是让阅读代码的人（包括 code/IDE）能直接理解"我们想做什么、为什么这么做、落到哪些代码模块"。

## 阅读顺序

1. **`实验进展汇报_VLM-Refusal_20260625.md`** — 当前实验进展与已有结论的快照。
2. **`RefusalVector汇报_20260625.md`** — 围绕"拒绝向量 / 拒绝方向"的整体汇报材料。
3. **`后续研究方法探索_SensorToGate_20260625.md`** — Sensor→Gate（传感器到门控）研究方法的早期探索。
4. **`前沿对标与方向升级_20260629.md`** — 2026 最新文献对标（DSH、Perfect-Detection、ReGap、MoRAS、OmniSteer 等），碰撞地图、护城河与三条升级路线。
5. **`研究执行框架_落地代码改造_20260629.md`** — 把研究问题 / 假设 / 方法映射到具体代码模块的执行框架（H1 可读性、H2 解离、H3 可修复性 + 选层→搬运→门控三段式流水线）。

## 与代码的对应关系

- 框架文档对应的代码迭代版本位于 `experiments/mibd_routing_v2/`（隔离目录，便于版本分离与回退）。
- 本轮已落地的纯 CPU 模块：
  - `probes/dissociation.py` — H2 传感-门控解离角度量化
  - `probes/subspace.py` — 多方向拒绝子空间读出（diff-of-means 一阶矩方法，存在已知局限）
  - `data/matched_safe_images.py` — 类别匹配安全图选择
  - `bridge/oracle_bridge.py` — H3 oracle dynamic sensor-to-gate bridge 的 CPU 侧聚合与效果摘要
  - `data/build_phase2a_dataset.py` / `run_build_phase2a_dataset.py` — v2 专属 matched benign Phase 2A 数据构建入口
  - `run_phase2a_vlm_behavior.py` / `run_phase2b_extract_probe.py` — 离线 GPU 环境的 behavior generation 与 hidden extraction/probe 入口
  - `tests/test_v2_cpu.py` — 纯 CPU 测试（17 项全通过）
- 本地真实 benign 图池准备说明见 `mibd_routing_v2_benign_pool_setup.md`。
- 离线 GPU 启动命令见 `mibd_routing_v2_offline_gpu_runbook.md`。
- GPU 相关部分（白盒 hidden-state 读写、门控前向筛选、Guard judge 接入、桥接驱动）为后续批次，需在有 GPU 的环境执行。

> 注：文档文件名中的日期为撰写日期（YYYYMMDD），并非严格的版本号。
