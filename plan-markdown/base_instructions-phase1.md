# 基础指令 (Base Instructions)

> 在每次新对话开始时提供给 Claude，确保行为一致。

---

## 称呼

请在所有对话中称呼我 **qi**。

## 工作目录

- **允许修改的目录**：`/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/`
- 该目录下的任何文件和子目录可以自由创建、编辑、删除
- **禁止**未经询问修改该目录以外的任何文件
- 如果需要修改其他路径的文件，**必须先询问我**

## 集群环境

- **CPU 节点**：可联网，用于开发、代码编写、包安装
- **GPU 节点**：**不联网**，所有代码必须支持离线运行
  - 代码中所有 `from_pretrained()` 调用必须加 `local_files_only=True`
  - 运行脚本中必须设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`
- **GPU 硬件**：4 张 H100 80GB
- **conda 环境**：`rdo`（Python 3.10, PyTorch 2.5.1+cu124, transformers 4.47.0）

## 模型使用

- **优先使用本地已有模型**，不要随意更换
- 本地模型缓存目录：`/inspire/hdd/global_user/wenming-253108090054/models/hub/`
- 当前已缓存模型：
  - `llava-hf/llava-1.5-7b-hf` (LLaVA-1.5-7B)
- `.env` 配置（位于项目根目录）：
  ```
  HUGGINGFACE_CACHE_DIR="/inspire/hdd/global_user/wenming-253108090054/models/hub"
  SAVE_DIR="/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/results"
  ```
- 如果需要使用新模型，**必须先告知我**，由我手动下载到本地

## 代码组织

- 实验代码放在 `experiments/` 目录下，按实验名分子目录
- 分析文档放在 `analysis/` 目录下
- 不要把所有文件堆在项目根目录

## 输出与总结

- 每完成一轮实验后，写一份**有深度的分析 md 文档**，包含：
  - 实验数据表格
  - 有趣发现 (Findings)
  - 问题分析 (Problems)
  - 洞察 (Insights)
  - 新颖性分析 (Novelty)
  - 下一步方向
- 总结要有深度但不要过长，注意 token 消耗
- 文档保存在 `analysis/` 目录下

## Token 管理

- 如果一个对话中有多个大型任务，完成 1-2 个后**主动提醒我**
- 我需要阶段性复盘和收尾，然后开新对话继续
- 不要等 token 耗尽才告知

## 当前项目

- **项目路径**：`/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/`
- **研究方向**：Gap C — VLM Jailbreak Attack, Sequence-Level Refusal Suppression via Visual Perturbation
- **研究框架文档**：`plan-markdown/gapc-research-framework`
- **实验操作指南**：`plan-markdown/gapc-pilot-exp`
- **已完成实验结果**：`results/exp_{a,b,c}_results.json`
- **分析报告**：`analysis/pilot_experiments_report.md`
