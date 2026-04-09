"""
Category A 数据集加载工具。
支持 SaladBench (Config-1) 和 HarmBench (Config-2)。
"""

import json
from pathlib import Path
from typing import List, Dict

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DATA_DIR = _PROJ_ROOT / "data"


def load_saladbench_test() -> List[Dict[str, str]]:
    """
    加载 SaladBench harmful_test.json 全量 (572 条)。
    返回 [{"instruction": str, "source": str}, ...]
    """
    path = _DATA_DIR / "saladbench_splits" / "harmful_test.json"
    with open(path) as f:
        data = json.load(f)
    prompts = []
    for item in data:
        prompts.append({
            "instruction": item["instruction"],
            "source": item.get("source", "unknown"),
        })
    print(f"[data_utils] Loaded {len(prompts)} prompts from SaladBench harmful_test")
    return prompts


def load_harmbench_test() -> List[Dict[str, str]]:
    """
    加载 HarmBench 标准 test set (Config-2, 需下载)。
    返回 [{"instruction": str, "category": str}, ...]
    """
    path = _DATA_DIR / "harmbench" / "harmbench_test.json"
    if not path.exists():
        raise FileNotFoundError(
            f"HarmBench data not found at {path}. "
            "Download from https://github.com/centerforaisafety/HarmBench"
        )
    with open(path) as f:
        data = json.load(f)
    prompts = []
    for item in data:
        prompts.append({
            "instruction": item.get("instruction", item.get("goal", "")),
            "category": item.get("category", "unknown"),
        })
    print(f"[data_utils] Loaded {len(prompts)} prompts from HarmBench")
    return prompts


def load_dataset(name: str) -> List[Dict[str, str]]:
    """统一入口: name='saladbench' 或 'harmbench'。"""
    if name == "saladbench":
        return load_saladbench_test()
    elif name == "harmbench":
        return load_harmbench_test()
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'saladbench' or 'harmbench'.")
