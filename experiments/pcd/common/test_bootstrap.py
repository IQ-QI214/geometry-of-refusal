"""Unit tests for bootstrap helpers (CPU only — no model load)."""
from experiments.pcd.common.bootstrap import split_prompts


def test_split_disjoint():
    prompts = [f"p{i}" for i in range(100)]
    a, b = split_prompts(prompts, seed=42)
    assert len(a) == 50 and len(b) == 50
    assert set(a).isdisjoint(set(b))
    assert set(a) | set(b) <= set(range(100))


def test_split_deterministic():
    prompts = [f"p{i}" for i in range(100)]
    a1, b1 = split_prompts(prompts, seed=42)
    a2, b2 = split_prompts(prompts, seed=42)
    assert a1 == a2 and b1 == b2


def test_split_seed_variance():
    prompts = [f"p{i}" for i in range(100)]
    a1, _ = split_prompts(prompts, seed=42)
    a2, _ = split_prompts(prompts, seed=43)
    assert a1 != a2
