"""
Unit test for T1: verify orthogonalize_qwen_weights and act_add_qwen_weights
execute without AttributeError on a mock Qwen2.5-style model structure.
Run: python experiments/repro_arditi_wollschlager/test_t1_qwen_fix.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../refusal_direction"))

import torch
import torch.nn as nn
from pipeline.model_utils.qwen_model import orthogonalize_qwen_weights, act_add_qwen_weights

D = 64

class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(D, D, bias=False)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(D, D, bias=False)

class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(1000, D)
        self.model.layers = nn.ModuleList([FakeLayer() for _ in range(4)])

direction = torch.randn(D)
model = FakeModel()

try:
    orthogonalize_qwen_weights(model, direction)
    print("[PASS] orthogonalize_qwen_weights: no AttributeError")
except AttributeError as e:
    print(f"[FAIL] orthogonalize_qwen_weights: {e}")
    raise

try:
    act_add_qwen_weights(model, direction, coeff=1.0, layer=1)
    assert hasattr(model.model.layers[0].mlp.down_proj, 'bias'), "bias not set"
    print("[PASS] act_add_qwen_weights: bias set on layers[0].mlp.down_proj")
except (AttributeError, AssertionError) as e:
    print(f"[FAIL] act_add_qwen_weights: {e}")
    raise

print("\nAll T1 tests passed.")
