"""Step 0: Direction stability check via two disjoint prompt subsets."""
from __future__ import annotations
import json
import random
import torch
from pathlib import Path
from typing import List, Dict


def split_prompts(prompts: List[str], seed: int) -> tuple[list[int], list[int]]:
    """Deterministically partition indices into two disjoint halves of equal size."""
    n = len(prompts)
    half = n // 2
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    return idx[:half], idx[half:2 * half]


def extract_mean_diff_direction(
    model, harmful_prompts: List[str], harmless_prompts: List[str],
    layer: int, pos: int, tokenize_fn, batch_size: int = 8,
) -> torch.Tensor:
    """Compute mean(harmful activation) − mean(harmless activation) at (layer, pos), unit-norm."""
    block = model.model_block_modules[layer]

    def _collect(prompts):
        acts = []

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            acts.append(h[:, pos, :].detach().float().cpu())

        handle = block.register_forward_hook(hook)
        try:
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                tok = tokenize_fn(instructions=batch)
                with torch.no_grad():
                    ids = tok.input_ids.to(model.model.device)
                    am = tok.attention_mask.to(model.model.device)
                    kwargs = {"input_ids": ids, "attention_mask": am}
                    if hasattr(tok, "pixel_values") and tok.pixel_values is not None:
                        kwargs["pixel_values"] = tok.pixel_values.to(
                            device=model.model.device, dtype=model.model.dtype
                        )
                    model.model(**kwargs)
        finally:
            handle.remove()
        return torch.cat(acts, dim=0).mean(dim=0)

    mh = _collect(harmful_prompts)
    ml = _collect(harmless_prompts)
    d = mh - ml
    return d / (d.norm() + 1e-8)


def bootstrap_stability(
    model, harmful_prompts: List[str], harmless_prompts: List[str],
    layer: int, pos: int, tokenize_fn, seed: int = 42, output_path: Path = None,
) -> Dict:
    """Split prompts 50/50, extract two directions, return cosine + metadata."""
    ih_a, ih_b = split_prompts(harmful_prompts, seed)
    il_a, il_b = split_prompts(harmless_prompts, seed + 1)
    d_a = extract_mean_diff_direction(
        model,
        [harmful_prompts[i] for i in ih_a],
        [harmless_prompts[i] for i in il_a],
        layer=layer, pos=pos, tokenize_fn=tokenize_fn,
    )
    d_b = extract_mean_diff_direction(
        model,
        [harmful_prompts[i] for i in ih_b],
        [harmless_prompts[i] for i in il_b],
        layer=layer, pos=pos, tokenize_fn=tokenize_fn,
    )
    cos = torch.dot(d_a, d_b).item()
    result = {
        "cos": cos,
        "layer": layer, "pos": pos, "seed": seed,
        "n_per_subset": len(ih_a),
        "verdict": "PASS" if cos >= 0.9 else ("MARGINAL" if cos >= 0.7 else "FAIL"),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    import argparse
    from pipeline.model_utils.model_factory import construct_model_base
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--pos", type=int, required=True)
    p.add_argument("--harmful_prompts", required=True)
    p.add_argument("--harmless_prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    model = construct_model_base(a.model_path, a.model_name)
    harmful = json.loads(Path(a.harmful_prompts).read_text())
    harmless = json.loads(Path(a.harmless_prompts).read_text())
    harmful_ins = [x["instruction"] for x in harmful]
    harmless_ins = [x["instruction"] for x in harmless]

    r = bootstrap_stability(
        model, harmful_ins, harmless_ins,
        layer=a.layer, pos=a.pos,
        tokenize_fn=model.tokenize_instructions_fn,
        seed=a.seed, output_path=Path(a.output),
    )
    print(json.dumps(r, indent=2))
