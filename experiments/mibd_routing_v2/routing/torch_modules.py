"""Trainable torch port of the CaRoB routing core (GPU phase).

This is the thin torch wrapper the blueprint defers to the GPU stage
(CaRoB代码实现交接蓝图 §七 "GPU torch (后续): 薄封装成 nn.Module"). Every op
mirrors the numpy reference in :mod:`experiments.mibd_routing_v2.routing` so the
two stay numerically aligned (a parity test asserts this). The difference is
that here parameters are ``nn.Parameter`` with autograd, so the router, gate and
bridge can actually be *trained* against the round-1 objective.

Design notes
------------
* ``GatingLosses`` are stateless functions matching ``routing.gating`` but
  operating on ``torch.Tensor`` and differentiable end-to-end.
* ``CarrierRouter`` is the trainable counterpart of ``routing.router.route``:
  a 2-layer MLP gating with a learned carrier embedding and the DirMoE-style
  decoupled layer / trust heads.
* ``ThresholdGate`` / ``LowRankBridge`` mirror ``routing.gate`` and the LoRA
  delta in ``routing.bridge_losses``.

Import is torch-gated: the module raises a clear error if torch is missing so
the CPU-only environment can still import the package without crashing.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on CPU-only boxes
    _TORCH_AVAILABLE = False

    class _Missing:
        def __getattr__(self, name):
            raise ImportError(
                "torch is required for experiments.mibd_routing_v2.routing.torch_modules; "
                "install torch in the GPU environment (qwen3-vl / rdo conda env)."
            )

    torch = _Missing()  # type: ignore
    nn = _Missing()  # type: ignore
    F = _Missing()  # type: ignore


def torch_available() -> bool:
    return _TORCH_AVAILABLE


# --------------------------------------------------------------------------- #
# Gating losses (differentiable; mirror routing.gating)
# --------------------------------------------------------------------------- #
if _TORCH_AVAILABLE:

    def switch_load_balancing_loss(gate_probs: "torch.Tensor", expert_mask: "torch.Tensor") -> "torch.Tensor":
        """L * sum_i f_i * P_i (Switch Transformer, arXiv:2101.03961)."""
        if gate_probs.shape != expert_mask.shape:
            raise ValueError(f"shape mismatch: {gate_probs.shape} vs {expert_mask.shape}")
        n, L = gate_probs.shape
        f = expert_mask.sum(dim=0) / max(n, 1)
        P = gate_probs.mean(dim=0)
        return L * torch.dot(f, P)

    def router_z_loss(logits: "torch.Tensor") -> "torch.Tensor":
        """mean((logsumexp(logits) - log K)^2) (ST-MoE, arXiv:2202.08906).

        Subtracts the uniform-logits baseline log(K) so all-zero logits give 0,
        matching the numpy reference.
        """
        K = logits.shape[-1]
        lse = torch.logsumexp(logits, dim=-1)
        return torch.mean((lse - torch.log(torch.tensor(float(K)))) ** 2)

    def update_loss_free_bias(
        bias: "torch.Tensor",
        expert_load: "torch.Tensor",
        target_load: "torch.Tensor",
        lr: float = 1e-3,
    ) -> "torch.Tensor":
        """Loss-Free dynamic bias balancing step (arXiv:2408.15664).

        Returns the *updated bias tensor*; this is a no-grad bookkeeping update
        applied between optimizer steps, not part of the loss graph.
        """
        with torch.no_grad():
            return bias + lr * (target_load - expert_load)

    def gumbel_softmax_sample(
        logits: "torch.Tensor",
        temperature: float,
        hard: bool = False,
        generator: "torch.Generator | None" = None,
    ) -> "torch.Tensor":
        """Gumbel-Softmax sampling (arXiv:1611.01144), straight-through if hard."""
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        u = torch.rand(logits.shape, generator=generator, device=logits.device, dtype=logits.dtype)
        u = u.clamp(1e-12, 1.0 - 1e-12)
        gumbel = -torch.log(-torch.log(u))
        y = F.softmax((logits + gumbel) / temperature, dim=-1)
        if not hard:
            return y
        idx = y.argmax(dim=-1, keepdim=True)
        hard_y = torch.zeros_like(y).scatter_(-1, idx, 1.0)
        # straight-through: forward is one-hot, backward flows through y
        return (hard_y - y).detach() + y

    def anneal_temperature(step: int, t_start: float = 2.0, t_end: float = 0.5, total_steps: int = 1000) -> float:
        if total_steps <= 0:
            return float(t_end)
        progress = min(max(step / total_steps, 0.0), 1.0)
        return float(t_start + (t_end - t_start) * progress)


# --------------------------------------------------------------------------- #
# Trainable modules
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TorchRouterConfig:
    hidden_dim: int
    num_layers: int
    num_carriers: int
    carrier_embed_dim: int = 8
    mlp_hidden: int = 32
    mode: str = "soft"  # "soft" | "top1"


if _TORCH_AVAILABLE:

    class CarrierRouter(nn.Module):
        """Carrier-conditioned routing head (trainable counterpart of routing.router).

        forward(features, carrier_ids, temperature) -> dict with keys
        ``layer_logits``, ``layer_probs``, ``selected``, ``trust``.
        """

        def __init__(self, config: TorchRouterConfig):
            super().__init__()
            self.config = config
            self.carrier_embed = nn.Embedding(config.num_carriers, config.carrier_embed_dim)
            in_dim = config.hidden_dim + config.carrier_embed_dim
            self.fc1 = nn.Linear(in_dim, config.mlp_hidden)
            self.layer_head = nn.Linear(config.mlp_hidden, config.num_layers)
            self.trust_head = nn.Linear(config.mlp_hidden, config.num_layers)

        def forward(
            self,
            features: "torch.Tensor",
            carrier_ids: "torch.Tensor",
            temperature: float = 1.0,
            gumbel: bool = False,
            generator: "torch.Generator | None" = None,
        ) -> dict:
            if features.dim() != 2 or features.shape[1] != self.config.hidden_dim:
                raise ValueError(
                    f"features must be (n, {self.config.hidden_dim}), got {tuple(features.shape)}"
                )
            carrier_vec = self.carrier_embed(carrier_ids)
            x = torch.cat([features, carrier_vec], dim=-1)
            h = torch.tanh(self.fc1(x))
            layer_logits = self.layer_head(h)
            trust_logits = self.trust_head(h)
            layer_probs = F.softmax(layer_logits / temperature, dim=-1)
            trust = F.softmax(trust_logits, dim=-1)

            if gumbel:
                selected = gumbel_softmax_sample(
                    layer_logits, temperature=temperature, hard=(self.config.mode == "top1"),
                    generator=generator,
                )
            elif self.config.mode == "soft":
                selected = layer_probs
            elif self.config.mode == "top1":
                idx = layer_probs.argmax(dim=-1, keepdim=True)
                selected = torch.zeros_like(layer_probs).scatter_(-1, idx, 1.0)
            else:
                raise ValueError(f"unknown router mode: {self.config.mode}")

            return {
                "layer_logits": layer_logits,
                "layer_probs": layer_probs,
                "selected": selected,
                "trust": trust,
            }

    def aggregate_risk_score(layer_probs: "torch.Tensor", per_layer_probe_scores: "torch.Tensor") -> "torch.Tensor":
        """s(x) = sum_l p_l * probe_l(h_l(x))."""
        if layer_probs.shape != per_layer_probe_scores.shape:
            raise ValueError(
                f"shape mismatch: {layer_probs.shape} vs {per_layer_probe_scores.shape}"
            )
        return (layer_probs * per_layer_probe_scores).sum(dim=-1)

    class ThresholdGate(nn.Module):
        """Soft / hard threshold gate (mirror routing.gate)."""

        def __init__(self, tau: float = 0.0, alpha: float = 1.0, learn_threshold: bool = False):
            super().__init__()
            if learn_threshold:
                self.tau = nn.Parameter(torch.tensor(float(tau)))
                self.alpha = nn.Parameter(torch.tensor(float(alpha)))
            else:
                self.register_buffer("tau", torch.tensor(float(tau)))
                self.register_buffer("alpha", torch.tensor(float(alpha)))

        def soft(self, risk_score: "torch.Tensor") -> "torch.Tensor":
            return torch.sigmoid(self.alpha * (risk_score - self.tau))

        def hard(self, risk_score: "torch.Tensor") -> "torch.Tensor":
            return (risk_score > self.tau).to(risk_score.dtype)

        def forward(self, risk_score: "torch.Tensor") -> "torch.Tensor":
            return self.soft(risk_score)

    def nullspace_projection(delta: "torch.Tensor", benign_basis: "torch.Tensor") -> "torch.Tensor":
        """delta - delta @ B^T @ B, with orthonormal benign basis rows B."""
        if delta.shape[-1] != benign_basis.shape[-1]:
            raise ValueError("delta and benign_basis last dim must match")
        return delta - delta @ benign_basis.t() @ benign_basis

    def apply_gated_delta(hidden: "torch.Tensor", delta: "torch.Tensor", gate: "torch.Tensor") -> "torch.Tensor":
        """h' = h + gate[:, None] * delta."""
        return hidden + gate.unsqueeze(-1) * delta

    class LowRankBridge(nn.Module):
        """Gated low-rank transport bridge: delta = scaling * (h @ A @ B).

        Only applied on the gate side (later layers). ``scaling`` follows the
        LoRA convention ``alpha / r``.
        """

        def __init__(self, hidden_dim: int, rank: int = 4, alpha: float = 8.0):
            super().__init__()
            self.A = nn.Parameter(torch.zeros(hidden_dim, rank))
            self.B = nn.Parameter(torch.zeros(rank, hidden_dim))
            nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
            # B stays zero at init so the bridge starts as identity (no perturbation)
            self.scaling = alpha / rank

        def delta(self, hidden: "torch.Tensor") -> "torch.Tensor":
            return self.scaling * (hidden @ self.A @ self.B)

        def forward(self, hidden: "torch.Tensor", gate: "torch.Tensor") -> "torch.Tensor":
            return apply_gated_delta(hidden, self.delta(hidden), gate)

    # ------------------------------------------------------------------ #
    # Bridge losses (mirror routing.bridge_losses)
    # ------------------------------------------------------------------ #
    def distillation_loss_c2(student_gate_hidden: "torch.Tensor", teacher_gate_hidden: "torch.Tensor") -> "torch.Tensor":
        return F.mse_loss(student_gate_hidden, teacher_gate_hidden)

    def self_supervised_corr_loss_c3(risk_score: "torch.Tensor", safety_margin: "torch.Tensor") -> "torch.Tensor":
        s = risk_score.flatten()
        m = safety_margin.flatten()
        s_c = s - s.mean()
        m_c = m - m.mean()
        denom = s_c.norm() * m_c.norm()
        if denom < 1e-12:
            return torch.zeros((), device=s.device, dtype=s.dtype)
        return -(s_c @ m_c) / denom

    def utility_anchor_penalty(delta: "torch.Tensor", gate: "torch.Tensor") -> "torch.Tensor":
        weight = (1.0 - gate.clamp(0.0, 1.0))
        sq_norm = (delta ** 2).sum(dim=-1)
        return torch.mean(weight * sq_norm)
