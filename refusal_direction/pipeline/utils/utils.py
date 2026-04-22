import torch
import einops
from torch import Tensor
try:
    from jaxtyping import Int, Float
except ImportError:
    class _JaxStub:
        def __class_getitem__(cls, item): return cls
    Int = Float = _JaxStub

def get_orthogonalized_matrix(matrix: Float[Tensor, '... d_model'], vec: Float[Tensor, 'd_model']) -> Float[Tensor, '... d_model']:
    vec = vec / torch.norm(vec)
    vec = vec.to(matrix)

    proj = einops.einsum(matrix, vec.unsqueeze(-1), '... d_model, d_model single -> ... single') * vec
    return matrix - proj