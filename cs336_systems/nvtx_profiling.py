"""NVTX-annotated functions for GPU profiling."""

import math

import torch
import torch.cuda.nvtx as nvtx
from einops import einsum
from jaxtyping import Float, Bool
from torch import Tensor

from cs336_basics.nn_utils import softmax


def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """NVTX-annotated version of scaled dot-product attention."""
    
    with nvtx.range("scaled_dot_product_attention"):
        with nvtx.range("computing attention scores"):
            d_k = K.shape[-1]
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
            
        with nvtx.range("applying mask"):
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))
                
        with nvtx.range("computing softmax"):
            attention_weights = softmax(attention_scores, dim=-1)
            
        with nvtx.range("final matmul"):
            output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
            
    return output