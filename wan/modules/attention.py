import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

# Try importing flash attention libraries.
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

__all__ = [
    'flash_attention',
    'attention',
]

##########################################
# Linformer-style Attention Module
##########################################

class LinformerAttention(nn.Module):
    def __init__(self, max_seq_len, projection_dim, head_dim, dropout_p=0.0):
        """
        Args:
            max_seq_len (int): Maximum sequence length expected.
            projection_dim (int): The reduced dimension for keys/values.
            head_dim (int): Dimension of each head.
            dropout_p (float): Dropout probability.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.projection_dim = projection_dim
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        # Learnable projection matrices for keys and values.
        self.proj_k = nn.Parameter(torch.randn(max_seq_len, projection_dim))
        self.proj_v = nn.Parameter(torch.randn(max_seq_len, projection_dim))
        nn.init.xavier_normal_(self.proj_k)
        nn.init.xavier_normal_(self.proj_v)

    def forward(self, q, k, v, q_scale=None, softmax_scale=None, causal=False):
        """
        Args:
            q: Query tensor of shape [B, Lq, H, C].
            k: Key tensor of shape [B, Lk, H, C].
            v: Value tensor of shape [B, Lk, H, C].
            q_scale (float or None): Optional scaling applied to queries.
            softmax_scale (float or None): Scaling factor applied after dot product.
            causal (bool): If True, applies a simple causal mask.
        Returns:
            Output tensor of shape [B, Lq, H, C].
        """
        B, Lq, H, C = q.shape
        _, Lk, _, _ = k.shape

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(C)
        if q_scale is not None:
            q = q * q_scale

        # Use only the first Lk rows of the projection matrices.
        proj_k = self.proj_k[:Lk]  # shape [Lk, r]
        proj_v = self.proj_v[:Lk]  # shape [Lk, r]

        # Project keys and values along the sequence dimension.
        # k_proj: [B, r, H, C]
        k_proj = torch.einsum("blhc,lr->brhc", k, proj_k)
        v_proj = torch.einsum("blhc,lr->brhc", v, proj_v)

        # Compute attention scores: [B, H, Lq, r]
        attn_scores = torch.einsum("blhc,brhc->bhlr", q, k_proj) * softmax_scale

        if causal:
            # Simple causal mask: note r is not exactly time-indexed,
            # so this is an approximation.
            mask = torch.triu(torch.full((Lq, attn_scores.size(-1)), float("-inf")), diagonal=1).to(attn_scores.device)
            attn_scores = attn_scores + mask.unsqueeze(0).unsqueeze(0)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Weighted sum over projected values: [B, Lq, H, C]
        output = torch.einsum("bhlr,brhc->blhc", attn_probs, v_proj)
        return output

##########################################
# Flash Attention Function
##########################################

def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    Flash attention implementation.
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. Scaling for QK^T.
    causal:         bool. Whether to apply causal mask.
    window_size:    (left, right). If not (-1, -1), apply local attention.
    deterministic:  bool.
    dtype:          torch.dtype to enforce (float16/bfloat16).
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # Preprocess queries
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # Preprocess keys and values
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    try:
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if q_scale is not None:
            q = q * q_scale

        if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
            warnings.warn('Flash attention 3 is not available, using flash attention 2 instead.')
        
        if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
            x = flash_attn_interface.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device),
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic
            )[0].unflatten(0, (b, lq))
        else:
            assert FLASH_ATTN_2_AVAILABLE, "Flash attention not available."
            x = flash_attn.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic
            ).unflatten(0, (b, lq))
    except RuntimeError as e:
        if "FlashAttention only supports Ampere GPUs or newer" in str(e):
            # Fallback: use scaled_dot_product_attention for older GPUs.
            from torch import nn
            q = q.view(b, lq, -1).transpose(1, 2).to(dtype)
            k = k.view(b, lk, -1).transpose(1, 2).to(dtype)
            v = v.view(b, lk, -1).transpose(1, 2).to(dtype)
            if softmax_scale is None:
                softmax_scale = 1.0 / (q.size(-1) ** 0.5)
            if q_scale is not None:
                q = q * q_scale
            if causal:
                attn_mask = torch.triu(torch.full((q.size(2), k.size(2)), -float("inf")), diagonal=1).to(q.device)
            else:
                attn_mask = None
            x = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)
            x = x.transpose(1, 2).contiguous()
        else:
            raise
    return x.type(out_dtype)

##########################################
# Unified Attention Function with Fallback
##########################################

def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """
    Unified attention function.
    If flash attention (FA) libraries are available, it uses FA.
    Otherwise, it falls back to a Linformer-style attention module.
    """
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        try:
            return flash_attention(
                q=q,
                k=k,
                v=v,
                q_lens=q_lens,
                k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                dtype=dtype,
                version=fa_version,
            )
        except RuntimeError as e:
            warnings.warn(f"Flash attention failed with error: {e}. Falling back to Linformer attention.")
    # Fallback: Use Linformer attention.
    # Ensure input q, k, v are of shape [B, L, H, C].
    B, Lq, H, C = q.shape
    _, Lk, _, _ = k.shape
    # Instantiate LinformerAttention with a fixed projection dimension (adjust as needed).
    # Here we set projection_dim to a default value (e.g., 64) or you can choose Lk // reduction_factor.
    projection_dim = 64  
    linformer_attn = LinformerAttention(max_seq_len=Lk, projection_dim=projection_dim, head_dim=C, dropout_p=dropout_p).to(q.device)
    return linformer_attn(q, k, v, q_scale=q_scale, softmax_scale=softmax_scale, causal=causal)
