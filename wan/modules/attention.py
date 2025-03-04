import xformers.ops as xops
from xformers.ops import memory_efficient_attention, fmha
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
    dtype=torch.float16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
   
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes, f"dtype must be float16 or bfloat16, got {dtype}"
    assert q.device.type == "cuda" and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # 预处理查询
    if q_lens is None:
        q = half(q.flatten(0, 1))  # [B*Lq, Nq, C1]
        q_lens = torch.full((b,), lq, dtype=torch.int32, device=q.device)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)], dim=0))

    # 预处理键和值
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.full((b,), lk, dtype=torch.int32, device=k.device)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)], dim=0))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)], dim=0))

    # 确保数据类型一致
    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    if q_scale is not None:
        q = q * q_scale

    # 调整键和值的头数以匹配查询
    n_q_heads = q.size(1)
    n_k_heads = k.size(1)
    if n_k_heads != n_q_heads:
        assert n_q_heads % n_k_heads == 0, "Nq must be divisible by Nk"
        repeat_factor = n_q_heads // n_k_heads
        k = k.repeat(1, repeat_factor, 1)
        v = v.repeat(1, repeat_factor, 1)

    # if window_size != (-1, -1):
    #     raise NotImplementedError("Sliding window attention not supported with xFormers")
    window_size = (-1, -1)

    # 生成块对角掩码
    q_lens_list = q_lens.cpu().tolist()
    k_lens_list = k_lens.cpu().tolist()

    if causal:
        
        attn_bias = fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(q_seqlen=q_lens_list)
    else:
        attn_bias = fmha.attn_bias.BlockDiagonalMask.from_seqlens(q_seqlen=q_lens_list, kv_seqlen=k_lens_list)

    # 添加虚拟批次维度以适应xFormers接口
    q = q.unsqueeze(0)  # [1, sum_q, nh, hd]
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)

    # 调用xFormers的高效注意力实现
    x = xops.memory_efficient_attention(
        q, k, v,
        attn_bias=attn_bias,
        p=dropout_p,
        scale=softmax_scale,
        # deterministic=deterministic  # xFormers可能不支持此参数
    )

    # 移除虚拟批次维度并恢复原始形状
    x = x.squeeze(0).unflatten(0, (b, lq))  # [B, Lq, Nq, C2]

    return x.to(out_dtype)
