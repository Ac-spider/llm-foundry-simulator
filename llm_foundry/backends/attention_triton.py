"""
FlashAttention-2 Triton implementation.

Adapted from CS336 Assignment 2 (Systems) - attention_triton.py.
Provides efficient attention computation on GPUs with Triton support.
"""

import math

import torch

# Triton is optional - this module may not be importable if triton is not installed
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ─────────────────────────────────────────
# FlashAttention-2 反向传播（PyTorch 编译版）
# ─────────────────────────────────────────
@torch.compile()
def _flash_backward_compiled(Q, K, V, O, dO, L, is_causal, scale):
    """
    FlashAttention-2 backward pass using compiled PyTorch operations.

    Args:
        Q: Query tensor, shape (batch, n_heads, N_q, d)
        K: Key tensor, shape (batch, n_heads, N_k, d)
        V: Value tensor, shape (batch, n_heads, N_k, d)
        O: Forward output, shape (batch, n_heads, N_q, d)
        dO: Output gradient, shape (batch, n_heads, N_q, d)
        L: Log-sum-exp from forward, shape (batch, n_heads, N_q)
        is_causal: Whether causal mask was applied
        scale: Attention scale factor (1/sqrt(d))

    Returns:
        Tuple of (dQ, dK, dV) gradients
    """
    # D_i: scalar correction term per row
    D = (dO * O).sum(dim=-1, keepdim=True)

    # Reconstruct attention scores
    S = (Q @ K.transpose(-2, -1)) * scale
    if is_causal:
        n_q = Q.shape[-2]
        n_k = K.shape[-2]
        q_idx = torch.arange(n_q, device=Q.device).unsqueeze(-1)
        k_idx = torch.arange(n_k, device=Q.device).unsqueeze(0)
        mask = q_idx >= k_idx
        S = torch.where(mask, S, float("-1e6"))

    # Reconstruct softmax probabilities using cached L
    P = torch.exp(S - L.unsqueeze(-1))
    if is_causal:
        P = torch.where(mask, P, 0)

    # Compute gradients
    dV = P.transpose(-2, -1) @ dO
    dP = dO @ V.transpose(-2, -1)
    dS = P * (dP - D)
    dQ = (dS @ K) * scale
    dK = (dS.transpose(-2, -1) @ Q) * scale

    return dQ, dK, dV


# ─────────────────────────────────────────
# FlashAttention-2 Triton GPU Kernel (Forward)
# ─────────────────────────────────────────
if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_warps=4),
            triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 64}, num_warps=8),
            triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 128}, num_warps=4),
            triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 128}, num_warps=8),
        ],
        key=["N_QUERIES", "N_KEYS", "D"],
    )
    @triton.jit
    def flash_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        L_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        N_QUERIES,
        N_KEYS,
        scale,
        is_causal: tl.constexpr,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
    ):
        """Triton kernel for FlashAttention-2 forward pass."""
        batch_index = tl.program_id(1)
        query_tile_index = tl.program_id(0)

        # Block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        K_block_ptr = tl.make_block_ptr(
            base=K_ptr + batch_index * stride_kb,
            shape=(D, N_KEYS),
            strides=(stride_kd, stride_kk),
            offsets=(0, 0),
            block_shape=(D, K_TILE_SIZE),
            order=(0, 1),
        )

        V_block_ptr = tl.make_block_ptr(
            base=V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        O_block_ptr = tl.make_block_ptr(
            base=O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        # Online softmax state
        m_i = tl.full([Q_TILE_SIZE], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
        O_i = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

        # Load Q block
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Determine number of key tiles
        num_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
        if is_causal:
            loop_end = (query_tile_index + 1) * Q_TILE_SIZE
            num_key_tiles = tl.cdiv(tl.minimum(loop_end, N_KEYS), K_TILE_SIZE)

        # Iterate over K/V blocks
        for j in range(num_key_tiles):
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # Compute attention scores
            S_ij = tl.dot(q, k, out_dtype=tl.float32) * scale

            if is_causal:
                q_offset = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                k_offset = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
                mask = q_offset[:, None] >= k_offset[None, :]
                S_ij = tl.where(mask, S_ij, -1e6)

            # Online softmax update
            m_ij = tl.max(S_ij, -1)
            m_new = tl.maximum(m_i, m_ij)

            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(S_ij - m_new[:, None])

            l_i = l_i * alpha + tl.sum(beta, -1)
            P_ij = beta.to(v.type.element_ty)
            O_i = O_i * alpha[:, None] + tl.dot(P_ij, v, out_dtype=tl.float32)

            m_i = m_new

            # Advance pointers
            K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

        # Normalize and store output
        O_i = O_i / l_i[:, None]
        L_i = m_i + tl.log(l_i)

        tl.store(O_block_ptr, O_i.to(O_ptr.type.element_ty), boundary_check=(0, 1))

        # Store L
        l_offsets = batch_index * stride_lb + query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        l_mask = (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)) < N_QUERIES
        tl.store(L_ptr + l_offsets, L_i, mask=l_mask)


# ─────────────────────────────────────────
# FlashAttention-2 Triton Wrapper Class
# ─────────────────────────────────────────
class FlashAttention2Triton(torch.autograd.Function):
    """
    FlashAttention-2 Triton implementation wrapped as PyTorch autograd Function.

    Falls back to PyTorch SDPA if Triton is not available.
    """

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        """
        Forward pass.

        Args:
            q: Query tensor, shape (batch, n_heads, seq_len_q, d)
            k: Key tensor, shape (batch, n_heads, seq_len_k, d)
            v: Value tensor, shape (batch, n_heads, seq_len_k, d)
            is_causal: Whether to apply causal mask

        Returns:
            Attention output, shape (batch, n_heads, seq_len_q, d)
        """
        if not TRITON_AVAILABLE:
            # Fallback to PyTorch SDPA
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal
            )

        batch_size, n_heads, seq_len_q, d = q.shape
        _, _, seq_len_k, _ = k.shape

        # Flatten batch and heads dimensions
        q = q.reshape(batch_size * n_heads, seq_len_q, d)
        k = k.reshape(batch_size * n_heads, seq_len_k, d)
        v = v.reshape(batch_size * n_heads, seq_len_k, d)

        o = torch.empty_like(q)
        L = torch.empty((batch_size * n_heads, seq_len_q), device=q.device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(d)

        grid = lambda META: (triton.cdiv(seq_len_q, META["Q_TILE_SIZE"]), batch_size * n_heads)

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        flash_fwd_kernel[grid](
            q,
            k,
            v,
            o,
            L,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            L.stride(0),
            L.stride(1),
            seq_len_q,
            seq_len_k,
            scale,
            is_causal=is_causal,
            D=d,
        )

        # Restore original shapes
        o = o.view(batch_size, n_heads, seq_len_q, d)
        L = L.view(batch_size, n_heads, seq_len_q)
        q = q.view(batch_size, n_heads, seq_len_q, d)
        k = k.view(batch_size, n_heads, seq_len_k, d)
        v = v.view(batch_size, n_heads, seq_len_k, d)

        ctx.save_for_backward(q, k, v, o, L)
        ctx.is_causal = is_causal

        return o

    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass.

        Args:
            ctx: Autograd context
            dO: Output gradient, shape (batch, n_heads, N_q, d)

        Returns:
            Tuple of (dQ, dK, dV, None) gradients
        """
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = 1.0 / math.sqrt(Q.shape[-1])

        dQ, dK, dV = _flash_backward_compiled(Q, K, V, O, dO, L, is_causal, scale)

        return dQ, dK, dV, None
