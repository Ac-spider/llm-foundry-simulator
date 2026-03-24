"""
Transformer language model with ModelConfig.

Adapted from CS336 Assignment 4 - BasicsTransformerLM with attention backend injection.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

if TYPE_CHECKING:
    from jaxtyping import Float, Int
    from torch import Tensor

    from ..backends.attention import get_attention_fn

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for Transformer language model.

    Attributes:
        vocab_size: Vocabulary size
        context_length: Maximum sequence length
        d_model: Model hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        rope_theta: RoPE base frequency (default 10000.0)
        dropout: Dropout rate (default 0.0)
    """

    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    rope_theta: float = 10000.0
    dropout: float = 0.0

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "rope_theta": self.rope_theta,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Linear(nn.Module):
    """Linear layer with fan-in/fan-out truncated normal initialization (no bias)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        std = math.sqrt(2 / (d_in + d_out))
        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(d_out, d_in), std=std, a=-3 * std, b=3 * std)
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

    def extra_repr(self):
        return f"d_out={self.weight.shape[0]}, d_in={self.weight.shape[1]}"


class Embedding(nn.Module):
    """Token embedding layer with truncated normal initialization."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        std = 1.0
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(vocab_size, d_model), std=std, a=-3 * std, b=3 * std)
        )

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids, :]

    def extra_repr(self):
        return f"vocab_size={self.weight.shape[0]}, d={self.weight.shape[1]}"


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, context_length: int, dim: int, theta: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache",
            RotaryEmbedding._init_cache(context_length, dim, theta),
            persistent=False,
        )

    @staticmethod
    def _init_cache(context_length: int, dim: int, theta: float) -> Float[Tensor, " 2 context_length half_dim"]:
        """Precompute RoPE cos/sin cache."""
        assert dim % 2 == 0
        d = torch.arange(0, dim, 2) / dim
        freqs = theta**-d
        t = torch.arange(context_length)
        freqs = torch.outer(t, freqs)
        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(
        self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"]
    ) -> Float[Tensor, " ... seq d"]:
        """Apply RoPE rotation to input tensor."""
        import einx

        x1, x2 = rearrange(x, "... (half_d xy) -> xy ... half_d", xy=2)
        cos, sin = einx.get_at(
            "cos_sin [pos] half_dim, ... -> cos_sin ... half_dim", self._freq_cis_cache, pos_ids
        )
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = torch.cat([x1_rot, x2_rot], dim=-1)
        return result.contiguous()


class CausalMultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with RoPE and backend injection."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryEmbedding,
        attention_fn: callable | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)
        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)

        self.positional_encoder = positional_encoder
        self.attention_fn = attention_fn

    def forward(
        self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None
    ) -> Float[Tensor, " ... seq d_v"]:
        """Forward pass with injected attention function."""
        import einx

        *b, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        # Project to Q/K/V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to multi-head
        Q, K, V = (
            rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
            for X in (Q, K, V)
        )

        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=x.device).unsqueeze(0)
            # Broadcast to match batch dimensions
            if len(b) > 0:
                token_positions = token_positions.expand(*b, -1)

        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

        # Apply RoPE to Q and K
        Q = self.positional_encoder(Q, token_positions)
        K = self.positional_encoder(K, token_positions)

        # Use injected attention function or fallback to SDPA
        if self.attention_fn is not None:
            attn_output = self.attention_fn(Q, K, V, is_causal=True)
        else:
            attn_output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # Reshape and project output
        attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()
        output = self.output_proj(attn_output)
        return output


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Single Transformer decoder layer (Pre-LN)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_encoder: RotaryEmbedding,
        attention_fn: callable | None = None,
    ):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            positional_encoder=positional_encoder,
            attention_fn=attention_fn,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln1 = nn.RMSNorm(d_model)
        self.ln2 = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        # Pre-LN: normalize before sublayer
        x_attn = self.attn(self.ln1(x))
        attn_sublayer_output = x + x_attn

        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output


class BasicsTransformerLM(nn.Module):
    """Transformer language model with attention backend injection."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        attention_fn: callable | None = None,
    ):
        self.config = {
            k: v for k, v in locals().items() if k != "self" and not k.startswith("__")
        }
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        self.token_embeddings = Embedding(vocab_size, d_model)
        d_head = d_model // num_heads
        self.positional_encoder = RotaryEmbedding(
            context_length=context_length, dim=d_head, theta=rope_theta
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder,
                    attention_fn=attention_fn,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = nn.RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embeddings.weight.numel()
            n_params -= self.lm_head.weight.numel()
        return n_params

    def forward(self, x: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length vocab_size"]:
        """Language model forward pass."""
        *batch_dims, sequence_length = x.shape
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ):
        """Autoregressive text generation."""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        original_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            logits = self.forward(x)
            next_token_logits = logits[:, -1]
            temperature_scaled_next_token_logits = next_token_logits / temperature

            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(topk_mask, float("-inf"))

            next_token_probabilities = F.softmax(temperature_scaled_next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probabilities, 1)

            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

            x = torch.cat((x, next_token_id), dim=-1)

        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        """Load pretrained model from directory."""
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weights_path, weights_only=True)

        # Remove _orig_mod. prefix from torch.compile
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            unwanted_prefix_start_idx = k.find(unwanted_prefix)
            if unwanted_prefix_start_idx != -1:
                state_dict[k[unwanted_prefix_start_idx + len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        return model


def create_model(config: ModelConfig, use_flash_attn: bool = True) -> BasicsTransformerLM:
    """
    Create a Transformer model with the specified configuration.

    Args:
        config: Model configuration
        use_flash_attn: Whether to use flash attention backend

    Returns:
        Initialized Transformer model
    """
    from ..backends.attention import get_attention_fn

    attention_fn = get_attention_fn(use_flash_attn=use_flash_attn)

    return BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        attention_fn=attention_fn,
    )
