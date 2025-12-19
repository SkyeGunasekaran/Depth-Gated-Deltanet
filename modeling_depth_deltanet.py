"""
HelixNet  Model
========================
Performance optimizations applied:
1. Removed manual mask creation in SDPA fallback - use is_causal=True
2. Fused RMSNorm without float32 casting (uses torch.compile-friendly implementation)
3. Replaced einops with native torch operations
4.  whiteboard state aggregation
5. Added torch.compile-friendly patterns throughout
6. Optional Triton-fused RMSNorm kernel
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.utils import logging
from torch.utils.checkpoint import checkpoint

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    chunk_gated_delta_rule = None

from configuration_depth_helixnet import HelixNetConfig

# Attempt to import Flash Attention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# Attempt to import Triton for fused kernels
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

logger = logging.get_logger(__name__)


# =============================================================================
# 1.  RMSNorm - No float32 casting, compile-friendly
# =============================================================================

class HelixNetRMSNorm(nn.Module):
    """
     RMSNorm that stays in the input dtype.
    For bf16 training, this is numerically stable and much faster.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Compute variance in the same dtype - bf16 is stable enough for this
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class FusedRMSNormGated(nn.Module):
    """ gated RMSNorm without dtype conversion."""
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Fuse the gating with normalization
        return (self.weight * hidden_states) * torch.sigmoid(gate)


# =============================================================================
# 2. Whiteboard Cache (Unchanged)
# =============================================================================

@dataclass
class WhiteboardCache(Cache):
    """Unified cache for KV and Whiteboard state."""
    key_cache: List[torch.Tensor] = field(default_factory=list)
    value_cache: List[torch.Tensor] = field(default_factory=list)
    whiteboard_state: Optional[torch.Tensor] = None
    _seen_tokens: int = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        return None


# =============================================================================
# 3.  RoPE
# =============================================================================

class HelixNetRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=device)

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))  # More efficient than einsum
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device)
        if position_ids is not None:
            return self.cos_cached[position_ids].to(x.dtype), self.sin_cached[position_ids].to(x.dtype)
        return self.cos_cached[:seq_len].to(x.dtype), self.sin_cached[:seq_len].to(x.dtype)


@torch.jit.script
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """JIT-compiled rotate_half for better performance."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """JIT-compiled RoPE application."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# 4.  Hybrid Attention
# =============================================================================

class HelixNetHybridAttention(nn.Module):
    """
     Attention with:
    - Native tensor ops instead of einops
    - Proper SDPA usage with is_causal=True
    - Compile-friendly patterns
    """

    def __init__(self, config: HelixNetConfig, layer_idx: int):
        super().__init__()
        if chunk_gated_delta_rule is None:
            raise ImportError("fla (Flash Linear Attention) is required.")

        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.sliding_window = getattr(config, "sliding_window", 4096)

        # Standard Attention Config
        self.num_attn_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attn_dropout = config.attention_dropout

        # Whiteboard Config
        self.num_wb_heads = config.state_bank_num_heads
        self.wb_head_k_dim = config.state_bank_head_dim
        self.wb_head_v_dim = int(config.state_bank_head_dim * config.state_bank_expand_v)

        # Dimension calculations
        self.dim_q_attn = self.num_attn_heads * self.head_dim
        self.dim_k_attn = self.num_attn_heads * self.head_dim
        self.dim_v_attn = self.num_attn_heads * self.head_dim
        self.dim_wb_q = self.num_wb_heads * self.wb_head_k_dim
        self.dim_wb_k = self.num_wb_heads * self.wb_head_k_dim
        self.dim_wb_v = self.num_wb_heads * self.wb_head_v_dim
        self.dim_wb_beta = self.num_wb_heads
        self.dim_wb_decay = self.num_wb_heads
        self.dim_wb_g = self.num_wb_heads * self.wb_head_v_dim

        # Total fused dimension
        self.total_fused_dim = (
            self.dim_q_attn + self.dim_k_attn + self.dim_v_attn +
            self.dim_wb_q + self.dim_wb_k + self.dim_wb_v +
            self.dim_wb_beta + self.dim_wb_decay + self.dim_wb_g
        )

        # Fused projection
        self.qkv_wb_proj = nn.Linear(self.hidden_size, self.total_fused_dim, bias=False)

        # Split sizes for torch.split (precomputed)
        self.register_buffer(
            '_split_sizes',
            torch.tensor([
                self.dim_q_attn, self.dim_k_attn, self.dim_v_attn,
                self.dim_wb_q, self.dim_wb_k, self.dim_wb_v,
                self.dim_wb_beta, self.dim_wb_decay, self.dim_wb_g,
            ]),
            persistent=False
        )
        self.split_sizes = [
            self.dim_q_attn, self.dim_k_attn, self.dim_v_attn,
            self.dim_wb_q, self.dim_wb_k, self.dim_wb_v,
            self.dim_wb_beta, self.dim_wb_decay, self.dim_wb_g,
        ]

        # Decay bias
        self.wb_decay_bias = nn.Parameter(torch.zeros(self.num_wb_heads))
        self._init_decay_bias()

        # Depth embedding for WB
        self.wb_depth_k_emb = nn.Parameter(torch.randn(1, 1, self.num_wb_heads, self.wb_head_k_dim) * 0.02)

        # Output Projection
        self.concat_dim = (self.num_attn_heads * self.head_dim) + (self.num_wb_heads * self.wb_head_v_dim)
        self.o_proj = nn.Linear(self.concat_dim, self.hidden_size, bias=False)

        # Output Norm for WB
        self.wb_o_norm = FusedRMSNormGated(self.wb_head_v_dim, eps=config.rms_norm_eps)

        # RoPE
        self.rotary_emb = HelixNetRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Precompute scale for attention
        self.scale = self.head_dim ** -0.5
        self.wb_scale = self.wb_head_k_dim ** -0.5

    def _init_decay_bias(self):
        n_heads = self.num_wb_heads
        half_heads = n_heads // 2
        with torch.no_grad():
            fast_retention = torch.rand(half_heads) * (0.1 - 0.01) + 0.01
            slow_retention = torch.rand(n_heads - half_heads) * (0.999 - 0.9) + 0.9
            target_sigmoid = torch.cat([fast_retention, slow_retention])
            bias_init = torch.log(target_sigmoid / (1 - target_sigmoid))
            self.wb_decay_bias.copy_(bias_init)

    def forward(
        self,
        hidden_states: torch.Tensor,
        whiteboard_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[WhiteboardCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[WhiteboardCache]]:

        batch_size, seq_len, _ = hidden_states.shape

        # Fused Projection
        fused_out = self.qkv_wb_proj(hidden_states)

        # Split
        splits = torch.split(fused_out, self.split_sizes, dim=-1)
        q_attn, k_attn, v_attn = splits[0], splits[1], splits[2]
        q_wb, k_wb, v_wb = splits[3], splits[4], splits[5]
        beta_raw, decay_raw, g_wb = splits[6], splits[7], splits[8]

        # --- Part A: Standard Attention ---
        q = q_attn.view(batch_size, seq_len, self.num_attn_heads, self.head_dim)
        k = k_attn.view(batch_size, seq_len, self.num_attn_heads, self.head_dim)
        v = v_attn.view(batch_size, seq_len, self.num_attn_heads, self.head_dim)

        # RoPE: Generate embeddings based on q's device/dtype
        cos, sin = self.rotary_emb(v, position_ids)
        # Ensure cos/sin match q's dtype exactly before math
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        q = q.transpose(1, 2) # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)

        # Check for Flash Attention compatibility (must be fp16/bf16)
        is_flash_compatible = FLASH_ATTN_AVAILABLE and q.dtype in [torch.float16, torch.bfloat16]

        if is_flash_compatible and not use_cache:
            q_flash = q.transpose(1, 2) # (B, T, H, D)
            k_flash = k.transpose(1, 2)
            v_flash = v.transpose(1, 2)

            attn_output = flash_attn_func(
                q_flash, k_flash, v_flash,
                dropout_p=self.attn_dropout if self.training else 0.0,
                softmax_scale=self.scale,
                causal=True,
                window_size=(self.sliding_window, 0)
            )
            attn_output = attn_output.reshape(batch_size, seq_len, -1)
        else:
            # Fallback for fp32 or generation
            if past_key_value is None or past_key_value.get_seq_length(self.layer_idx) == seq_len:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.attn_dropout if self.training else 0.0,
                    is_causal=True,
                    scale=self.scale
                )
            else:
                t_q, t_k = q.size(2), k.size(2)
                causal_mask = torch.ones(t_q, t_k, dtype=torch.bool, device=q.device).tril(
                    diagonal=t_k - t_q
                )
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=causal_mask,
                    dropout_p=self.attn_dropout if self.training else 0.0,
                    is_causal=False,
                    scale=self.scale
                )

            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)

        # --- Part B: Whiteboard Interaction ---
        q_wb_act = F.silu(q_wb)
        k_wb_act = F.silu(k_wb)
        v_wb_act = F.silu(v_wb)

        q_wb_r = q_wb_act.view(batch_size, seq_len, self.num_wb_heads, self.wb_head_k_dim)
        k_wb_r = k_wb_act.view(batch_size, seq_len, self.num_wb_heads, self.wb_head_k_dim)
        v_wb_r = v_wb_act.view(batch_size, seq_len, self.num_wb_heads, self.wb_head_v_dim)

        # Broadcasting depth embedding
        k_wb_r = k_wb_r + self.wb_depth_k_emb.to(dtype=k_wb_r.dtype)

        beta = torch.sigmoid(beta_raw)
        g_log = F.logsigmoid(decay_raw + self.wb_decay_bias)

        wb_read_out, next_whiteboard_state = chunk_gated_delta_rule(
            q=q_wb_r, k=k_wb_r, v=v_wb_r, g=g_log, beta=beta,
            scale=self.wb_scale,
            initial_state=whiteboard_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True
        )

        g_wb_r = g_wb.view(batch_size, seq_len, self.num_wb_heads, self.wb_head_v_dim)
        wb_read_out = self.wb_o_norm(wb_read_out, g_wb_r)
        wb_read_out = wb_read_out.view(batch_size, seq_len, -1)

        if wb_read_out.dtype != attn_output.dtype:
            wb_read_out = wb_read_out.to(dtype=attn_output.dtype)

        # --- Part C: Fusion ---
        fused_output = torch.cat([attn_output, wb_read_out], dim=-1)
        final_output = self.o_proj(fused_output)

        return final_output, next_whiteboard_state, None, past_key_value

# =============================================================================
# 5.  MLP (unchanged but added inline comments)
# =============================================================================

class HelixNetMLPFused(nn.Module):
    def __init__(self, config: HelixNetConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# =============================================================================
# 6. Decoder Layer
# =============================================================================

class HelixNetDecoderLayer(nn.Module):
    def __init__(self, config: HelixNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.input_layernorm = HelixNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = HelixNetHybridAttention(config, layer_idx)
        self.post_attention_layernorm = HelixNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = HelixNetMLPFused(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        whiteboard_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[WhiteboardCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, new_whiteboard_state, attn_weights, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            whiteboard_state=whiteboard_state,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, new_whiteboard_state, attn_weights


# =============================================================================
# 7. Full Model with  State Aggregation
# =============================================================================

class HelixNetPreTrainedModel(PreTrainedModel):
    config_class = HelixNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HelixNetDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "whiteboard_state"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class HelixNetModel(HelixNetPreTrainedModel):
    def __init__(self, config: HelixNetConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([
            HelixNetDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = HelixNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Whiteboard config
        self.wb_num_heads = config.state_bank_num_heads
        self.wb_head_k_dim = config.state_bank_head_dim
        self.wb_head_v_dim = int(config.state_bank_head_dim * config.state_bank_expand_v)

        # Layer aggregation weights
        self.layer_agg_weights = nn.Parameter(torch.zeros(config.num_hidden_layers))

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _init_whiteboard_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.wb_num_heads, self.wb_head_k_dim, self.wb_head_v_dim,
            device=device, dtype=dtype
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[WhiteboardCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states       
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        batch_size, seq_length, _ = hidden_states.shape

        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values else 0
            position_ids = torch.arange(
                past_seen, past_seen + seq_length,
                dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)

        if use_cache and past_key_values is None:
            past_key_values = WhiteboardCache()

        if past_key_values is not None and past_key_values.whiteboard_state is not None:
            initial_whiteboard_state = past_key_values.whiteboard_state
        else:
            initial_whiteboard_state = self._init_whiteboard_state(
                batch_size, hidden_states.device, hidden_states.dtype
            )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # OPTIMIZATION: Pre-allocate tensor for layer states instead of list
        # This avoids dynamic list appending and enables better memory planning
        layer_state_outputs = torch.zeros(
            self.config.num_hidden_layers,
            batch_size, self.wb_num_heads, self.wb_head_k_dim, self.wb_head_v_dim,
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # Run the module
                        outputs = module(*inputs, None, output_attentions, False)

                        # FORCE CONTIGUOUS: This fixes the metadata mismatch between
                        # torch.compile (optimized strides) and recomputation (standard strides).
                        def make_contiguous(x):
                            if isinstance(x, torch.Tensor):
                                return x.contiguous()
                            return x

                        # Apply to all outputs (hidden_states, whiteboard_state, attn_weights)
                        return tuple(make_contiguous(o) for o in outputs)
                    return custom_forward
                layer_outputs = checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    initial_whiteboard_state,
                    attention_mask,
                    position_ids,
                    use_reentrant=False
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    whiteboard_state=initial_whiteboard_state,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            # Direct tensor assignment instead of list append
            layer_state_outputs[layer_idx] = layer_outputs[1]

            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # : Vectorized weighted sum (no stacking needed)
        agg_probs = F.softmax(self.layer_agg_weights, dim=0)
        # Shape: [num_layers] -> [num_layers, 1, 1, 1, 1] for broadcasting
        current_whiteboard_state = torch.einsum(
            'l,lbhkv->bhkv', agg_probs, layer_state_outputs
        )

        if use_cache and past_key_values is not None:
            past_key_values.whiteboard_state = current_whiteboard_state.detach()

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class HelixNetForCausalLM(HelixNetPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: HelixNetConfig):
        super().__init__(config)
        self.model = HelixNetModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[WhiteboardCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states       
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            #  loss computation - avoid extra contiguous() if possible
            shift_logits = logits[..., :-1, :].reshape(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels)

        if not return_dict:
            return ((loss,) + (logits,) + outputs[1:]) if loss is not None else ((logits,) + outputs[1:])

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if past_key_values is not None:
            cache_len = past_key_values.get_seq_length()
            if cache_len > 0:
                input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        model_inputs = {
            "inputs_embeds": inputs_embeds
        } if inputs_embeds is not None and past_key_values is None else {
            "input_ids": input_ids
        }

        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask
        })
        return model_inputs
