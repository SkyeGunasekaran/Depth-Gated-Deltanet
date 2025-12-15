"""
Depth-Gated DeltaNet Model Implementation

A hybrid transformer combining standard attention with depth-recurrent state banks.
The state matrix S evolves across layers (depth) rather than time, enabling deeper
effective computation and long-term information persistence.

Architecture Overview:
1. Each layer has standard Llama-style attention + MLP (main stream)
2. Each layer has a state bank (DepthGatedDeltaNet) that:
   - Receives the main stream output
   - Updates a persistent state matrix S using the Gated Delta Rule
   - Returns a query result that gets added to the residual
3. The state S flows from layer L to layer L+1 (depth recurrence)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

from configuration_depth_deltanet import DepthDeltaNetConfig

logger = logging.get_logger(__name__)


# =============================================================================
# Custom Cache for Depth State
# =============================================================================

@dataclass
class DepthDeltaNetCacheOutput:
    """
    Output container that includes both standard KV cache and depth state.
    
    Attributes:
        past_key_values: Standard attention KV cache (for each layer)
        depth_states: State matrices for each layer's state bank
        conv_states: Convolution states for short convolutions (if used)
    """
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    depth_states: Optional[Tuple[torch.Tensor]] = None
    conv_states: Optional[Tuple[Tuple[torch.Tensor]]] = None


class DepthDeltaNetCache(DynamicCache):
    """
    Cache class for DepthDeltaNet that handles both:
    1. Standard attention KV cache (for autoregressive generation)
    2. Depth state matrices (passed layer-to-layer, persisted across tokens)
    3. Convolution states for short convolutions
    
    The key insight: depth states are updated WITHIN a forward pass (layer to layer),
    while the final state from layer N is stored and used as the initial state
    for the NEXT forward pass (next token generation step).
    """
    
    def __init__(self):
        super().__init__()
        # Standard KV cache: List[Tuple[key, value]] per layer
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
        # Depth state: The state matrix for each layer's state bank
        # During generation, we store the final_state from each layer
        # and use it as initial_state in the next token's forward pass
        self.depth_states: List[Optional[torch.Tensor]] = []
        
        # Convolution states for incremental conv processing
        # Each layer has (q_conv_state, k_conv_state, v_conv_state)
        self.conv_states: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        
        self._seen_tokens = 0
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]
    
    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length."""
        return None
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the KV cache for a layer.
        
        Args:
            key_states: New key states to add
            value_states: New value states to add  
            layer_idx: Which layer this is for
            cache_kwargs: Additional arguments (unused)
            
        Returns:
            Tuple of (full_key_cache, full_value_cache) for this layer
        """
        # Initialize if needed
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Concatenate along sequence dimension
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def update_depth_state(
        self,
        new_state: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """
        Update the depth state for a layer.
        
        This stores the final state from the layer's state bank,
        to be used as initial_state in the next forward pass.
        """
        while len(self.depth_states) <= layer_idx:
            self.depth_states.append(None)
        self.depth_states[layer_idx] = new_state
    
    def get_depth_state(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get the stored depth state for a layer."""
        if layer_idx < len(self.depth_states):
            return self.depth_states[layer_idx]
        return None
    
    def update_conv_states(
        self,
        conv_states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        layer_idx: int,
    ) -> None:
        """Update convolution states for incremental processing."""
        while len(self.conv_states) <= layer_idx:
            self.conv_states.append(None)
        self.conv_states[layer_idx] = conv_states
    
    def get_conv_states(
        self, 
        layer_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get stored convolution states for a layer."""
        if layer_idx < len(self.conv_states):
            return self.conv_states[layer_idx]
        return None



# =============================================================================
# RMSNorm
# =============================================================================

class DepthDeltaNetRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Matches Llama implementation: no bias, no mean subtraction.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FusedRMSNormGated(nn.Module):
    """
    Gated RMS Normalization: norm(x) * sigmoid(g) 
    Used in the state bank output path.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        gate: torch.Tensor
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        return hidden_states * F.sigmoid(gate)


# =============================================================================
# Rotary Position Embeddings
# =============================================================================

class DepthDeltaNetRotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) with optional scaling.
    
    Supports:
    - Standard RoPE
    - Linear scaling
    - Dynamic NTK scaling
    - YaRN-style scaling (future)
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        rope_type: str = "default",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.rope_type = rope_type
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings, device=device)
    
    def _set_cos_sin_cache(
        self, 
        seq_len: int, 
        device: Optional[torch.device] = None
    ):
        """Build and cache cos/sin tensors for efficiency."""
        self.max_seq_len_cached = seq_len
        
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        # Apply scaling
        if self.rope_type == "linear":
            t = t / self.scaling_factor
        elif self.rope_type == "dynamic":
            # Dynamic NTK scaling
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) 
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin for given positions.
        
        Args:
            x: Input tensor (for shape/device inference)
            position_ids: Optional explicit position IDs
            
        Returns:
            Tuple of (cos, sin) tensors
        """
        seq_len = x.shape[-2]
        
        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device)
        
        if position_ids is not None:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        else:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine part of RoPE
        sin: Sine part of RoPE
        position_ids: Position indices
        unsqueeze_dim: Dimension to unsqueeze for broadcasting
        
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


# =============================================================================
# Short Convolution for Local Mixing
# =============================================================================

class ShortConvolution(nn.Module):
    """
    Causal 1D convolution for local mixing.
    
    Used in the state bank to provide local context before the
    gated delta rule computation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        activation: str = "silu",
        use_bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        
        # Causal conv: pad on left only
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size - 1,  # Will trim right side
            groups=hidden_size,  # Depthwise
            bias=use_bias,
        )
        
        if activation == "silu":
            self.activation = F.silu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            self.activation = None
    
    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        cache_enabled: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply causal convolution.
        
        Args:
            x: Input tensor (B, T, D)
            conv_state: Previous conv state for incremental decoding (B, D, K-1)
            cache_enabled: Whether to return updated conv state
            
        Returns:
            Tuple of (output, new_conv_state)
        """
        batch_size, seq_len, _ = x.shape
        
        # (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        
        if conv_state is not None and seq_len == 1:
            # Incremental mode: prepend state, single step
            x = torch.cat([conv_state, x], dim=-1)
            y = self.conv(x)[..., -1:]  # Only take last output
            new_state = x[..., -(self.kernel_size - 1):] if cache_enabled else None
        else:
            # Full sequence mode
            y = self.conv(x)[..., :seq_len]  # Trim right padding
            new_state = x[..., -(self.kernel_size - 1):] if cache_enabled else None
        
        # (B, D, T) -> (B, T, D)
        y = y.transpose(1, 2)
        
        if self.activation is not None:
            y = self.activation(y)
        
        return y, new_state


# =============================================================================
# Gated Delta Rule (Reference Implementation)
# =============================================================================

def gated_delta_rule_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Reference implementation of the Gated Delta Rule.
    
    The delta rule updates a state matrix S as:
        S_{t+1} = alpha_t * S_t + beta_t * (v_t - S_t @ k_t) @ k_t^T
        o_t = S_t @ q_t
    
    Where:
        - alpha_t = exp(g_t) is the decay/retention gate
        - beta_t is the write strength gate
        - The (v - Sk)k^T term is the "delta" update
    
    This is a naive recurrence implementation for correctness verification.
    In practice, use the optimized chunk_gated_delta_rule from fla library.
    
    Args:
        q: Query tensor (B, T, H, D_k)
        k: Key tensor (B, T, H, D_k)
        v: Value tensor (B, T, H, D_v)
        g: Decay gate logits (B, T, H) - will be exp'd for alpha
        beta: Write strength (B, T, H) - already in [0, 1]
        initial_state: Initial state matrix (B, H, D_k, D_v)
        output_final_state: Whether to return final state
        use_qk_l2norm: Whether to L2 normalize q and k
        
    Returns:
        Tuple of (output, final_state)
        - output: (B, T, H, D_v)
        - final_state: (B, H, D_k, D_v) if output_final_state else None
    """
    batch_size, seq_len, num_heads, head_k_dim = q.shape
    head_v_dim = v.shape[-1]
    
    # L2 normalize q and k if requested
    if use_qk_l2norm:
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
    
    # Initialize state
    if initial_state is None:
        state = torch.zeros(
            batch_size, num_heads, head_k_dim, head_v_dim,
            dtype=q.dtype, device=q.device
        )
    else:
        state = initial_state.clone()
    
    # Compute alpha from g (decay gate)
    alpha = torch.exp(g)  # (B, T, H)
    
    outputs = []
    
    for t in range(seq_len):
        # Get current timestep tensors
        q_t = q[:, t]  # (B, H, D_k)
        k_t = k[:, t]  # (B, H, D_k)
        v_t = v[:, t]  # (B, H, D_v)
        alpha_t = alpha[:, t, :, None, None]  # (B, H, 1, 1)
        beta_t = beta[:, t, :, None]  # (B, H, 1)
        
        # Query the current state: o_t = S @ q
        # state: (B, H, D_k, D_v), q_t: (B, H, D_k) -> (B, H, D_v)
        o_t = torch.einsum('bhkv,bhk->bhv', state, q_t)
        outputs.append(o_t)
        
        # Compute delta: v - S @ k
        # state @ k: (B, H, D_k, D_v) @ (B, H, D_k) -> (B, H, D_v)
        Sk = torch.einsum('bhkv,bhk->bhv', state, k_t)
        delta = v_t - Sk  # (B, H, D_v)
        
        # Update state: S = alpha * S + beta * delta @ k^T
        # delta @ k^T: (B, H, D_v) @ (B, H, D_k) -> (B, H, D_k, D_v)
        # We want outer product, so (B, H, D_v, 1) @ (B, H, 1, D_k) -> (B, H, D_v, D_k)
        # Then transpose to (B, H, D_k, D_v)
        update = torch.einsum('bhv,bhk->bhkv', delta, k_t)
        state = alpha_t * state + beta_t.unsqueeze(-1) * update
    
    # Stack outputs: List of (B, H, D_v) -> (B, T, H, D_v)
    output = torch.stack(outputs, dim=1)
    
    final_state = state if output_final_state else None
    
    return output, final_state


# =============================================================================
# Depth-Gated Delta Net State Bank
# =============================================================================

class DepthGatedDeltaNetStateBank(nn.Module):
    """
    The State Bank: a Gated DeltaNet module for depth-recurrent architectures.
    
    This module:
    1. Takes hidden states from the main transformer stream
    2. Updates a persistent state matrix S using the Gated Delta Rule
    3. Returns a query result to be injected back into the residual stream
    
    The key innovation is that the state S flows ACROSS LAYERS (depth),
    not across time steps. Each layer receives the state from the previous
    layer, updates it, and passes it to the next layer.
    
    For autoregressive generation, the final state from layer N after
    processing token t is cached and used as the initial state when
    processing token t+1.
    """
    
    def __init__(self, config: DepthDeltaNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.state_bank_num_heads
        self.head_k_dim = config.state_bank_head_dim
        self.head_v_dim = int(config.state_bank_head_dim * config.state_bank_expand_v)
        
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_heads * self.head_v_dim
        
        # Projections
        self.q_proj = nn.Linear(config.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        
        # Gating projections
        # a_proj: for decay gate (alpha)
        # b_proj: for write strength (beta)
        self.a_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.b_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        
        # Output gating and projection
        self.g_proj = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)
        
        # Short convolution for local mixing
        self.use_short_conv = config.state_bank_use_short_conv
        if self.use_short_conv:
            conv_kernel = config.state_bank_conv_kernel_size
            self.q_conv = ShortConvolution(self.key_dim, conv_kernel, activation='silu')
            self.k_conv = ShortConvolution(self.key_dim, conv_kernel, activation='silu')
            self.v_conv = ShortConvolution(self.value_dim, conv_kernel, activation='silu')
        
        # Initialize decay parameters (crucial for depth stability)
        self._init_decay_parameters(config)
    
    def _init_decay_parameters(self, config: DepthDeltaNetConfig):
        """
        Initialize decay-related parameters.
        
        For depth recurrence, we want the state to persist across layers,
        so we initialize with small dt (high retention).
        """
        # A_log: log of decay rate base
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(
            0, config.state_bank_gate_logit_normalizer
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        # dt_bias: bias for temporal discretization
        # Small dt means high retention (alpha close to 1)
        if config.depth_init:
            dt_min, dt_max = 0.0001, 0.001
        else:
            dt_min, dt_max = 0.001, 0.1
        
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min)) 
            + math.log(dt_min)
        )
        # Inverse softplus to get bias that produces dt after softplus
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        previous_state: Optional[torch.Tensor] = None,
        conv_states: Optional[Tuple[torch.Tensor, ...]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:
        """
        Process input through the state bank.
        
        Args:
            hidden_states: Input tensor (B, T, D)
            previous_state: State from previous layer or cached (B, H, D_k, D_v)
            conv_states: Cached convolution states for incremental decoding
            use_cache: Whether to return states for caching
            
        Returns:
            Tuple of:
            - output: Processed tensor (B, T, D)
            - new_state: Updated state matrix (B, H, D_k, D_v)
            - new_conv_states: Updated conv states (if use_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Short convolution for local mixing
        new_conv_states = None
        if self.use_short_conv:
            q_conv_state = conv_states[0] if conv_states else None
            k_conv_state = conv_states[1] if conv_states else None
            v_conv_state = conv_states[2] if conv_states else None
            
            q, new_q_conv = self.q_conv(q, q_conv_state, cache_enabled=use_cache)
            k, new_k_conv = self.k_conv(k, k_conv_state, cache_enabled=use_cache)
            v, new_v_conv = self.v_conv(v, v_conv_state, cache_enabled=use_cache)
            
            if use_cache:
                new_conv_states = (new_q_conv, new_k_conv, new_v_conv)
        else:
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)
        
        # Reshape for multi-head
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_heads)
        
        # Compute gates
        # Beta: write strength [0, 1]
        beta = self.b_proj(hidden_states).sigmoid()
        
        # G (decay): computed from A_log and softplus(a_proj(x) + dt_bias)
        # Negative because we want decay (g < 0 means alpha = exp(g) < 1)
        g = -self.A_log.float().exp() * F.softplus(
            self.a_proj(hidden_states).float() + self.dt_bias
        )
        
        # Apply gated delta rule
        # Try to use optimized kernel if available, fall back to reference
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            o, new_state = chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=previous_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True
            )
        except ImportError:
            # Fall back to reference implementation
            o, new_state = gated_delta_rule_recurrence(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=previous_state,
                output_final_state=True,
                use_qk_l2norm=True
            )
        
        # Output gating and projection
        g_out = rearrange(self.g_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_heads)
        o = self.o_norm(o, g_out)
        o = rearrange(o, 'b t h d -> b t (h d)')
        output = self.o_proj(o)
        
        return output, new_state, new_conv_states


# =============================================================================
# Standard Attention
# =============================================================================

class DepthDeltaNetAttention(nn.Module):
    """
    Multi-head attention with RoPE and optional GQA support.
    Standard Llama-style attention for the main transformer stream.
    """
    
    def __init__(self, config: DepthDeltaNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.attention_dropout = config.attention_dropout
        
        # Projections
        self.q_proj = nn.Linear(
            config.hidden_size, 
            self.num_heads * self.head_dim, 
            bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, 
            config.hidden_size, 
            bias=False
        )
        
        # RoPE
        self.rotary_emb = DepthDeltaNetRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for attention.
        
        Args:
            hidden_states: Input (B, T, D)
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: KV cache
            output_attentions: Whether to return attention weights
            use_cache: Whether to use/update cache
            
        Returns:
            Tuple of (output, attention_weights, updated_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape: (B, T, H, D)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=2)
        
        # Transpose for attention: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Handle KV cache
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)
        
        # Repeat KV for GQA
        if self.num_key_value_groups > 1:
            k = repeat(k, 'b h t d -> b (h g) t d', g=self.num_key_value_groups)
            v = repeat(v, 'b h t d -> b (h g) t d', g=self.num_key_value_groups)
        
        # Handle mask dtype 
        if attention_mask is not None and attention_mask.dtype == torch.long:
            attention_mask = attention_mask.bool()

        # 2. Determine causality and handle mask
        # If seq_len > 1, we are in prefill/training and need causal masking.
        # If attention_mask is all True (no padding), we can drop it and use is_causal=True.
        use_causal = False
        if seq_len > 1:
            if attention_mask is None:
                use_causal = True
            elif attention_mask.all():
                # Optimization: if mask is all 1s (no padding), ignore it and enforce causal
                attention_mask = None
                use_causal = True
        
        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=use_causal,
        )        
        # Reshape back: (B, H, T, D) -> (B, T, H*D)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value


# =============================================================================
# MLP
# =============================================================================

class DepthDeltaNetMLP(nn.Module):
    """
    Llama-style MLP with SwiGLU activation.
    """
    
    def __init__(self, config: DepthDeltaNetConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
        if config.hidden_act == "silu":
            self.act_fn = F.silu
        elif config.hidden_act == "gelu":
            self.act_fn = F.gelu
        else:
            self.act_fn = F.silu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# Hybrid Decoder Layer
# =============================================================================

class DepthDeltaNetDecoderLayer(nn.Module):
    """
    A single decoder layer combining:
    1. Standard attention + MLP (main stream)
    2. State bank interaction (depth recurrence)
    
    The state bank receives the main stream output, updates its state S,
    and returns a query result that gets added to the residual.
    """
    
    def __init__(self, config: DepthDeltaNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Main stream: Pre-norm attention
        self.input_layernorm = DepthDeltaNetRMSNorm(
            config.hidden_size, 
            eps=config.rms_norm_eps
        )
        self.self_attn = DepthDeltaNetAttention(config, layer_idx)
        
        # Main stream: Pre-norm MLP
        self.post_attention_layernorm = DepthDeltaNetRMSNorm(
            config.hidden_size, 
            eps=config.rms_norm_eps
        )
        self.mlp = DepthDeltaNetMLP(config)
        
        # State bank
        self.state_bank_layernorm = DepthDeltaNetRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )
        self.state_bank = DepthGatedDeltaNetStateBank(config, layer_idx)
        
        # State injection configuration
        self.state_injection_mode = config.state_injection_mode
        self.state_injection_scale = config.state_injection_scale
        
        # Optional gating for state injection
        if self.state_injection_mode == "gated":
            self.state_gate = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        previous_depth_state: Optional[torch.Tensor] = None,
        previous_conv_states: Optional[Tuple[torch.Tensor, ...]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for decoder layer.
        
        Args:
            hidden_states: Input (B, T, D)
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: KV cache for attention
            previous_depth_state: State from previous layer / cached
            previous_conv_states: Cached conv states for state bank
            output_attentions: Whether to return attention weights
            use_cache: Whether to use/update caches
            
        Returns:
            Tuple of (hidden_states, attention_weights, cache, depth_state, conv_states)
        """
        # === Main Stream: Attention ===
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_output, attn_weights, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output
        
        # === Main Stream: MLP ===
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        main_stream_output = residual + mlp_output
        
        # === State Bank Interaction ===
        bank_input = self.state_bank_layernorm(main_stream_output)
        state_query_output, new_depth_state, new_conv_states = self.state_bank(
            hidden_states=bank_input,
            previous_state=previous_depth_state,
            conv_states=previous_conv_states,
            use_cache=use_cache,
        )
        
        # === State Injection ===
        if self.state_injection_mode == "residual":
            hidden_states = main_stream_output + self.state_injection_scale * state_query_output
        elif self.state_injection_mode == "gated":
            gate_input = torch.cat([main_stream_output, state_query_output], dim=-1)
            gate = torch.sigmoid(self.state_gate(gate_input))
            hidden_states = main_stream_output + gate * state_query_output
        else:  # concat_proj - would need additional projection
            hidden_states = main_stream_output + state_query_output
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (attn_weights,)
        
        if use_cache:
            outputs += (past_key_value, new_depth_state, new_conv_states)
        
        return outputs


# =============================================================================
# Full Model
# =============================================================================

class DepthDeltaNetPreTrainedModel(PreTrainedModel):
    """Base class for DepthDeltaNet models."""
    
    config_class = DepthDeltaNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DepthDeltaNetDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "depth_states"]
    
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


class DepthDeltaNetModel(DepthDeltaNetPreTrainedModel):
    """
    The bare DepthDeltaNet Model outputting raw hidden-states.
    
    This model combines standard transformer attention with depth-recurrent
    state banks. The state flows across layers (depth) providing persistent
    memory that can be queried at each layer.
    """
    
    def __init__(self, config: DepthDeltaNetConfig):
        super().__init__(config)
        self.config = config
        
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=config.pad_token_id
        )
        
        self.layers = nn.ModuleList([
            DepthDeltaNetDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = DepthDeltaNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, DepthDeltaNetCache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass.
        
        The key difference from standard transformers is the depth state
        that flows from layer to layer. For generation, this state is
        cached and reused across token generation steps.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Input handling
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds
        
        # Position IDs
        if position_ids is None:
            past_seen_tokens = 0
            if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
                past_seen_tokens = past_key_values.get_seq_length()
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_length,
                dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)
        
        # Initialize cache if needed
        if use_cache and past_key_values is None:
            past_key_values = DepthDeltaNetCache()
        
        # Prepare outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        # === Layer-by-layer processing with depth state flow ===
        # The depth state starts as None and evolves across layers
        current_depth_state = None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # Get cached states for this layer
            if past_key_values is not None and isinstance(past_key_values, DepthDeltaNetCache):
                # For the first layer, get cached depth state from previous forward pass
                # For subsequent layers, use the state from previous layer
                if idx == 0:
                    cached_depth_state = past_key_values.get_depth_state(idx)
                    if cached_depth_state is not None:
                        current_depth_state = cached_depth_state
                conv_states = past_key_values.get_conv_states(idx)
            else:
                conv_states = None
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values if isinstance(past_key_values, Cache) else None,
                    current_depth_state,
                    conv_states,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    previous_depth_state=current_depth_state,
                    previous_conv_states=conv_states,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            if use_cache:
                # Update depth state for next layer
                new_depth_state = layer_outputs[-2]
                new_conv_states = layer_outputs[-1]
                current_depth_state = new_depth_state
                
                # Cache the state for this layer (for next forward pass)
                if isinstance(past_key_values, DepthDeltaNetCache):
                    past_key_values.update_depth_state(new_depth_state, idx)
                    if new_conv_states is not None:
                        past_key_values.update_conv_states(new_conv_states, idx)
        
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DepthDeltaNetForCausalLM(DepthDeltaNetPreTrainedModel, GenerationMixin):
    """
    DepthDeltaNet Model with a language modeling head.
    """
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: DepthDeltaNetConfig):
        super().__init__(config)
        self.model = DepthDeltaNetModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def set_decoder(self, decoder):
        self.model = decoder
    
    def get_decoder(self):
        return self.model
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, DepthDeltaNetCache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for causal language modeling.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_values: Cached key-values and depth states
            inputs_embeds: Input embeddings (alternative to input_ids)
            labels: Labels for computing loss
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict
            
        Returns:
            CausalLMOutputWithPast containing loss, logits, and cached states
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through base model
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
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        # Compute loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> dict:
        """
        Prepare inputs for generation step.
        
        This handles:
        1. Slicing input_ids when using cache (only process new tokens)
        2. Managing position_ids for correct RoPE application
        3. Preserving depth states across generation steps
        """
        # If we have cache, only process new tokens
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
            else:
                cache_length = 0
            
            if cache_length > 0:
                # Only use the last token
                input_ids = input_ids[:, -1:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # Create position_ids on the fly
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        
        # If embeds are passed, only use them for first generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorder cache for beam search.
        
        Handles both standard KV cache and depth states.
        """
        if isinstance(past_key_values, DepthDeltaNetCache):
            # Reorder KV cache
            reordered_past = DepthDeltaNetCache()
            for layer_idx in range(len(past_key_values.key_cache)):
                reordered_past.key_cache.append(
                    past_key_values.key_cache[layer_idx].index_select(0, beam_idx)
                )
                reordered_past.value_cache.append(
                    past_key_values.value_cache[layer_idx].index_select(0, beam_idx)
                )
            
            # Reorder depth states
            for layer_idx, state in enumerate(past_key_values.depth_states):
                if state is not None:
                    reordered_past.depth_states.append(
                        state.index_select(0, beam_idx)
                    )
                else:
                    reordered_past.depth_states.append(None)
            
            # Reorder conv states
            for layer_idx, conv_state in enumerate(past_key_values.conv_states):
                if conv_state is not None:
                    reordered_conv = tuple(
                        s.index_select(0, beam_idx) for s in conv_state
                    )
                    reordered_past.conv_states.append(reordered_conv)
                else:
                    reordered_past.conv_states.append(None)
            
            return reordered_past
        else:
            # Standard cache reordering
            return tuple(
                tuple(
                    past_state.index_select(0, beam_idx) 
                    for past_state in layer_past
                )
                for layer_past in past_key_values
            )