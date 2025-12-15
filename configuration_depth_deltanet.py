"""
Depth-Gated DeltaNet Configuration

A hybrid transformer architecture combining standard attention with a depth-recurrent
state bank using the Gated Delta Rule. The state evolves across layers (depth) rather
than across time steps, enabling deeper effective computation.

Paper Reference: Gated Delta Networks (Yang et al., 2024)
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DepthDeltaNetConfig(PretrainedConfig):
    """
    Configuration class for DepthDeltaNet model.
    
    This model hybridizes standard Llama-style attention with a depth-recurrent
    state bank powered by the Gated Delta Rule. The state matrix S evolves across
    layers rather than time, enabling:
    - Long-term information persistence across depth
    - Efficient layer-to-layer communication
    - Deeper effective computation without proportional parameter increase
    
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to None):
            Dimension of the MLP intermediate layer. If None, uses 8/3 * hidden_size
            rounded to nearest 256.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of transformer blocks.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for standard attention.
        head_dim (`int`, *optional*, defaults to 64):
            Dimension of each attention head. If None, uses hidden_size // num_attention_heads.
        num_key_value_heads (`int`, *optional*, defaults to None):
            Number of key-value heads for Grouped Query Attention. If None, uses num_attention_heads.
        
        # State Bank Configuration
        state_bank_num_heads (`int`, *optional*, defaults to None):
            Number of heads for the depth state bank. If None, uses num_attention_heads.
        state_bank_head_dim (`int`, *optional*, defaults to None):
            Head dimension for state bank. If None, uses head_dim.
        state_bank_expand_v (`float`, *optional*, defaults to 2.0):
            Expansion factor for value dimension in state bank.
        state_bank_use_short_conv (`bool`, *optional*, defaults to True):
            Whether to use short convolution for local mixing in state bank.
        state_bank_conv_kernel_size (`int`, *optional*, defaults to 4):
            Kernel size for short convolution.
        depth_init (`bool`, *optional*, defaults to True):
            Use depth initialization (small dt) to encourage state persistence across layers.
        state_bank_gate_logit_normalizer (`int`, *optional*, defaults to 16):
            Normalizer for gate logits.
        
        # Position Embeddings
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            Maximum sequence length.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base value for RoPE frequency computation.
        rope_scaling (`dict`, *optional*, defaults to None):
            Configuration for RoPE scaling (e.g., for long context).
            
        # Regularization
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            Dropout probability for hidden states.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        
        # Normalization
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMS normalization.
        
        # Initialization
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        
        # Architecture choices
        use_cache (`bool`, *optional*, defaults to True):
            Whether to return past key values for caching.
        tie_word_embeddings (`bool`, *optional*, defaults to False):
            Whether to tie input and output embeddings.
        hidden_act (`str`, *optional*, defaults to "silu"):
            Activation function for MLP.
        
        # State bank integration
        state_injection_mode (`str`, *optional*, defaults to "residual"):
            How to inject state bank output: "residual", "gated", or "concat_proj".
        state_injection_scale (`float`, *optional*, defaults to 1.0):
            Scale factor for state bank contribution.
    """
    
    model_type = "depth_deltanet"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 32,
        head_dim: int | None = None,
        num_key_value_heads: int | None = None,
        # State Bank Config
        state_bank_num_heads: int | None = None,
        state_bank_head_dim: int | None = None,
        state_bank_expand_v: float = 2.0,
        state_bank_use_short_conv: bool = True,
        state_bank_conv_kernel_size: int = 4,
        depth_init: bool = True,
        state_bank_gate_logit_normalizer: int = 16,
        # Position Embeddings
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        # Regularization
        hidden_dropout_prob: float = 0.0,
        attention_dropout: float = 0.0,
        # Normalization
        rms_norm_eps: float = 1e-6,
        # Initialization
        initializer_range: float = 0.02,
        # Architecture
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        hidden_act: str = "silu",
        # State injection
        state_injection_mode: str = "residual",
        state_injection_scale: float = 1.0,
        # Padding
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        
        # Head dimension - default to hidden_size // num_attention_heads
        if head_dim is None:
            self.head_dim = hidden_size // num_attention_heads
        else:
            self.head_dim = head_dim
            
        # GQA support
        if num_key_value_heads is None:
            self.num_key_value_heads = num_attention_heads
        else:
            self.num_key_value_heads = num_key_value_heads
        
        # MLP intermediate size
        if intermediate_size is None:
            # Standard Llama formula: 8/3 * hidden_size, rounded to 256
            self.intermediate_size = 256 * ((int(8/3 * hidden_size) + 255) // 256)
        else:
            self.intermediate_size = intermediate_size
        
        # State Bank Configuration
        self.state_bank_num_heads = state_bank_num_heads or num_attention_heads
        self.state_bank_head_dim = state_bank_head_dim or self.head_dim
        self.state_bank_expand_v = state_bank_expand_v
        self.state_bank_use_short_conv = state_bank_use_short_conv
        self.state_bank_conv_kernel_size = state_bank_conv_kernel_size
        self.depth_init = depth_init
        self.state_bank_gate_logit_normalizer = state_bank_gate_logit_normalizer
        
        # Position embeddings
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        
        # Regularization
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout = attention_dropout
        
        # Normalization
        self.rms_norm_eps = rms_norm_eps
        
        # Initialization
        self.initializer_range = initializer_range
        
        # Architecture
        self.use_cache = use_cache
        self.hidden_act = hidden_act
        
        # State injection
        self.state_injection_mode = state_injection_mode
        self.state_injection_scale = state_injection_scale
        
        # Validation
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.state_injection_mode not in ["residual", "gated", "concat_proj"]:
            raise ValueError(
                f"state_injection_mode must be 'residual', 'gated', or 'concat_proj', "
                f"got {self.state_injection_mode}"
            )
        
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) cannot be greater "
                f"than num_attention_heads ({self.num_attention_heads})"
            )
        
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible "
                f"by num_key_value_heads ({self.num_key_value_heads})"
            )
    
    @property
    def state_bank_key_dim(self) -> int:
        """Total key dimension for state bank."""
        return self.state_bank_num_heads * self.state_bank_head_dim
    
    @property
    def state_bank_value_dim(self) -> int:
        """Total value dimension for state bank."""
        return self.state_bank_num_heads * int(self.state_bank_head_dim * self.state_bank_expand_v)
    
    @property
    def state_shape(self) -> tuple:
        """Shape of the state matrix per batch element."""
        head_v_dim = int(self.state_bank_head_dim * self.state_bank_expand_v)
        return (self.state_bank_num_heads, self.state_bank_head_dim, head_v_dim)