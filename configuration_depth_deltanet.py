"""
HelixNet Configuration

A hybrid transformer architecture combining standard attention with a depth-recurrent
state bank using the Gated Delta Rule. The state evolves across layers (depth) rather
than across time steps, enabling deeper effective computation.
"""

from typing import Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class HelixNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HelixNetModel`]. 
    It is used to instantiate an HelixNet model according to the specified arguments, 
    defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. 
    Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the HelixNet model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`HelixNetModel`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        head_dim (`int`, *optional*, defaults to 64):
            The dimension of the attention heads.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimension of the MLP representations.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder and mlp.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon value used for the RMS normalization layers.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        state_bank_num_heads (`int`, *optional*, defaults to 8):
            Number of heads used for the global whiteboard (state bank) interface.
        state_bank_head_dim (`int`, *optional*, defaults to 64):
            Dimension of each head in the global whiteboard.
        state_bank_expand_v (`int`, *optional*, defaults to 2):
            Expansion factor for the value dimension in the whiteboard. 
            Effective value dim = `state_bank_head_dim * state_bank_expand_v`.
        state_injection_scale (`float`, *optional*, defaults to 1.0):
            Scaling factor for injecting the retrieved whiteboard state back into the residual stream.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie the weights of the input embeddings and the output linear layer.
    """
    
    model_type = "helixnet"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 1024,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 16,
        head_dim: int = 64,
        intermediate_size: int = 4096,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        # Whiteboard / State Bank specific configs
        state_bank_num_heads: int = 8,
        state_bank_head_dim: int = 64,
        state_bank_expand_v: int = 2,
        # General
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        
        # Whiteboard configuration
        self.state_bank_num_heads = state_bank_num_heads
        self.state_bank_head_dim = state_bank_head_dim
        self.state_bank_expand_v = state_bank_expand_v
        
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        
        # Validation: Ensure hidden size is divisible by heads (optional but recommended)
        if hidden_size % num_attention_heads != 0:
            logger.warning(
                f"Hidden size ({hidden_size}) is not divisible by num_attention_heads ({num_attention_heads}). "
                "This is allowed but may lead to unexpected behavior in some attention implementations."
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
