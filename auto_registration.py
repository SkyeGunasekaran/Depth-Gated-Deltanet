"""
Auto-registration utilities for HuggingFace Transformers integration.

This module provides functions to register the DepthDeltaNet model with
HuggingFace's AutoModel and AutoConfig classes, enabling:
- `AutoConfig.from_pretrained("path/to/model")`
- `AutoModelForCausalLM.from_pretrained("path/to/model")`
"""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from configuration_depth_deltanet import DepthDeltaNetConfig
from modeling_depth_deltanet import (
    DepthDeltaNetModel,
    DepthDeltaNetForCausalLM,
)


def register_auto_classes():
    """
    Register DepthDeltaNet with HuggingFace Auto classes.
    
    After calling this function, you can use:
        - AutoConfig.from_pretrained("depth_deltanet_model")
        - AutoModel.from_pretrained("depth_deltanet_model") 
        - AutoModelForCausalLM.from_pretrained("depth_deltanet_model")
    """
    AutoConfig.register("depth_deltanet", DepthDeltaNetConfig)
    AutoModel.register(DepthDeltaNetConfig, DepthDeltaNetModel)
    AutoModelForCausalLM.register(DepthDeltaNetConfig, DepthDeltaNetForCausalLM)


def save_pretrained_with_auto_map(model, save_directory: str, **kwargs):
    """
    Save model with auto_map for standalone loading.
    
    This adds the necessary auto_map to config.json so the model
    can be loaded without explicit imports.
    
    Args:
        model: DepthDeltaNet model instance
        save_directory: Directory to save to
        **kwargs: Additional arguments passed to save_pretrained
    """
    import shutil
    import os
    
    model.config.auto_map = {
        "AutoConfig": "configuration_depth_deltanet.DepthDeltaNetConfig",
        "AutoModel": "modeling_depth_deltanet.DepthDeltaNetModel",
        "AutoModelForCausalLM": "modeling_depth_deltanet.DepthDeltaNetForCausalLM",
    }
    
    model.save_pretrained(save_directory, **kwargs)
    
    source_dir = os.path.dirname(__file__)
    files_to_copy = [
        "configuration_depth_deltanet.py",
        "modeling_depth_deltanet.py",
    ]
    
    for filename in files_to_copy:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(save_directory, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)


def get_config_for_size(size: str = "small") -> DepthDeltaNetConfig:
    """
    Get a predefined configuration for common model sizes.
    
    Args:
        size: One of "tiny", "small", "medium", "large", "xl", "3b", "7b"
        
    Returns:
        DepthDeltaNetConfig instance
    """
    configs = {
        "tiny": {
            "hidden_size": 512,
            "intermediate_size": 1408,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "head_dim": 64,
            "state_bank_num_heads": 4,
            "state_bank_head_dim": 64,
        },
        "small": {
            "hidden_size": 768,
            "intermediate_size": 2048,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "head_dim": 64,
            "state_bank_num_heads": 8,
            "state_bank_head_dim": 64,
        },
        "medium": {
            "hidden_size": 1024,
            "intermediate_size": 2816,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "head_dim": 64,
            "state_bank_num_heads": 8,
            "state_bank_head_dim": 64,
        },
        "large": {
            "hidden_size": 1536,
            "intermediate_size": 4096,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "head_dim": 96,
            "state_bank_num_heads": 12,
            "state_bank_head_dim": 96,
        },
        "xl": {
            "hidden_size": 2048,
            "intermediate_size": 5504,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "head_dim": 128,
            "state_bank_num_heads": 16,
            "state_bank_head_dim": 128,
        },
        "3b": {
            "hidden_size": 2560,
            "intermediate_size": 6912,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "head_dim": 80,
            "state_bank_num_heads": 16,
            "state_bank_head_dim": 80,
        },
        "7b": {
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "head_dim": 128,
            "num_key_value_heads": 32,
            "state_bank_num_heads": 16,
            "state_bank_head_dim": 128,
        },
    }
    
    if size not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Size must be one of: {available}. Got: {size}")
    
    return DepthDeltaNetConfig(**configs[size])