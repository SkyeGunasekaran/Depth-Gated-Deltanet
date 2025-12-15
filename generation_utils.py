"""
Generation utilities for Depth-Gated DeltaNet.

This module provides specialized generation functions that properly handle
the depth state cache, enabling efficient autoregressive generation.
"""

from typing import Optional, List, Union, Callable
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    min_new_tokens: int = 0
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    use_cache: bool = True


def sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus (top-p) sampling.
    
    Args:
        probs: Probability distribution (B, V)
        top_p: Cumulative probability threshold
        
    Returns:
        Sampled token indices (B,)
    """
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Keep first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Zero out removed indices
    sorted_probs[sorted_indices_to_remove] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    
    # Sample from filtered distribution
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    
    # Map back to original indices
    next_token = torch.gather(sorted_indices, -1, sampled)
    
    return next_token.squeeze(-1)


def sample_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Top-k sampling.
    
    Args:
        logits: Logits distribution (B, V)
        top_k: Number of top tokens to consider
        
    Returns:
        Sampled token indices (B,)
    """
    top_k = min(top_k, logits.size(-1))
    
    # Get top-k logits and indices
    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
    
    # Sample from top-k
    probs = F.softmax(top_k_logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    
    # Map back to vocabulary indices
    next_token = torch.gather(top_k_indices, -1, sampled)
    
    return next_token.squeeze(-1)


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """
    Apply repetition penalty to logits.
    
    Args:
        logits: Current logits (B, V)
        input_ids: Previously generated tokens (B, T)
        penalty: Penalty factor (>1.0 reduces repetition)
        
    Returns:
        Penalized logits
    """
    if penalty == 1.0:
        return logits
    
    # Create penalty mask
    batch_size = logits.size(0)
    for i in range(batch_size):
        unique_tokens = input_ids[i].unique()
        for token_id in unique_tokens:
            if logits[i, token_id] > 0:
                logits[i, token_id] /= penalty
            else:
                logits[i, token_id] *= penalty
    
    return logits


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    config: Optional[GenerationConfig] = None,
    attention_mask: Optional[torch.Tensor] = None,
    streamer: Optional[Callable] = None,
    stopping_criteria: Optional[List[Callable]] = None,
) -> torch.Tensor:
    """
    Generate text autoregressively with depth state caching.
    
    This function properly manages both the standard KV cache and the
    depth state cache that flows across layers.
    
    Args:
        model: DepthDeltaNetForCausalLM model
        input_ids: Input token IDs (B, T)
        config: Generation configuration
        attention_mask: Attention mask
        streamer: Optional callback for streaming tokens
        stopping_criteria: List of stopping condition functions
        
    Returns:
        Generated token IDs (B, T + max_new_tokens)
    """
    if config is None:
        config = GenerationConfig()
    
    device = input_ids.device
    batch_size = input_ids.size(0)
    
    # Initialize
    generated = input_ids.clone()
    past_key_values = None
    
    # Create attention mask if not provided
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    
    # Track which sequences are done
    unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    for step in range(config.max_new_tokens):
        # Get current input (full sequence on first step, single token after)
        if past_key_values is None:
            current_input = generated
        else:
            current_input = generated[:, -1:]
        
        # Forward pass
        outputs = model(
            input_ids=current_input,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=config.use_cache,
        )
        
        logits = outputs.logits[:, -1, :]  # (B, V)
        past_key_values = outputs.past_key_values
        
        # Apply repetition penalty
        if config.repetition_penalty != 1.0:
            logits = apply_repetition_penalty(
                logits, generated, config.repetition_penalty
            )
        
        # Temperature scaling
        if config.temperature != 1.0:
            logits = logits / config.temperature
        
        # Sampling
        if config.do_sample:
            if config.top_p < 1.0:
                probs = F.softmax(logits, dim=-1)
                next_token = sample_top_p(probs, config.top_p)
            elif config.top_k > 0:
                next_token = sample_top_k(logits, config.top_k)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_token = torch.argmax(logits, dim=-1)
        
        # Update generated sequence
        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
        ], dim=-1)
        
        # Stream token if callback provided
        if streamer is not None:
            streamer(next_token)
        
        # Check EOS
        if config.eos_token_id is not None:
            unfinished_sequences = unfinished_sequences & (next_token != config.eos_token_id)
            if not unfinished_sequences.any():
                break
        
        # Check stopping criteria
        if stopping_criteria is not None:
            for criteria in stopping_criteria:
                if criteria(generated, logits):
                    break
        
        # Minimum length constraint
        if step < config.min_new_tokens:
            continue
    
    return generated


class TextStreamer:
    """
    Simple text streamer for generation.
    
    Usage:
        streamer = TextStreamer(tokenizer)
        generate(model, input_ids, streamer=streamer)
    """
    
    def __init__(self, tokenizer, skip_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.token_cache = []
    
    def __call__(self, token_id: torch.Tensor):
        """Process and print new token."""
        token_id = token_id.cpu().tolist()
        
        if isinstance(token_id, list):
            # Batch mode - just handle first
            token_id = token_id[0]
        
        self.token_cache.append(token_id)
        
        text = self.tokenizer.decode(
            self.token_cache,
            skip_special_tokens=self.skip_special_tokens,
        )
        
        # Print incrementally
        print(text[-1] if len(text) > 0 else "", end="", flush=True)
    
    def end(self):
        """Signal end of generation."""
        print()  # Newline


def get_model_size(model) -> dict:
    """
    Get model size statistics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count by component
    attention_params = 0
    mlp_params = 0
    state_bank_params = 0
    embedding_params = 0
    other_params = 0
    
    for name, p in model.named_parameters():
        if 'embed' in name or 'lm_head' in name:
            embedding_params += p.numel()
        elif 'self_attn' in name:
            attention_params += p.numel()
        elif 'mlp' in name:
            mlp_params += p.numel()
        elif 'state_bank' in name:
            state_bank_params += p.numel()
        else:
            other_params += p.numel()
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'attention_params': attention_params,
        'mlp_params': mlp_params,
        'state_bank_params': state_bank_params,
        'embedding_params': embedding_params,
        'other_params': other_params,
        'total_mb': total_params * 4 / 1024 / 1024,  # Assuming fp32
    }


def print_model_size(model):
    """Print formatted model size information."""
    stats = get_model_size(model)
    
    print("=" * 50)
    print("Model Size Statistics")
    print("=" * 50)
    print(f"Total parameters:     {stats['total_params']:,}")
    print(f"Trainable parameters: {stats['trainable_params']:,}")
    print(f"Model size (fp32):    {stats['total_mb']:.2f} MB")
    print("-" * 50)
    print("Parameter breakdown:")
    print(f"  Embeddings:   {stats['embedding_params']:,} ({100*stats['embedding_params']/stats['total_params']:.1f}%)")
    print(f"  Attention:    {stats['attention_params']:,} ({100*stats['attention_params']/stats['total_params']:.1f}%)")
    print(f"  MLP:          {stats['mlp_params']:,} ({100*stats['mlp_params']/stats['total_params']:.1f}%)")
    print(f"  State Bank:   {stats['state_bank_params']:,} ({100*stats['state_bank_params']/stats['total_params']:.1f}%)")
    print(f"  Other:        {stats['other_params']:,} ({100*stats['other_params']/stats['total_params']:.1f}%)")
    print("=" * 50)