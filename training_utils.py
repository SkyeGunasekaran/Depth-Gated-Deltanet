"""
Training utilities for Depth-Gated DeltaNet.

This module provides training helpers including:
- Learning rate schedulers appropriate for depth-recurrent models
- Gradient clipping and monitoring utilities  
- Training loop helpers
- Loss computation utilities
"""

import math
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    
    # Learning rate schedule
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    warmup_ratio: float = 0.0  # Alternative to warmup_steps
    num_training_steps: int = 100000
    min_lr_ratio: float = 0.1  # Minimum LR as ratio of peak
    
    # Gradient handling
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # State bank specific
    state_bank_lr_multiplier: float = 1.0  # Can use different LR for state bank
    decay_param_lr_multiplier: float = 0.1  # Lower LR for A_log and dt_bias
    
    # Training settings
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000


def get_parameter_groups(
    model: nn.Module,
    config: TrainingConfig,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with different learning rates.
    
    Groups:
    1. Regular parameters (default LR, with weight decay)
    2. No-decay parameters (biases, norms - no weight decay)
    3. State bank parameters (optional different LR)
    4. Decay parameters (A_log, dt_bias - lower LR, no decay)
    """
    # Collect parameter names for no weight decay
    no_decay_patterns = ["bias", "layernorm", "rmsnorm", "norm"]
    
    # Collect decay parameters (state bank specific)
    decay_param_names = ["A_log", "dt_bias"]
    
    # State bank patterns
    state_bank_patterns = ["state_bank"]
    
    # Categorize parameters
    regular_decay = []
    regular_no_decay = []
    state_bank_decay = []
    state_bank_no_decay = []
    decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if it's a decay parameter (A_log, dt_bias)
        if any(dp in name for dp in decay_param_names):
            decay_params.append(param)
            continue
        
        # Check if state bank parameter
        is_state_bank = any(sp in name for sp in state_bank_patterns)
        
        # Check if no-decay parameter
        is_no_decay = any(nd in name.lower() for nd in no_decay_patterns)
        
        if is_state_bank:
            if is_no_decay:
                state_bank_no_decay.append(param)
            else:
                state_bank_decay.append(param)
        else:
            if is_no_decay:
                regular_no_decay.append(param)
            else:
                regular_decay.append(param)
    
    # Build parameter groups
    param_groups = []
    
    if regular_decay:
        param_groups.append({
            "params": regular_decay,
            "weight_decay": config.weight_decay,
            "lr": config.learning_rate,
        })
    
    if regular_no_decay:
        param_groups.append({
            "params": regular_no_decay,
            "weight_decay": 0.0,
            "lr": config.learning_rate,
        })
    
    if state_bank_decay:
        param_groups.append({
            "params": state_bank_decay,
            "weight_decay": config.weight_decay,
            "lr": config.learning_rate * config.state_bank_lr_multiplier,
        })
    
    if state_bank_no_decay:
        param_groups.append({
            "params": state_bank_no_decay,
            "weight_decay": 0.0,
            "lr": config.learning_rate * config.state_bank_lr_multiplier,
        })
    
    if decay_params:
        param_groups.append({
            "params": decay_params,
            "weight_decay": 0.0,  # These should not have weight decay
            "lr": config.learning_rate * config.decay_param_lr_multiplier,
        })
    
    return param_groups


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> LambdaLR:
    """
    Create learning rate scheduler.
    
    Supports:
    - linear: Linear warmup then linear decay
    - cosine: Linear warmup then cosine decay
    - constant: Constant LR after warmup
    """
    num_warmup_steps = config.warmup_steps
    if config.warmup_ratio > 0:
        num_warmup_steps = int(config.num_training_steps * config.warmup_ratio)
    
    def lr_lambda(current_step: int) -> float:
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # After warmup
        progress = float(current_step - num_warmup_steps) / float(
            max(1, config.num_training_steps - num_warmup_steps)
        )
        
        if config.lr_scheduler_type == "linear":
            return max(config.min_lr_ratio, 1.0 - progress * (1.0 - config.min_lr_ratio))
        elif config.lr_scheduler_type == "cosine":
            return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        elif config.lr_scheduler_type == "constant":
            return 1.0
        else:
            raise ValueError(f"Unknown scheduler type: {config.lr_scheduler_type}")
    
    return LambdaLR(optimizer, lr_lambda)


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: TrainingConfig,
) -> tuple:
    """
    Create optimizer and scheduler with proper parameter groups.
    
    Returns:
        Tuple of (optimizer, scheduler)
    """
    param_groups = get_parameter_groups(model, config)
    
    optimizer = AdamW(
        param_groups,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )
    
    scheduler = get_scheduler(optimizer, config)
    
    return optimizer, scheduler


class GradientMonitor:
    """
    Monitor gradient statistics during training.
    
    Useful for debugging depth-recurrent models where gradient
    flow through the state bank is critical.
    """
    
    def __init__(self, model: nn.Module, log_freq: int = 100):
        self.model = model
        self.log_freq = log_freq
        self.step = 0
        self.history = {
            "grad_norm": [],
            "state_bank_grad_norm": [],
            "attention_grad_norm": [],
            "mlp_grad_norm": [],
        }
    
    def log_gradients(self) -> Dict[str, float]:
        """Compute and log gradient statistics."""
        self.step += 1
        
        if self.step % self.log_freq != 0:
            return {}
        
        stats = {}
        
        # Total gradient norm
        total_norm = 0.0
        state_bank_norm = 0.0
        attention_norm = 0.0
        mlp_norm = 0.0
        
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            
            param_norm = param.grad.data.norm(2).item() ** 2
            total_norm += param_norm
            
            if "state_bank" in name:
                state_bank_norm += param_norm
            elif "self_attn" in name:
                attention_norm += param_norm
            elif "mlp" in name:
                mlp_norm += param_norm
        
        stats["grad_norm"] = total_norm ** 0.5
        stats["state_bank_grad_norm"] = state_bank_norm ** 0.5
        stats["attention_grad_norm"] = attention_norm ** 0.5
        stats["mlp_grad_norm"] = mlp_norm ** 0.5
        
        # Store history
        for key, value in stats.items():
            self.history[key].append(value)
        
        return stats


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """
    Compute training metrics.
    
    Args:
        logits: Model logits (B, T, V)
        labels: Target labels (B, T)
        ignore_index: Label index to ignore
        
    Returns:
        Dict with loss, perplexity, accuracy
    """
    # Shift for autoregressive loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    
    # Compute loss
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fct(flat_logits, flat_labels)
    
    # Compute perplexity
    perplexity = torch.exp(loss).item()
    
    # Compute accuracy (ignoring padding)
    mask = flat_labels != ignore_index
    predictions = flat_logits.argmax(dim=-1)
    correct = (predictions == flat_labels) & mask
    accuracy = correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0
    
    return {
        "loss": loss.item(),
        "perplexity": perplexity,
        "accuracy": accuracy,
    }


class DepthStateAnalyzer:
    """
    Analyze depth state dynamics during training/inference.
    
    This helps understand how information flows through the depth
    state bank across layers.
    """
    
    def __init__(self):
        self.state_norms = []
        self.state_changes = []
    
    @torch.no_grad()
    def analyze_states(
        self,
        states: List[Optional[torch.Tensor]],
    ) -> Dict[str, Any]:
        """
        Analyze state matrices across layers.
        
        Args:
            states: List of state tensors, one per layer
            
        Returns:
            Dict with analysis results
        """
        norms = []
        changes = []
        
        prev_state = None
        for idx, state in enumerate(states):
            if state is None:
                norms.append(0.0)
                changes.append(0.0)
                continue
            
            # State norm
            norm = state.norm().item()
            norms.append(norm)
            
            # Change from previous layer
            if prev_state is not None:
                change = (state - prev_state).norm().item()
                changes.append(change)
            else:
                changes.append(0.0)
            
            prev_state = state
        
        self.state_norms.append(norms)
        self.state_changes.append(changes)
        
        return {
            "layer_norms": norms,
            "layer_changes": changes,
            "mean_norm": sum(norms) / len(norms) if norms else 0.0,
            "mean_change": sum(changes) / len(changes) if changes else 0.0,
        }


def clip_grad_norm_with_logging(
    model: nn.Module,
    max_norm: float,
) -> Dict[str, float]:
    """
    Clip gradients and return statistics.
    
    Returns:
        Dict with grad_norm before clipping
    """
    # Compute total norm before clipping
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # Clip
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    return {
        "grad_norm_before_clip": total_norm,
        "was_clipped": total_norm > max_norm,
    }


def enable_gradient_checkpointing(model: nn.Module):
    """Enable gradient checkpointing for memory efficiency."""
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "model"):
        model.model.gradient_checkpointing = True
    else:
        raise AttributeError("Model does not support gradient checkpointing")


def count_tokens(
    batch: Dict[str, torch.Tensor],
    ignore_index: int = -100,
) -> int:
    """Count non-padding tokens in a batch."""
    if "labels" in batch:
        labels = batch["labels"]
        return (labels != ignore_index).sum().item()
    elif "input_ids" in batch:
        return batch["input_ids"].numel()
    return 0