"""
Example usage of Depth-Gated DeltaNet.

This script demonstrates:
1. Model creation and configuration
2. Forward pass with depth state caching
3. Autoregressive generation
4. Training loop setup
5. Model saving and loading
"""

import torch
from modeling_depth_deltanet import (
    DepthDeltaNetConfig,
    DepthDeltaNetForCausalLM,
    DepthDeltaNetCache,
)
from auto_registration import register_auto_classes, get_config_for_size
from generation_utils import (
    generate,
    GenerationConfig,
    print_model_size,
)
from training_utils import (
    TrainingConfig,
    create_optimizer_and_scheduler,
    GradientMonitor,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def example_basic_usage():
    """Basic model creation and forward pass."""
    print("=" * 60)
    print("Example 1: Basic Model Usage")
    print("=" * 60)
    
    
    # Create configuration
    config = DepthDeltaNetConfig(
        vocab_size=32000,
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=16,
        head_dim=64,
        # State bank configuration
        state_bank_num_heads=8,
        state_bank_head_dim=64,
        state_bank_expand_v=2.0,
        state_bank_use_short_conv=True,
        depth_init=True,  # Crucial for layer-to-layer state persistence
    )
    
    print(f"Config: {config}")
    
    # Create model
    model = DepthDeltaNetForCausalLM(config).to(DEVICE)
    print_model_size(model)
    
    # Example forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(DEVICE)
    
    # Forward pass (training mode)
    outputs = model(input_ids=input_ids, labels=input_ids)
    print(f"\nForward pass:")
    print(f"  Loss: {outputs.loss.item():.4f}")
    print(f"  Logits shape: {outputs.logits.shape}")
    
    return model, config


def example_generation():
    """Autoregressive generation with depth state caching."""
    print("\n" + "=" * 60)
    print("Example 2: Text Generation with Depth Caching")
    print("=" * 60)
    
    # Use small model for demo
    config = get_config_for_size("tiny")
    model = DepthDeltaNetForCausalLM(config).to(DEVICE)
    model.eval()
    
    # Prompt
    prompt_ids = torch.randint(0, config.vocab_size, (1, 10)).to(DEVICE)
    
    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        use_cache=True,  # Enable depth state caching
    )
    
    # Generate
    print("Generating with depth state caching...")
    generated = generate(model, prompt_ids, gen_config)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")


def example_caching_mechanism():
    """Demonstrate how depth state caching works."""
    print("\n" + "=" * 60)
    print("Example 3: Understanding Depth State Caching")
    print("=" * 60)
    
    config = get_config_for_size("tiny")
    model = DepthDeltaNetForCausalLM(config).to(DEVICE)
    model.eval()
    
    # Initial prompt
    prompt_ids = torch.randint(0, config.vocab_size, (1, 20)).to(DEVICE)
    
    # First forward pass - processes full prompt
    print("First forward pass (full prompt)...")
    outputs1 = model(input_ids=prompt_ids, use_cache=True)
    cache = outputs1.past_key_values
    
    print(f"  Cache type: {type(cache).__name__}")
    print(f"  Number of layers with KV cache: {len(cache.key_cache)}")
    print(f"  Number of layers with depth state: {len(cache.depth_states)}")
    
    # Check depth states
    for i, state in enumerate(cache.depth_states[:3]):
        if state is not None:
            print(f"  Layer {i} depth state shape: {state.shape}")
    
    # Second forward pass - processes single new token using cache
    next_token = torch.randint(0, config.vocab_size, (1, 1)).to(DEVICE)
    print("\nSecond forward pass (single token with cache)...")
    outputs2 = model(input_ids=next_token, past_key_values=cache, use_cache=True)
    
    print(f"  Input: 1 token")
    print(f"  Output logits shape: {outputs2.logits.shape}")
    print(f"  Cache seq length now: {outputs2.past_key_values.get_seq_length()}")


def example_training_setup():
    """Setup for training with gradient monitoring."""
    print("\n" + "=" * 60)
    print("Example 4: Training Setup")
    print("=" * 60)
    
    # Configuration
    config = get_config_for_size("tiny")
    model = DepthDeltaNetForCausalLM(config).to(DEVICE)
    
    # Training config with special handling for depth-recurrent params
    train_config = TrainingConfig(
        learning_rate=1e-4,
        weight_decay=0.1,
        warmup_steps=100,
        num_training_steps=10000,
        lr_scheduler_type="cosine",
        # State bank specific settings
        state_bank_lr_multiplier=1.0,
        decay_param_lr_multiplier=0.1,  # Lower LR for A_log, dt_bias
        max_grad_norm=1.0,
    )
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, train_config)
    
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Number of parameter groups: {len(optimizer.param_groups)}")
    for i, group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in group['params'])
        print(f"  Group {i}: {num_params:,} params, lr={group['lr']:.2e}, wd={group['weight_decay']}")
    
    # Gradient monitor
    grad_monitor = GradientMonitor(model, log_freq=1)
    
    # Mini training loop demo
    print("\nMini training loop (3 steps):")
    model.train()
    
    for step in range(3):
        # Dummy batch
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(DEVICE)
        
        # Forward
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        
        # Backward
        loss.backward()
        
        # Monitor gradients
        grad_stats = grad_monitor.log_gradients()
        
        # Clip and step
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        print(f"  Step {step}: loss={loss.item():.4f}, grad_norm={grad_stats.get('grad_norm', 0):.4f}")


def example_save_load():
    """Saving and loading models."""
    print("\n" + "=" * 60)
    print("Example 5: Model Saving and Loading")
    print("=" * 60)
    
    # Create and configure model
    config = get_config_for_size("tiny")
    model = DepthDeltaNetForCausalLM(config).to(DEVICE)
    
    # Save
    save_path = "/tmp/depth_deltanet_demo"
    print(f"Saving model to {save_path}...")
    model.save_pretrained(save_path)
    config.save_pretrained(save_path)
    
    # Load
    print("Loading model back...")
    loaded_config = DepthDeltaNetConfig.from_pretrained(save_path)
    loaded_model = DepthDeltaNetForCausalLM.from_pretrained(save_path)
    
    print("Model loaded successfully!")
    print(f"  Config hidden_size: {loaded_config.hidden_size}")
    print(f"  Config num_layers: {loaded_config.num_hidden_layers}")


def example_depth_state_analysis():
    """Analyze how depth state evolves across layers."""
    print("\n" + "=" * 60)
    print("Example 6: Depth State Analysis")
    print("=" * 60)
    
    config = get_config_for_size("tiny")
    model = DepthDeltaNetForCausalLM(config).to(DEVICE)
    model.eval()
    
    # Hook to capture states
    states_by_layer = []
    
    def capture_state_hook(module, input, output):
        # output is (hidden_states, ..., depth_state, ...)
        if len(output) > 3:  # Has depth state
            depth_state = output[-2]
            if depth_state is not None:
                states_by_layer.append(depth_state.detach().clone())
    
    # Register hooks
    for layer in model.model.layers:
        layer.state_bank.register_forward_hook(capture_state_hook)
    
    # Forward pass
    input_ids = torch.randint(0, config.vocab_size, (1, 64)).to(DEVICE)
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=True)
    
    # Analyze states
    print(f"Captured {len(states_by_layer)} state matrices")
    
    for i, state in enumerate(states_by_layer[:5]):
        norm = state.norm().item()
        print(f"  Layer {i}: state norm = {norm:.4f}, shape = {state.shape}")
    
    # State evolution
    if len(states_by_layer) > 1:
        print("\nState evolution (change between layers):")
        for i in range(1, min(5, len(states_by_layer))):
            change = (states_by_layer[i] - states_by_layer[i-1]).norm().item()
            print(f"  Layer {i-1} -> {i}: change = {change:.4f}")


def main():
    """Run all examples."""
    print("Depth-Gated DeltaNet Examples")
    print("=" * 60)
    
    # Register with HuggingFace Auto classes
    register_auto_classes()
    
    # Run examples
    example_basic_usage()
    example_generation()
    example_caching_mechanism()
    example_training_setup()
    example_save_load()
    example_depth_state_analysis()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()