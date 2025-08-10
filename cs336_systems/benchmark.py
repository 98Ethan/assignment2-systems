import argparse
import csv
import logging
import os
import timeit

import torch
import torch.cuda.nvtx as nvtx

import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

from cs336_systems.nvtx_profiling import annotated_scaled_dot_product_attention

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_model(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
    """Initialize a BasicsTransformerLM model with given hyperparameters."""
    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )


def generate_random_batch(batch_size, sequence_length, vocab_size, device):
    """Generate a random batch of token IDs."""
    return torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)


def _warmup(model, batch_data, steps, backward_pass):
    """Run warmup steps."""
    logger.info(f"Running {steps} warmup steps...")
    with nvtx.range("warm-up"):
        for _ in range(steps):
            model.zero_grad(set_to_none=True)
            logits = model(batch_data)
            if backward_pass:
                loss = logits.mean()
                loss.backward()
            torch.cuda.synchronize()

def _time_step(model, batch_data, backward_pass, optimizer=None):
    """Time a single forward/backward step."""
    model.zero_grad(set_to_none=True)
    
    # Forward pass
    with nvtx.range("forward"):
        start_time = timeit.default_timer()
        logits = model(batch_data)
        loss = logits.mean()
        torch.cuda.synchronize()
        forward_end = timeit.default_timer()
    
    if not backward_pass:
        return forward_end - start_time, 0, 0
    
    # Backward pass
    with nvtx.range("backward"):
        loss.backward()
        torch.cuda.synchronize()
        backward_end = timeit.default_timer()
    
    if optimizer:
        with nvtx.range("optimizer"):
            optimizer.step()
            torch.cuda.synchronize()
            optimizer_end = timeit.default_timer()
    
    return forward_end - start_time, backward_end - forward_end, optimizer_end - backward_end

def _calc_stats(times):
    """Calculate mean, std, and sum for timing data."""
    if not times or all(t == 0 for t in times):
        return 0, 0, 0
    times_tensor = torch.tensor(times)
    std, mean = torch.std_mean(times_tensor)
    return mean.item(), std.item(), sum(times)

def benchmark_model(model, batch_data, warmup_steps=10, timing_steps=100, backward_pass=True):
    """Benchmark forward and optionally backward passes of the model."""
    model.train()
    
    # Create optimizer if doing backward pass
    optimizer = AdamW(model.parameters(), lr=1e-4) if backward_pass else None
    
    # Warmup
    _warmup(model, batch_data, warmup_steps, backward_pass)
    
    # Timing
    logger.info(f"Timing {timing_steps} steps...")
    forward_times = []
    backward_times = []
    optimizer_times = []
    
    for _ in range(timing_steps):
        forward_time, backward_time, optimizer_time = _time_step(model, batch_data, backward_pass, optimizer)
        forward_times.append(forward_time)
        backward_times.append(backward_time)
        optimizer_times.append(optimizer_time)
    
    # Calculate statistics
    total_times = [f + b + o for f, b, o in zip(forward_times, backward_times, optimizer_times)]
    avg_total, std_total, sum_total = _calc_stats(total_times)
    
    return {
        "total_time": sum_total,
        "avg_time_per_step": avg_total,
        "std_time_per_step": std_total,
        "avg_forward_time": sum(forward_times) / len(forward_times),
        "avg_backward_time": sum(backward_times) / len(backward_times) if backward_pass else 0,
        "avg_optimizer_time": sum(optimizer_times) / len(optimizer_times) if optimizer else 0,
        "steps": timing_steps
    }


def log_to_csv(args, results, num_params):
    """Log benchmark results to CSV file."""
    csv_file = args.log_file
    
    # Check if file exists to write header
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        fieldnames = ['model_name', 'mode', 'd_model', 'num_layers', 'num_heads', 'd_ff', 
                     'batch_size', 'sequence_length', 'params_millions', 'warmup_steps', 
                     'timing_steps', 'total_time', 'avg_time_per_step', 'std_time_per_step',
                     'forward_pass_time', 'backward_pass_time', 'optimizer_time']
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if new file
        if not file_exists:
            writer.writeheader()
        
        # Write data
        writer.writerow({
            'model_name': args.model_name,
            'mode': 'inference' if args.forward_only else 'training',
            'd_model': args.d_model,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'd_ff': args.d_ff,
            'batch_size': args.batch_size,
            'sequence_length': args.sequence_length,
            'params_millions': num_params / 1e6,
            'warmup_steps': args.warmup_steps,
            'timing_steps': args.timing_steps,
            'total_time': results['total_time'],
            'avg_time_per_step': results['avg_time_per_step'],
            'std_time_per_step': results['std_time_per_step'],
            'forward_pass_time': results['avg_forward_time'],
            'backward_pass_time': results['avg_backward_time'],
            'optimizer_time': results['avg_optimizer_time']
        })
    
    logger.info(f"Results logged to {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM forward and backward passes")
    
    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=2048, help="Context length")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta value")
    
    # Benchmarking parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--sequence_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--timing_steps", type=int, default=100, help="Number of timing steps")
    parser.add_argument("--forward_only", action="store_true", help="Only benchmark forward pass")
    parser.add_argument("--log_file", type=str, default="benchmark_results.csv", help="CSV file to log results")
    parser.add_argument("--model_name", type=str, default="custom", help="Model name (e.g., small, medium, large)")
    
    args = parser.parse_args()
    
    # Swap the scaled_dot_product_attention function with annotated version
    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    
    # Set device to CUDA
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    # Initialize model
    logger.info("Initializing model...")
    model = create_model(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    model.to(device)
    
    logger.info(f"Model initialized with {model.get_num_params() / 1e6:.2f}M parameters")
    
    # Generate random batch
    logger.info("Generating random batch...")
    batch_data = generate_random_batch(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        vocab_size=args.vocab_size,
        device=device
    )
    
    logger.info(f"Batch shape: {batch_data.shape}")
    
    # Run benchmarks based on mode
    if args.forward_only:
        logger.info("\n" + "="*50)
        logger.info("BENCHMARKING FORWARD-ONLY PASS (INFERENCE)")
        logger.info("="*50)
        
        results = benchmark_model(
            model=model,
            batch_data=batch_data,
            warmup_steps=args.warmup_steps,
            timing_steps=args.timing_steps,
            backward_pass=False
        )
        
    else:
        logger.info("\n" + "="*50)
        logger.info("BENCHMARKING FORWARD + BACKWARD PASS (TRAINING)")
        logger.info("="*50)
        
        results = benchmark_model(
            model=model,
            batch_data=batch_data,
            warmup_steps=args.warmup_steps,
            timing_steps=args.timing_steps,
            backward_pass=True
        )
        
    logger.info(f"  Total time: {results['total_time']:.4f} seconds")
    logger.info(f"  Average time per step: {results['avg_time_per_step']:.4f} seconds")
    logger.info(f"  Standard deviation: {results['std_time_per_step']:.4f} seconds")
    logger.info(f"  Throughput: {1.0 / results['avg_time_per_step']:.2f} steps/second")
    
    # Log results to CSV
    log_to_csv(args, results, model.get_num_params())
    
    logger.info("\n" + "="*50)
    logger.info("BENCHMARK COMPLETED")
    logger.info("="*50)


if __name__ == "__main__":
    main()