import argparse
import logging
import timeit
import torch
import csv
import os
from cs336_basics.model import BasicsTransformerLM

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


def benchmark_model(
    model,
    batch_data,
    warmup_steps=10,
    timing_steps=100,
    backward_pass=True,
):
    """
    Benchmark forward and optionally backward passes of the model.
    
    Args:
        model: The model to benchmark
        batch_data: Input data for the model
        warmup_steps: Number of warmup steps before timing
        timing_steps: Number of steps to time
        backward_pass: Whether to include backward pass in timing
        
    Returns:
        Dictionary with timing results
    """
    model.train()
    
    # Warmup steps
    logger.info(f"Running {warmup_steps} warmup steps...")
    for _ in range(warmup_steps):
        if backward_pass:
            model.zero_grad(set_to_none=True)
            logits = model(batch_data)
            # Simple loss for backward pass
            loss = logits.mean()
            loss.backward()
        else:
            with torch.no_grad():
                logits = model(batch_data)
        torch.cuda.synchronize()
    
    # Timing steps
    logger.info(f"Timing {timing_steps} steps...")
    
    # Collect timing measurements
    forward_times = []
    backward_times = []
    total_times = []
    
    for _ in range(timing_steps):
        if backward_pass:
            model.zero_grad(set_to_none=True)
            
            # Time forward pass
            start_time = timeit.default_timer()
            logits = model(batch_data)
            loss = logits.mean()
            torch.cuda.synchronize()
            forward_end = timeit.default_timer()
            
            # Time backward pass
            loss.backward()
            torch.cuda.synchronize()
            backward_end = timeit.default_timer()
            
            forward_times.append(forward_end - start_time)
            backward_times.append(backward_end - forward_end)
            total_times.append(backward_end - start_time)
        else:
            # Forward only
            start_time = timeit.default_timer()
            with torch.inference_mode():
                _ = model(batch_data)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            
            forward_times.append(end_time - start_time)
            backward_times.append(0)
            total_times.append(end_time - start_time)
    
    # Calculate statistics using torch
    def calc_stats(times):
        if not times or all(t == 0 for t in times):
            return 0, 0, 0
        times_tensor = torch.tensor(times)
        std, mean = torch.std_mean(times_tensor)
        return mean.item(), std.item(), sum(times)
    
    total_mean, total_std, total_sum = calc_stats(total_times)
    forward_mean, forward_std, _ = calc_stats(forward_times)
    backward_mean, backward_std, _ = calc_stats(backward_times)
    
    return {
        "total_time": total_sum,
        "avg_time_per_step": total_mean,
        "std_time_per_step": total_std,
        "avg_forward_time": forward_mean,
        "std_forward_time": forward_std,
        "avg_backward_time": backward_mean,
        "std_backward_time": backward_std,
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
                     'forward_pass_time', 'backward_pass_time']
        
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
            'backward_pass_time': results['avg_backward_time']
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