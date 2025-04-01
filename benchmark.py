#!/usr/bin/env python3
import subprocess
import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Configuration
METHODS = [
    "CPU_DIRECT_SUM",        # 0
    "CPU_SFC_DIRECT_SUM",    # 1
    "GPU_DIRECT_SUM",        # 2
    "GPU_SFC_DIRECT_SUM",    # 3
    "CPU_BARNES_HUT",        # 4
    "CPU_SFC_BARNES_HUT",    # 5
    "GPU_BARNES_HUT",        # 6
    "GPU_SFC_BARNES_HUT"     # 7
]

BODY_COUNTS = [1000, 5000, 10000, 50000]
ITERATIONS = 10  # Number of simulation steps to run
REPEATS = 5       # Number of times to repeat each test for reliability

# Ensure build exists
def ensure_build():
    if not os.path.exists("build/BarnesHutSimulation"):
        print("Building project...")
        
        # Create build directory if it doesn't exist
        if not os.path.exists("build"):
            os.makedirs("build", exist_ok=True)
            
        # Run cmake
        print("Running CMake...")
        try:
            cmake_result = subprocess.run(["cmake", "-B", "build", "-S", "."], 
                                        check=False, 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True)
            
            if cmake_result.returncode != 0:
                print("CMake configuration failed:")
                print(cmake_result.stderr)
                raise Exception("CMake configuration failed")
                
            # Run make
            print("Running make...")
            make_result = subprocess.run(["make", "-C", "build", "-j4"], 
                                      check=False, 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE,
                                      universal_newlines=True)
            
            if make_result.returncode != 0:
                print("Make build failed:")
                print(make_result.stderr)
                raise Exception("Make build failed")
                
        except Exception as e:
            print(f"Build error: {str(e)}")
            raise Exception("Failed to build the project")
    
    if not os.path.exists("build/BarnesHutSimulation"):
        raise Exception("Failed to build the project - executable not found")

# Run a benchmark for a specific method and body count
def run_benchmark(method_idx, body_count, iterations):
    cmd = [
        "./build/BarnesHutSimulation",
        "--headless",
        "--method", str(method_idx),
        "--bodies", str(body_count),
        "--iterations", str(iterations),
        "--report-metrics"
    ]
    
    # For SFC methods, enable SFC
    if method_idx in [1, 3, 5, 7]:
        cmd.extend(["--use-sfc", "true"])
    
    print(f"    Running command: {' '.join(cmd)}")
    
    # Run the process and capture output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error running benchmark: {stderr}")
        return None
    
    # Parse metrics from output
    metrics = {}
    for line in stdout.split('\n'):
        # Skip log messages and empty lines
        if line.strip() == "" or line.startswith("[INFO]") or line.startswith("[ERROR]"):
            continue
            
        if ":" in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key, value = parts
                key = key.strip()
                value = value.strip()
                try:
                    value = float(value)
                    metrics[key] = value
                except ValueError:
                    # Only include non-numeric values if they're not log messages
                    if not any(prefix in key for prefix in ["[INFO]", "[ERROR]"]):
                        metrics[key] = value
    
    return metrics

# Main benchmark function
def benchmark_all():
    ensure_build()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"benchmark_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    
    for method_idx, method_name in enumerate(METHODS):
        print(f"Benchmarking {method_name}...")
        
        for body_count in BODY_COUNTS:
            print(f"  Testing with {body_count} bodies...")
            
            method_results = []
            for repeat in range(REPEATS):
                print(f"    Run {repeat+1}/{REPEATS}")
                metrics = run_benchmark(method_idx, body_count, ITERATIONS)
                
                if metrics:
                    metrics["method"] = method_name
                    metrics["body_count"] = body_count
                    metrics["run"] = repeat
                    results.append(metrics)
                    method_results.append(metrics)
            
            # Calculate average metrics for this configuration
            if method_results:
                avg_metrics = {
                    "method": method_name,
                    "body_count": body_count,
                    "run": "average"
                }
                
                for key in method_results[0].keys():
                    if key not in ["method", "body_count", "run"] and isinstance(method_results[0][key], (int, float)):
                        avg_metrics[key] = sum(r[key] for r in method_results) / len(method_results)
                
                results.append(avg_metrics)
    
    # Save results to CSV
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        if results:
            # Collect all unique keys from all results
            all_fields = set()
            for result in results:
                all_fields.update(result.keys())
            fieldnames = list(all_fields)
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    print(f"Results saved to {csv_path}")
    
    # Generate charts
    generate_charts(results_dir, results)
    
    return results

# Generate visualization charts
def generate_charts(results_dir, results):
    df = pd.DataFrame(results)
    avg_results = df[df['run'] == 'average']
    
    # Performance comparison: Execution time vs Body Count
    plt.figure(figsize=(12, 8))
    for method in METHODS:
        method_data = avg_results[avg_results['method'] == method]
        if not method_data.empty and 'total_time_ms' in method_data.columns:
            plt.plot(method_data['body_count'], method_data['total_time_ms'], 
                     marker='o', linestyle='-', label=method)
    
    plt.xlabel('Number of Bodies')
    plt.ylabel('Execution Time (ms)')
    plt.title('Performance Comparison: Execution Time vs. Body Count')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(results_dir, 'execution_time.png'), dpi=300)
    
    # NEW CHART: Bar chart comparing mean execution time of all methods
    plt.figure(figsize=(15, 10))
    
    # For each body count, create a grouped bar chart
    for i, body_count in enumerate(BODY_COUNTS):
        plt.subplot(1, len(BODY_COUNTS), i+1)
        
        # Get data for this body count
        body_data = avg_results[avg_results['body_count'] == body_count]
        
        # Extract methods and their times
        methods = []
        times = []
        
        for method in METHODS:
            method_data = body_data[body_data['method'] == method]
            if not method_data.empty and 'total_time_ms' in method_data.columns:
                methods.append(method)
                times.append(method_data['total_time_ms'].iloc[0])
        
        # Create bar chart
        bar_positions = np.arange(len(methods))
        bars = plt.bar(bar_positions, times)
        
        # Add method names and values on top of bars
        plt.xticks(bar_positions, [m.replace('_', '\n') for m in methods], rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.1f}',
                    ha='center', va='bottom', rotation=0)
        
        plt.title(f'Mean Execution Time for {body_count} Bodies')
        plt.ylabel('Time (ms)')
        plt.tight_layout()
    
    plt.savefig(os.path.join(results_dir, 'mean_execution_time_comparison.png'), dpi=300)
    
    # SFC vs Non-SFC comparison
    plt.figure(figsize=(12, 8))
    for base_method_idx in [0, 2, 4, 6]:  # Non-SFC methods
        base_method = METHODS[base_method_idx]
        sfc_method = METHODS[base_method_idx + 1]
        
        base_data = avg_results[avg_results['method'] == base_method]
        sfc_data = avg_results[avg_results['method'] == sfc_method]
        
        if not base_data.empty and not sfc_data.empty and 'total_time_ms' in base_data.columns:
            speedup = []
            bodies = []
            
            for body_count in BODY_COUNTS:
                base_time = base_data[base_data['body_count'] == body_count]['total_time_ms']
                sfc_time = sfc_data[sfc_data['body_count'] == body_count]['total_time_ms']
                
                if not base_time.empty and not sfc_time.empty:
                    # Calculate speedup: base_time / sfc_time (>1 means SFC is faster)
                    speedup.append(base_time.iloc[0] / sfc_time.iloc[0])
                    bodies.append(body_count)
            
            if speedup:
                plt.plot(bodies, speedup, marker='o', linestyle='-', 
                         label=f"{base_method} vs {sfc_method}")
    
    plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline (No Speedup)')
    plt.xlabel('Number of Bodies')
    plt.ylabel('Speedup Factor (>1 means SFC is faster)')
    plt.title('SFC Performance Improvement')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'sfc_speedup.png'), dpi=300)
    
    # CPU vs GPU comparison
    plt.figure(figsize=(12, 8))
    for method_pair in [(0, 2), (1, 3), (4, 6), (5, 7)]:  # CPU vs GPU pairs
        cpu_method = METHODS[method_pair[0]]
        gpu_method = METHODS[method_pair[1]]
        
        cpu_data = avg_results[avg_results['method'] == cpu_method]
        gpu_data = avg_results[avg_results['method'] == gpu_method]
        
        if not cpu_data.empty and not gpu_data.empty and 'total_time_ms' in cpu_data.columns:
            speedup = []
            bodies = []
            
            for body_count in BODY_COUNTS:
                cpu_time = cpu_data[cpu_data['body_count'] == body_count]['total_time_ms']
                gpu_time = gpu_data[gpu_data['body_count'] == body_count]['total_time_ms']
                
                if not cpu_time.empty and not gpu_time.empty:
                    # Calculate speedup: cpu_time / gpu_time (>1 means GPU is faster)
                    speedup.append(cpu_time.iloc[0] / gpu_time.iloc[0])
                    bodies.append(body_count)
            
            if speedup:
                plt.plot(bodies, speedup, marker='o', linestyle='-', 
                         label=f"{cpu_method} vs {gpu_method}")
    
    plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline (No Speedup)')
    plt.xlabel('Number of Bodies')
    plt.ylabel('Speedup Factor (>1 means GPU is faster)')
    plt.title('GPU vs CPU Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'gpu_vs_cpu.png'), dpi=300)
    
    # Barnes-Hut vs Direct Sum comparison
    plt.figure(figsize=(12, 8))
    for category in ['CPU', 'CPU_SFC', 'GPU', 'GPU_SFC']:
        if category == 'CPU':
            direct_method = METHODS[0]
            barnes_method = METHODS[4]
        elif category == 'CPU_SFC':
            direct_method = METHODS[1]
            barnes_method = METHODS[5]
        elif category == 'GPU':
            direct_method = METHODS[2]
            barnes_method = METHODS[6]
        else:  # GPU_SFC
            direct_method = METHODS[3]
            barnes_method = METHODS[7]
        
        direct_data = avg_results[avg_results['method'] == direct_method]
        barnes_data = avg_results[avg_results['method'] == barnes_method]
        
        if not direct_data.empty and not barnes_data.empty and 'total_time_ms' in direct_data.columns:
            speedup = []
            bodies = []
            
            for body_count in BODY_COUNTS:
                direct_time = direct_data[direct_data['body_count'] == body_count]['total_time_ms']
                barnes_time = barnes_data[barnes_data['body_count'] == body_count]['total_time_ms']
                
                if not direct_time.empty and not barnes_time.empty:
                    # Calculate speedup: direct_time / barnes_time (>1 means Barnes-Hut is faster)
                    speedup.append(direct_time.iloc[0] / barnes_time.iloc[0])
                    bodies.append(body_count)
            
            if speedup:
                plt.plot(bodies, speedup, marker='o', linestyle='-', 
                         label=f"{category}: Direct Sum vs Barnes-Hut")
    
    plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline (No Speedup)')
    plt.xlabel('Number of Bodies')
    plt.ylabel('Speedup Factor (>1 means Barnes-Hut is faster)')
    plt.title('Barnes-Hut vs Direct Sum Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'barnes_vs_direct.png'), dpi=300)
    
    print(f"Charts generated in {results_dir}")

if __name__ == "__main__":
    print("Starting N-body simulation benchmark...")
    benchmark_all()
    print("Benchmark complete!") 