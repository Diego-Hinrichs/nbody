#!/usr/bin/env python3
"""
# N-body Simulation Benchmark Analysis

This script loads benchmark results from a CSV file and generates
visualization charts to analyze performance.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from glob import glob

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

# Find benchmark CSV file
def find_csv_file(path=None):
    """
    # Find benchmark CSV file
    
    If path is provided, use that file. Otherwise, look for CSV files in the
    benchmark_results directories and select the most recent one.
    """
    if path and os.path.exists(path):
        return path
    
    # Find all benchmark directories
    benchmark_dirs = sorted(glob("benchmark_results_*"))
    if not benchmark_dirs:
        print("No benchmark results directories found.")
        return None
    
    # Use the most recent directory
    latest_dir = benchmark_dirs[-1]
    csv_path = os.path.join(latest_dir, "benchmark_results.csv")
    
    if os.path.exists(csv_path):
        return csv_path
    else:
        print(f"No CSV file found in {latest_dir}")
        return None

# Load benchmark data
def load_benchmark_data(csv_path):
    """
    # Load benchmark data from CSV
    
    Read the CSV file and filter for average results.
    """
    print(f"Loading benchmark data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Display basic info about the dataset
    print(f"Total records: {len(df)}")
    print(f"Methods: {df['method'].unique()}")
    print(f"Body counts: {df['body_count'].unique()}")
    
    # Filter for average results
    avg_results = df[df['run'] == 'average']
    return df, avg_results

# Chart 1: Execution Time vs Body Count (Line Chart)
def plot_execution_time(avg_results, output_dir):
    """
    # Chart 1: Execution Time vs Body Count
    
    Line chart showing execution time for each method across different body counts.
    """
    plt.figure(figsize=(12, 8))
    for method in METHODS:
        method_data = avg_results[avg_results['method'] == method]
        if not method_data.empty and 'total_time_ms' in method_data.columns:
            plt.plot(method_data['body_count'], method_data['total_time_ms'], 
                     marker='o', linestyle='-', label=method)
    
    plt.xlabel('Number of Bodies')
    plt.ylabel('Execution Time (ms)')
    plt.title('Performance Comparison: Execution Time vs. Body Count')
    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize='small')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.savefig(os.path.join(output_dir, 'execution_time.png'), dpi=300)
    plt.close()
    print(f"Created execution_time.png in {output_dir}")

# Chart 3: SFC vs Non-SFC Speedup
def plot_sfc_speedup(avg_results, output_dir):
    """
    # Chart 3: SFC vs Non-SFC Speedup
    
    Compare performance improvement of SFC methods over their non-SFC counterparts.
    """
    plt.figure(figsize=(12, 8))
    
    # Create a bar chart for better visualization
    categories = ['CPU Direct', 'GPU Direct', 'CPU Barnes-Hut', 'GPU Barnes-Hut']
    method_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]  # Non-SFC vs SFC pairs
    
    # For each body count, create a grouped bar chart
    body_counts = sorted(avg_results['body_count'].unique())
    
    # Choose a subset of body counts if there are too many
    if len(body_counts) > 5:
        body_counts = body_counts[::max(1, len(body_counts)//5)]
    
    # Set up the plot
    fig, axes = plt.subplots(len(body_counts), 1, figsize=(12, 4*len(body_counts)), sharex=True)
    if len(body_counts) == 1:
        axes = [axes]
    
    for i, body_count in enumerate(body_counts):
        speedups = []
        
        for base_method_idx, sfc_method_idx in method_pairs:
            base_method = METHODS[base_method_idx]
            sfc_method = METHODS[sfc_method_idx]
            
            base_data = avg_results[(avg_results['method'] == base_method) & 
                                    (avg_results['body_count'] == body_count)]
            sfc_data = avg_results[(avg_results['method'] == sfc_method) & 
                                   (avg_results['body_count'] == body_count)]
            
            if not base_data.empty and not sfc_data.empty and 'total_time_ms' in base_data.columns:
                base_time = base_data['total_time_ms'].iloc[0]
                sfc_time = sfc_data['total_time_ms'].iloc[0]
                
                # Calculate speedup: base_time / sfc_time (>1 means SFC is faster)
                speedup = base_time / sfc_time
                speedups.append(speedup)
            else:
                speedups.append(0)
        
        # Create bar chart for this body count
        ax = axes[i]
        bars = ax.bar(categories, speedups, color=['blue', 'green', 'orange', 'red'])
        
        # Add a horizontal line at y=1 to show the baseline
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Baseline (No Speedup)')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}x', ha='center', va='bottom')
        
        ax.set_title(f'SFC Speedup for {body_count} Bodies')
        ax.set_ylabel('Speedup Factor')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Color bars based on speedup (red if < 1, green if > 1)
        for j, bar in enumerate(bars):
            if bar.get_height() < 1:
                bar.set_color('lightcoral')
            else:
                bar.set_color('lightgreen')
    
    # Add overall title and labels
    fig.suptitle('SFC Performance Improvement\n(Values > 1 mean SFC is faster)', fontsize=16)
    axes[-1].set_xlabel('Method Category')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, 'sfc_speedup.png'), dpi=300)
    plt.close()
    
    # Also create a line chart for trend visualization
    plt.figure(figsize=(12, 8))
    
    for base_method_idx, sfc_method_idx in method_pairs:
        base_method = METHODS[base_method_idx]
        sfc_method = METHODS[sfc_method_idx]
        
        base_data = avg_results[avg_results['method'] == base_method]
        sfc_data = avg_results[avg_results['method'] == sfc_method]
        
        if not base_data.empty and not sfc_data.empty and 'total_time_ms' in base_data.columns:
            speedup = []
            bodies = []
            
            for body_count in sorted(avg_results['body_count'].unique()):
                base_time = base_data[base_data['body_count'] == body_count]['total_time_ms']
                sfc_time = sfc_data[sfc_data['body_count'] == body_count]['total_time_ms']
                
                if not base_time.empty and not sfc_time.empty:
                    # Calculate speedup: base_time / sfc_time (>1 means SFC is faster)
                    speedup.append(base_time.iloc[0] / sfc_time.iloc[0])
                    bodies.append(body_count)
            
            if speedup:
                method_name = base_method.replace("_DIRECT_SUM", "").replace("_BARNES_HUT", "")
                algorithm_type = "Direct Sum" if "DIRECT_SUM" in base_method else "Barnes-Hut"
                plt.plot(bodies, speedup, marker='o', linestyle='-', 
                         label=f"{method_name} {algorithm_type} with SFC")
    
    plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline (No Speedup)')
    plt.xlabel('Number of Bodies')
    plt.ylabel('Speedup Factor (>1 means SFC is faster)')
    plt.title('SFC Performance Improvement vs Baseline')
    plt.legend(loc='best')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    
    # Set logarithmic scales
    plt.xscale('log')
    plt.yscale('log')
    
    # Add reference line
    plt.axhline(y=1.0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Add colored regions
    ymin, ymax = plt.ylim()
    plt.fill_between([min(bodies), max(bodies)], [1, 1], [ymin, ymin], 
                     alpha=0.1, color='red', label='_SFC Slower')
    plt.fill_between([min(bodies), max(bodies)], [1, 1], [ymax, ymax], 
                     alpha=0.1, color='green', label='_SFC Faster')
    
    # Adjust y-axis limits to make the baseline more centered
    plt.ylim(ymin / 1.5, ymax * 1.5)
    
    # Save with high DPI
    plt.savefig(os.path.join(output_dir, 'sfc_speedup_trend.png'), dpi=300)
    plt.close()
    
    print(f"Created sfc_speedup.png and sfc_speedup_trend.png in {output_dir}")

# Chart 4: CPU vs GPU Speedup
def plot_cpu_vs_gpu(avg_results, output_dir):
    """
    # Chart 4: CPU vs GPU Speedup
    
    Compare performance improvement of GPU methods over their CPU counterparts.
    """
    plt.figure(figsize=(12, 8))
    
    for method_pair in [(0, 2), (1, 3), (4, 6), (5, 7)]:  # CPU vs GPU pairs
        cpu_method = METHODS[method_pair[0]]
        gpu_method = METHODS[method_pair[1]]
        
        cpu_data = avg_results[avg_results['method'] == cpu_method]
        gpu_data = avg_results[avg_results['method'] == gpu_method]
        
        if not cpu_data.empty and not gpu_data.empty and 'total_time_ms' in cpu_data.columns:
            speedup = []
            bodies = []
            
            for body_count in sorted(avg_results['body_count'].unique()):
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
    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99))
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'gpu_vs_cpu.png'), dpi=300)
    plt.close()
    print(f"Created gpu_vs_cpu.png in {output_dir}")

# Chart 5: Barnes-Hut vs Direct Sum Speedup
def plot_barnes_vs_direct(avg_results, output_dir):
    """
    # Chart 5: Barnes-Hut vs Direct Sum Speedup
    
    Compare performance improvement of Barnes-Hut algorithms over Direct Sum.
    """
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
            
            for body_count in sorted(avg_results['body_count'].unique()):
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
    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99))
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'barnes_vs_direct.png'), dpi=300)
    plt.close()
    print(f"Created barnes_vs_direct.png in {output_dir}")

# Chart 6: All Methods Speedup vs CPU Direct Sum Baseline
def plot_all_speedups(avg_results, output_dir):
    """
    # Chart 6: All Methods Speedup vs CPU Direct Sum Baseline
    
    Compare performance improvement of all methods over CPU Direct Sum (baseline).
    """
    plt.figure(figsize=(12, 8))
    
    baseline_method = METHODS[0]  # CPU_DIRECT_SUM
    baseline_data = avg_results[avg_results['method'] == baseline_method]
    
    if baseline_data.empty or 'total_time_ms' not in baseline_data.columns:
        print(f"Warning: Baseline method {baseline_method} data not found or incomplete.")
        return
    
    # Create a line for baseline (speedup = 1.0)
    bodies = sorted(baseline_data['body_count'].unique())
    plt.plot(bodies, [1.0] * len(bodies), 'k--', linewidth=1.5, label=f"{baseline_method} (Baseline)")
    
    # Plot speedup for each method
    for method_idx, method in enumerate(METHODS):
        if method == baseline_method:
            continue  # Skip baseline method
        
        method_data = avg_results[avg_results['method'] == method]
        
        if not method_data.empty and 'total_time_ms' in method_data.columns:
            speedup = []
            method_bodies = []
            
            for body_count in bodies:
                baseline_time = baseline_data[baseline_data['body_count'] == body_count]['total_time_ms']
                method_time = method_data[method_data['body_count'] == body_count]['total_time_ms']
                
                if not baseline_time.empty and not method_time.empty:
                    # Calculate speedup: baseline_time / method_time (>1 means method is faster than baseline)
                    speedup.append(baseline_time.iloc[0] / method_time.iloc[0])
                    method_bodies.append(body_count)
            
            if speedup:
                plt.plot(method_bodies, speedup, marker='o', linestyle='-', label=method)
    
    plt.xlabel('Number of Bodies')
    plt.ylabel('Speedup Factor (>1 means faster than baseline)')
    plt.title('Performance Comparison: All Methods vs CPU Direct Sum Baseline')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    
    # Set logarithmic scales
    plt.xscale('log')
    plt.yscale('log')
    
    # Add colored regions
    ymin, ymax = plt.ylim()
    plt.fill_between([min(bodies), max(bodies)], [1, 1], [ymin, ymin], 
                     alpha=0.1, color='red', label='_Slower than baseline')
    plt.fill_between([min(bodies), max(bodies)], [1, 1], [ymax, ymax], 
                     alpha=0.1, color='green', label='_Faster than baseline')
    
    # Adjust y-axis limits to make the baseline more centered
    plt.ylim(ymin / 1.5, ymax * 1.5)
    
    plt.savefig(os.path.join(output_dir, 'speedups.png'), dpi=300)
    plt.close()
    print(f"Created all_speedups.png in {output_dir}")

# Main function
def main():
    """
    # Main function
    
    Process command line arguments and generate the speedup chart.
    """
    # Create output directory for charts
    output_dir = "benchmark_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get CSV file path from command line or find it automatically
    csv_path = None
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = find_csv_file()
    
    if not csv_path:
        print("Error: No benchmark CSV file found.")
        sys.exit(1)
    
    # Load data
    df, avg_results = load_benchmark_data(csv_path)
    
    # Generate only the speedup chart
    print("\n# Generating speedup chart...")
    plot_all_speedups(avg_results, output_dir)
    
    print(f"\n# Analysis complete. Chart saved to {output_dir}/all_speedups.png")

if __name__ == "__main__":
    print("# N-body Simulation Benchmark Analysis")
    print("======================================")
    main() 