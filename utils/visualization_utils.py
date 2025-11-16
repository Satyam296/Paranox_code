"""
Visualization Utilities for Space Station Challenge
===================================================
Functions for creating plots and visual reports.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

def plot_training_history(results_dir, output_path=None):
    """
    Plot training history from results.csv
    
    Args:
        results_dir: Path to training results directory
        output_path: Where to save the plot
    """
    results_dir = Path(results_dir)
    csv_path = results_dir / 'results.csv'
    
    if not csv_path.exists():
        print(f"Results file not found: {csv_path}")
        return
    
    # Read CSV
    import pandas as pd
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training History - Space Station Challenge', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss curves
    if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', linewidth=2)
        axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Box Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: mAP curves
    if 'metrics/mAP50(B)' in df.columns:
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2, color='green')
        if 'metrics/mAP50-95(B)' in df.columns:
            axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=2, color='blue')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].set_title('Mean Average Precision')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Precision & Recall
    if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
        axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2, color='purple')
        axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Class loss
    if 'train/cls_loss' in df.columns and 'val/cls_loss' in df.columns:
        axes[1, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', linewidth=2)
        axes[1, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Classification Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def create_performance_dashboard(metrics, class_names, output_path):
    """
    Create a comprehensive performance dashboard.
    
    Args:
        metrics: Dictionary of metrics
        class_names: List of class names
        output_path: Where to save the dashboard
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Overall metrics
    ax1 = fig.add_subplot(gs[0, :])
    overall_metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
    overall_values = [
        metrics.get('map50', 0),
        metrics.get('map', 0),
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics.get('f1', 0)
    ]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax1.barh(overall_metrics, overall_values, color=colors)
    ax1.set_xlim([0, 1])
    ax1.set_xlabel('Score', fontsize=12)
    ax1.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, overall_values)):
        ax1.text(value + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # Per-class mAP
    if 'class_metrics' in metrics:
        ax2 = fig.add_subplot(gs[1, :])
        class_map_values = [metrics['class_metrics'][name]['map50'] for name in class_names if name in metrics['class_metrics']]
        class_map_names = [name for name in class_names if name in metrics['class_metrics']]
        
        ax2.bar(range(len(class_map_names)), class_map_values, color='skyblue', edgecolor='navy')
        ax2.set_xticks(range(len(class_map_names)))
        ax2.set_xticklabels(class_map_names, rotation=45, ha='right')
        ax2.set_ylabel('mAP@0.5', fontsize=12)
        ax2.set_title('Per-Class Performance', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
    
    # Benchmark comparison
    ax3 = fig.add_subplot(gs[2, 0])
    benchmark_data = {
        'Baseline': 0.45,
        'Good': 0.60,
        'Excellent': 0.70,
        'Your Model': metrics.get('map50', 0)
    }
    colors_bench = ['#95a5a6', '#f39c12', '#2ecc71', '#e74c3c']
    ax3.bar(benchmark_data.keys(), benchmark_data.values(), color=colors_bench)
    ax3.set_ylabel('mAP@0.5', fontsize=10)
    ax3.set_title('Benchmark Comparison', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Precision-Recall comparison
    ax4 = fig.add_subplot(gs[2, 1])
    pr_data = ['Precision', 'Recall']
    pr_values = [metrics.get('precision', 0), metrics.get('recall', 0)]
    ax4.bar(pr_data, pr_values, color=['#9b59b6', '#e67e22'])
    ax4.set_ylabel('Score', fontsize=10)
    ax4.set_title('Precision vs Recall', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 1])
    ax4.grid(axis='y', alpha=0.3)
    
    # Performance category
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    map50 = metrics.get('map50', 0)
    if map50 >= 0.70:
        category = "EXCELLENT"
        color = '#2ecc71'
        emoji = "🌟"
    elif map50 >= 0.50:
        category = "GOOD"
        color = '#f39c12'
        emoji = "👍"
    elif map50 >= 0.40:
        category = "BASELINE"
        color = '#3498db'
        emoji = "✓"
    else:
        category = "NEEDS\nIMPROVEMENT"
        color = '#e74c3c'
        emoji = "⚠"
    
    ax5.text(0.5, 0.6, emoji, fontsize=60, ha='center', va='center')
    ax5.text(0.5, 0.3, category, fontsize=18, ha='center', va='center', 
            fontweight='bold', color=color)
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance dashboard saved to: {output_path}")
    plt.close()

def plot_class_performance_radar(metrics, class_names, output_path):
    """
    Create radar chart for per-class performance.
    
    Args:
        metrics: Dictionary with class_metrics
        class_names: List of class names
        output_path: Where to save the plot
    """
    if 'class_metrics' not in metrics:
        print("No per-class metrics available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(class_names), endpoint=False).tolist()
    angles += angles[:1]
    
    # Get mAP values for each class
    values = []
    for name in class_names:
        if name in metrics['class_metrics']:
            values.append(metrics['class_metrics'][name]['map50'])
        else:
            values.append(0)
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#3498db', label='mAP@0.5')
    ax.fill(angles, values, alpha=0.25, color='#3498db')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names, size=10)
    ax.set_ylim(0, 1)
    ax.set_title('Per-Class Performance (Radar Chart)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Radar chart saved to: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("Visualization utilities loaded")
    print("Available functions:")
    print("  - plot_training_history()")
    print("  - create_performance_dashboard()")
    print("  - plot_class_performance_radar()")
