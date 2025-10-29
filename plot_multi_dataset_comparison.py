#!/usr/bin/env python3
"""
Multi-Dataset Comparison Plots for Paper

Compares results across multiple dataset sizes (1k, 10k, 50k) to analyze:
- Scaling behavior
- Information retention trends
- Representation robustness across scales

Author: Claude
Date: 2025-10-29
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse

sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.4)

COLORS = {
    'all_layers': '#4A90E2',
    'visual_only': '#E24A4A',
    'drop': '#F5A623',
}

def load_results(results_json_path):
    """Load results from JSON file"""
    with open(results_json_path, 'r') as f:
        return json.load(f)

def extract_representation_name(full_name):
    """Extract clean representation name"""
    return full_name.replace('_all_layers', '').replace('_visual_only', '')

# ============================================================================
# 1. Multi-Dataset Information Retention Heatmap
# ============================================================================

def plot_multi_dataset_retention_heatmap(results_dict, output_dir):
    """
    2D heatmap: Dataset Size × Representation → % info retained
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get dataset names and sort by size
    dataset_names = sorted(results_dict.keys(),
                          key=lambda x: int(x.replace('k', '')) if x.replace('k', '').isdigit() else 0)

    # Get all representation names (assume same across datasets)
    first_dataset = list(results_dict.values())[0]
    rep_names = [extract_representation_name(r['representation'])
                 for r in first_dataset['all_layers_results']]

    # Build retention matrix
    retention_matrix = []
    for dataset_name in dataset_names:
        results = results_dict[dataset_name]
        dataset_retention = []

        for all_res, vis_res in zip(results['all_layers_results'], results['visual_only_results']):
            all_acc = all_res['test_acc_mean']
            vis_acc = vis_res['test_acc_mean']
            retention_pct = (vis_acc / all_acc * 100) if all_acc > 0 else 0
            dataset_retention.append(retention_pct)

        retention_matrix.append(dataset_retention)

    retention_array = np.array(retention_matrix)

    # Create heatmap
    im = ax.imshow(retention_array, cmap='RdYlGn', aspect='auto', vmin=50, vmax=100)

    # Set ticks
    ax.set_xticks(np.arange(len(rep_names)))
    ax.set_xticklabels(rep_names, rotation=45, ha='right', fontsize=11)
    ax.set_yticks(np.arange(len(dataset_names)))
    ax.set_yticklabels([f'{name} dataset' for name in dataset_names], fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Information Retained (%)', rotation=270, labelpad=25, fontweight='bold', fontsize=13)

    # Add text annotations
    for i in range(len(dataset_names)):
        for j in range(len(rep_names)):
            text = ax.text(j, i, f'{retention_array[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=10)

    ax.set_title('Information Retention Across Dataset Sizes',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Representation Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dataset Size', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/multi_dataset_retention_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/multi_dataset_retention_heatmap.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/multi_dataset_retention_heatmap.png")
    plt.close()

# ============================================================================
# 2. Scaling Behavior Curves
# ============================================================================

def plot_scaling_behavior(results_dict, output_dir):
    """
    Show how performance scales with dataset size for top representations
    """
    dataset_names = sorted(results_dict.keys(),
                          key=lambda x: int(x.replace('k', '')) if x.replace('k', '').isdigit() else 0)
    dataset_sizes = [int(name.replace('k', '')) for name in dataset_names]

    # Get representation names from first dataset
    first_dataset = list(results_dict.values())[0]
    all_rep_names = [extract_representation_name(r['representation'])
                     for r in first_dataset['all_layers_results']]

    # Select top 5 representations by average performance
    avg_perfs = []
    for i, rep_name in enumerate(all_rep_names):
        perfs = [results_dict[ds]['all_layers_results'][i]['test_acc_mean']
                for ds in dataset_names]
        avg_perfs.append(np.mean(perfs))

    top_5_indices = np.argsort(avg_perfs)[-5:]
    top_5_reps = [all_rep_names[i] for i in top_5_indices]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(top_5_reps)))

    # Plot each representation
    for idx, (rep_idx, rep_name, color) in enumerate(zip(top_5_indices, top_5_reps, colors)):
        ax = axes[idx]

        # Extract data for this representation
        all_layers_accs = []
        all_layers_stds = []
        visual_only_accs = []
        visual_only_stds = []

        for ds in dataset_names:
            results = results_dict[ds]
            all_layers_accs.append(results['all_layers_results'][rep_idx]['test_acc_mean'])
            all_layers_stds.append(results['all_layers_results'][rep_idx]['test_acc_std'])
            visual_only_accs.append(results['visual_only_results'][rep_idx]['test_acc_mean'])
            visual_only_stds.append(results['visual_only_results'][rep_idx]['test_acc_std'])

        # Plot with error bars
        ax.errorbar(dataset_sizes, all_layers_accs, yerr=all_layers_stds,
                   marker='o', markersize=10, linewidth=2.5, capsize=5,
                   label='All Layers', color=COLORS['all_layers'], alpha=0.8)

        ax.errorbar(dataset_sizes, visual_only_accs, yerr=visual_only_stds,
                   marker='s', markersize=10, linewidth=2.5, capsize=5,
                   label='Visual Only', color=COLORS['visual_only'], alpha=0.8)

        # Fill area between curves (information gap)
        ax.fill_between(dataset_sizes, all_layers_accs, visual_only_accs,
                       alpha=0.2, color=COLORS['drop'], label='Information Gap')

        ax.set_title(f'{rep_name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Dataset Size (thousands)', fontsize=11)
        ax.set_ylabel('Test Accuracy', fontsize=11)
        ax.legend(loc='lower right', frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, max(dataset_sizes) * 1.1)

    # Overall comparison in last panel
    ax = axes[-1]
    for rep_idx, rep_name, color in zip(top_5_indices, top_5_reps, colors):
        gaps = []
        for ds in dataset_names:
            results = results_dict[ds]
            all_acc = results['all_layers_results'][rep_idx]['test_acc_mean']
            vis_acc = results['visual_only_results'][rep_idx]['test_acc_mean']
            gaps.append(all_acc - vis_acc)

        ax.plot(dataset_sizes, gaps, marker='o', markersize=8, linewidth=2,
               label=rep_name, color=color, alpha=0.8)

    ax.set_title('Performance Gap Trends', fontsize=13, fontweight='bold')
    ax.set_xlabel('Dataset Size (thousands)', fontsize=11)
    ax.set_ylabel('Accuracy Drop (All → Visual)', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    fig.suptitle('Scaling Behavior: Performance vs Dataset Size (Top 5 Representations)',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/scaling_behavior_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/scaling_behavior_curves.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/scaling_behavior_curves.png")
    plt.close()

# ============================================================================
# 3. Dataset Size Sensitivity Comparison
# ============================================================================

def plot_dataset_sensitivity(results_dict, output_dir):
    """
    Compare which representations are most/least sensitive to dataset size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    dataset_names = sorted(results_dict.keys(),
                          key=lambda x: int(x.replace('k', '')) if x.replace('k', '').isdigit() else 0)
    dataset_sizes = [int(name.replace('k', '')) for name in dataset_names]

    # Get representation names
    first_dataset = list(results_dict.values())[0]
    rep_names = [extract_representation_name(r['representation'])
                 for r in first_dataset['all_layers_results']]

    # Calculate performance improvement from smallest to largest dataset
    improvements_all = []
    improvements_visual = []

    for i in range(len(rep_names)):
        # All layers
        perf_small = results_dict[dataset_names[0]]['all_layers_results'][i]['test_acc_mean']
        perf_large = results_dict[dataset_names[-1]]['all_layers_results'][i]['test_acc_mean']
        improvements_all.append(((perf_large - perf_small) / perf_small * 100) if perf_small > 0 else 0)

        # Visual only
        perf_small = results_dict[dataset_names[0]]['visual_only_results'][i]['test_acc_mean']
        perf_large = results_dict[dataset_names[-1]]['visual_only_results'][i]['test_acc_mean']
        improvements_visual.append(((perf_large - perf_small) / perf_small * 100) if perf_small > 0 else 0)

    # Sort by all_layers improvement
    sorted_indices = np.argsort(improvements_all)[::-1]
    rep_names_sorted = [rep_names[i] for i in sorted_indices]
    improvements_all_sorted = [improvements_all[i] for i in sorted_indices]
    improvements_visual_sorted = [improvements_visual[i] for i in sorted_indices]

    x = np.arange(len(rep_names_sorted))
    width = 0.35

    # Plot improvements
    bars1 = ax1.bar(x - width/2, improvements_all_sorted, width,
                    label='All Layers', color=COLORS['all_layers'], alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, improvements_visual_sorted, width,
                    label='Visual Only', color=COLORS['visual_only'], alpha=0.8, edgecolor='black')

    ax1.set_ylabel('Performance Improvement (%)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Representation Method', fontsize=13, fontweight='bold')
    ax1.set_title(f'Data Scaling Sensitivity\n({dataset_names[0]} → {dataset_names[-1]})',
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(rep_names_sorted, rotation=45, ha='right')
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8, fontweight='bold')

    # Scatter plot: dataset sensitivity comparison
    ax2.scatter(improvements_all_sorted, improvements_visual_sorted,
               s=200, alpha=0.7, edgecolors='black', linewidth=2, c=range(len(rep_names_sorted)),
               cmap='viridis')

    for i, name in enumerate(rep_names_sorted):
        ax2.annotate(name, (improvements_all_sorted[i], improvements_visual_sorted[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Add diagonal line
    min_val = min(min(improvements_all_sorted), min(improvements_visual_sorted))
    max_val = max(max(improvements_all_sorted), max(improvements_visual_sorted))
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal scaling')

    ax2.set_xlabel('All Layers Improvement (%)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Visual Only Improvement (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Data Scaling: All vs Visual-Only', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/dataset_sensitivity_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/dataset_sensitivity_comparison.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/dataset_sensitivity_comparison.png")
    plt.close()

# ============================================================================
# 4. Robustness Across Scales
# ============================================================================

def plot_robustness_across_scales(results_dict, output_dir):
    """
    Heatmap showing robustness (relative drop %) across dataset sizes
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    dataset_names = sorted(results_dict.keys(),
                          key=lambda x: int(x.replace('k', '')) if x.replace('k', '').isdigit() else 0)

    first_dataset = list(results_dict.values())[0]
    rep_names = [extract_representation_name(r['representation'])
                 for r in first_dataset['all_layers_results']]

    # Build robustness matrix (relative drops)
    robustness_matrix = []
    for dataset_name in dataset_names:
        results = results_dict[dataset_name]
        dataset_drops = []

        for all_res, vis_res in zip(results['all_layers_results'], results['visual_only_results']):
            all_acc = all_res['test_acc_mean']
            vis_acc = vis_res['test_acc_mean']
            drop = all_acc - vis_acc
            rel_drop = (drop / all_acc * 100) if all_acc > 0 else 0
            dataset_drops.append(rel_drop)

        robustness_matrix.append(dataset_drops)

    robustness_array = np.array(robustness_matrix)

    # Create heatmap (inverted colormap - lower is better)
    im = ax.imshow(robustness_array, cmap='RdYlGn_r', aspect='auto', vmin=20, vmax=50)

    ax.set_xticks(np.arange(len(rep_names)))
    ax.set_xticklabels(rep_names, rotation=45, ha='right', fontsize=11)
    ax.set_yticks(np.arange(len(dataset_names)))
    ax.set_yticklabels([f'{name} dataset' for name in dataset_names], fontsize=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance Drop (%)', rotation=270, labelpad=25, fontweight='bold', fontsize=13)

    # Add text annotations
    for i in range(len(dataset_names)):
        for j in range(len(rep_names)):
            color = 'white' if robustness_array[i, j] > 35 else 'black'
            text = ax.text(j, i, f'{robustness_array[i, j]:.1f}%',
                          ha="center", va="center", color=color, fontweight='bold', fontsize=10)

    ax.set_title('Representation Robustness Across Dataset Sizes\n(Lower is Better)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Representation Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dataset Size', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/robustness_across_scales.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/robustness_across_scales.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/robustness_across_scales.png")
    plt.close()

# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate multi-dataset comparison plots')
    parser.add_argument('--results-paths', nargs='+', required=True,
                       help='Paths to results.json files (e.g., results_1k.json results_10k.json results_50k.json)')
    parser.add_argument('--dataset-names', nargs='+', required=True,
                       help='Dataset names (e.g., 1k 10k 50k)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for plots')

    args = parser.parse_args()

    if len(args.results_paths) != len(args.dataset_names):
        raise ValueError("Number of results paths must match number of dataset names")

    # Load all results
    results_dict = {}
    for path, name in zip(args.results_paths, args.dataset_names):
        results_dict[name] = load_results(path)
        print(f"Loaded: {name} from {path}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("GENERATING MULTI-DATASET COMPARISON PLOTS")
    print("="*80)

    print("\n1. Multi-Dataset Information Retention Heatmap...")
    plot_multi_dataset_retention_heatmap(results_dict, args.output_dir)

    print("\n2. Scaling Behavior Curves...")
    plot_scaling_behavior(results_dict, args.output_dir)

    print("\n3. Dataset Size Sensitivity Comparison...")
    plot_dataset_sensitivity(results_dict, args.output_dir)

    print("\n4. Robustness Across Scales...")
    plot_robustness_across_scales(results_dict, args.output_dir)

    print("\n" + "="*80)
    print(f"✅ All multi-dataset comparison plots saved to: {args.output_dir}")
    print("="*80)
    print("\nGenerated files:")
    print("  - multi_dataset_retention_heatmap.png/pdf")
    print("  - scaling_behavior_curves.png/pdf")
    print("  - dataset_sensitivity_comparison.png/pdf")
    print("  - robustness_across_scales.png/pdf")

if __name__ == '__main__':
    main()
