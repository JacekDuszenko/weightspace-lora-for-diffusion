import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse

sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.3)

def load_results(results_json_path):
    with open(results_json_path, 'r') as f:
        return json.load(f)

def plot_performance_comparison(results, output_dir):
    """Bar plot comparing all layers vs visual-only across representations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    representations = [r['representation'].replace('_all_layers', '').replace('_visual_only', '') 
                      for r in results['all_layers_results']]
    
    all_layers_test_acc = [r['test_acc_mean'] for r in results['all_layers_results']]
    all_layers_test_std = [r['test_acc_std'] for r in results['all_layers_results']]
    visual_only_test_acc = [r['test_acc_mean'] for r in results['visual_only_results']]
    visual_only_test_std = [r['test_acc_std'] for r in results['visual_only_results']]
    
    x = np.arange(len(representations))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, all_layers_test_acc, width, yerr=all_layers_test_std,
                   label='All Layers (attn1+attn2)', capsize=5, alpha=0.8, color='skyblue')
    axes[0, 0].bar(x + width/2, visual_only_test_acc, width, yerr=visual_only_test_std,
                   label='Visual Only (attn1)', capsize=5, alpha=0.8, color='lightcoral')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('Test Accuracy: All Layers vs Visual-Only')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(representations, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    all_layers_test_f1 = [r['test_f1_mean'] for r in results['all_layers_results']]
    all_layers_f1_std = [r['test_f1_std'] for r in results['all_layers_results']]
    visual_only_test_f1 = [r['test_f1_mean'] for r in results['visual_only_results']]
    visual_only_f1_std = [r['test_f1_std'] for r in results['visual_only_results']]
    
    axes[0, 1].bar(x - width/2, all_layers_test_f1, width, yerr=all_layers_f1_std,
                   label='All Layers (attn1+attn2)', capsize=5, alpha=0.8, color='skyblue')
    axes[0, 1].bar(x + width/2, visual_only_test_f1, width, yerr=visual_only_f1_std,
                   label='Visual Only (attn1)', capsize=5, alpha=0.8, color='lightcoral')
    axes[0, 1].set_ylabel('Test F1 Score')
    axes[0, 1].set_title('Test F1: All Layers vs Visual-Only')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(representations, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    performance_drop = [(all_layers_test_acc[i] - visual_only_test_acc[i]) 
                        for i in range(len(representations))]
    
    colors = ['green' if d < 0 else 'red' for d in performance_drop]
    axes[1, 0].bar(x, performance_drop, color=colors, alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_ylabel('Accuracy Drop')
    axes[1, 0].set_title('Performance Drop (All Layers → Visual-Only)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(representations, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    all_layers_time = [r.get('training_time_mean', 0) for r in results['all_layers_results']]
    visual_only_time = [r.get('training_time_mean', 0) for r in results['visual_only_results']]
    
    axes[1, 1].bar(x - width/2, all_layers_time, width,
                   label='All Layers (attn1+attn2)', alpha=0.8, color='skyblue')
    axes[1, 1].bar(x + width/2, visual_only_time, width,
                   label='Visual Only (attn1)', alpha=0.8, color='lightcoral')
    axes[1, 1].set_ylabel('Training Time (seconds)')
    axes[1, 1].set_title('Training Time Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(representations, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/performance_comparison.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/performance_comparison.png")
    plt.close()

def plot_learning_curves(results, output_dir):
    """Plot learning curves for best performing representations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (result_set, title_prefix) in enumerate([
        (results['all_layers_results'], 'All Layers'),
        (results['visual_only_results'], 'Visual Only')
    ]):
        best_3 = sorted(result_set, key=lambda x: x['test_acc_mean'], reverse=True)[:3]
        
        for i, res in enumerate(best_3):
            ax = axes[idx * 3 + i]
            
            if 'learning_curves' in res and res['learning_curves']:
                for run_idx, curve in enumerate(res['learning_curves']):
                    val_acc = curve['val_acc_history']
                    epochs = list(range(1, len(val_acc) + 1))
                    ax.plot(epochs, val_acc, alpha=0.6, label=f'Run {run_idx+1}')
                    ax.axvline(x=curve['best_epoch'], color='red', linestyle='--', alpha=0.3)
                
                rep_name = res['representation'].replace('_all_layers', '').replace('_visual_only', '')
                ax.set_title(f'{title_prefix}: {rep_name}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Validation Accuracy')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/learning_curves.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/learning_curves.png")
    plt.close()

def plot_confusion_matrices(results, output_dir):
    """Plot confusion matrices for best representation"""
    best_all = max(results['all_layers_results'], key=lambda x: x['test_acc_mean'])
    best_visual = max(results['visual_only_results'], key=lambda x: x['test_acc_mean'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (res, title) in enumerate([
        (best_all, 'All Layers (Best)'),
        (best_visual, 'Visual Only (Best)')
    ]):
        if 'test_confusion_matrices' in res and res['test_confusion_matrices']:
            avg_cm = np.mean(res['test_confusion_matrices'], axis=0)
            
            sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'})
            rep_name = res['representation'].replace('_all_layers', '').replace('_visual_only', '')
            axes[idx].set_title(f'{title}\n{rep_name}\nAcc: {res["test_acc_mean"]:.3f}±{res["test_acc_std"]:.3f}')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/confusion_matrices.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/confusion_matrices.png")
    plt.close()

def plot_train_val_gap(results, output_dir):
    """Plot train-validation accuracy gap (overfitting analysis)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (result_set, title) in enumerate([
        (results['all_layers_results'], 'All Layers (attn1+attn2)'),
        (results['visual_only_results'], 'Visual Only (attn1)')
    ]):
        representations = [r['representation'].replace('_all_layers', '').replace('_visual_only', '') 
                          for r in result_set]
        train_accs = [r.get('train_acc_mean', 0) for r in result_set]
        val_accs = [r['val_acc_mean'] for r in result_set]
        test_accs = [r['test_acc_mean'] for r in result_set]
        
        x = np.arange(len(representations))
        width = 0.25
        
        axes[idx].bar(x - width, train_accs, width, label='Train', alpha=0.8, color='green')
        axes[idx].bar(x, val_accs, width, label='Validation', alpha=0.8, color='blue')
        axes[idx].bar(x + width, test_accs, width, label='Test', alpha=0.8, color='orange')
        
        axes[idx].set_ylabel('Accuracy')
        axes[idx].set_title(f'{title}\nTrain/Val/Test Comparison')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(representations, rotation=45, ha='right')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/train_val_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/train_val_test_comparison.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/train_val_test_comparison.png")
    plt.close()

def plot_precision_recall(results, output_dir):
    """Plot precision vs recall"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (result_set, title, color) in enumerate([
        (results['all_layers_results'], 'All Layers (attn1+attn2)', 'skyblue'),
        (results['visual_only_results'], 'Visual Only (attn1)', 'lightcoral')
    ]):
        representations = [r['representation'].replace('_all_layers', '').replace('_visual_only', '') 
                          for r in result_set]
        precisions = [r.get('test_precision_mean', 0) for r in result_set]
        recalls = [r.get('test_recall_mean', 0) for r in result_set]
        
        axes[idx].scatter(recalls, precisions, s=200, alpha=0.6, c=color, edgecolors='black')
        
        for i, rep in enumerate(representations):
            axes[idx].annotate(rep, (recalls[i], precisions[i]), 
                             fontsize=9, ha='center', va='bottom')
        
        axes[idx].set_xlabel('Recall')
        axes[idx].set_ylabel('Precision')
        axes[idx].set_title(f'{title}\nPrecision vs Recall')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, 1.05)
        axes[idx].set_ylim(0, 1.05)
        axes[idx].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_recall.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/precision_recall.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/precision_recall.png")
    plt.close()

def create_summary_table(results, output_dir):
    """Create a LaTeX-ready summary table"""
    with open(f'{output_dir}/summary_table.txt', 'w') as f:
        f.write("="*100 + "\n")
        f.write("SUMMARY TABLE - ALL LAYERS (attn1 + attn2)\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Representation':<25} {'Test Acc':<15} {'Test F1':<15} {'Precision':<15} {'Recall':<15} {'Time(s)':<10}\n")
        f.write("-"*100 + "\n")
        
        for res in results['all_layers_results']:
            rep_name = res['representation'].replace('_all_layers', '')
            f.write(f"{rep_name:<25} "
                   f"{res['test_acc_mean']:.3f}±{res['test_acc_std']:.3f}    "
                   f"{res['test_f1_mean']:.3f}±{res['test_f1_std']:.3f}    "
                   f"{res.get('test_precision_mean', 0):.3f}±{res.get('test_precision_std', 0):.3f}    "
                   f"{res.get('test_recall_mean', 0):.3f}±{res.get('test_recall_std', 0):.3f}    "
                   f"{res.get('training_time_mean', 0):.2f}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("SUMMARY TABLE - VISUAL ONLY (attn1 only - NO TEXT)\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Representation':<25} {'Test Acc':<15} {'Test F1':<15} {'Precision':<15} {'Recall':<15} {'Time(s)':<10}\n")
        f.write("-"*100 + "\n")
        
        for res in results['visual_only_results']:
            rep_name = res['representation'].replace('_visual_only', '')
            f.write(f"{rep_name:<25} "
                   f"{res['test_acc_mean']:.3f}±{res['test_acc_std']:.3f}    "
                   f"{res['test_f1_mean']:.3f}±{res['test_f1_std']:.3f}    "
                   f"{res.get('test_precision_mean', 0):.3f}±{res.get('test_precision_std', 0):.3f}    "
                   f"{res.get('test_recall_mean', 0):.3f}±{res.get('test_recall_std', 0):.3f}    "
                   f"{res.get('training_time_mean', 0):.2f}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("PERFORMANCE DROP ANALYSIS (All Layers → Visual Only)\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Representation':<25} {'Acc Drop':<15} {'F1 Drop':<15} {'Relative Drop %':<20}\n")
        f.write("-"*100 + "\n")
        
        for all_res, vis_res in zip(results['all_layers_results'], results['visual_only_results']):
            rep_name = all_res['representation'].replace('_all_layers', '')
            acc_drop = all_res['test_acc_mean'] - vis_res['test_acc_mean']
            f1_drop = all_res['test_f1_mean'] - vis_res['test_f1_mean']
            rel_drop = (acc_drop / all_res['test_acc_mean']) * 100 if all_res['test_acc_mean'] > 0 else 0
            
            f.write(f"{rep_name:<25} {acc_drop:+.4f}          {f1_drop:+.4f}          {rel_drop:+.2f}%\n")
    
    print(f"✅ Saved: {output_dir}/summary_table.txt")

def main():
    parser = argparse.ArgumentParser(description='Plot ablation experiment results')
    parser.add_argument('--results-json', type=str, required=True,
                       help='Path to results.json file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same as results.json)')
    
    args = parser.parse_args()
    
    results = load_results(args.results_json)
    
    if args.output_dir is None:
        args.output_dir = str(Path(args.results_json).parent)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GENERATING COMPREHENSIVE PLOTS FOR SCIENTIFIC PUBLICATION")
    print("="*80)
    
    print("\n1. Performance Comparison...")
    plot_performance_comparison(results, args.output_dir)
    
    print("\n2. Learning Curves...")
    plot_learning_curves(results, args.output_dir)
    
    print("\n3. Confusion Matrices...")
    plot_confusion_matrices(results, args.output_dir)
    
    print("\n4. Train/Val/Test Gap Analysis...")
    plot_train_val_gap(results, args.output_dir)
    
    print("\n5. Precision vs Recall...")
    plot_precision_recall(results, args.output_dir)
    
    print("\n6. Summary Table...")
    create_summary_table(results, args.output_dir)
    
    print("\n" + "="*80)
    print(f"✅ All plots saved to: {args.output_dir}")
    print("="*80)
    print("\nGenerated files:")
    print("  - performance_comparison.png/pdf")
    print("  - learning_curves.png/pdf")
    print("  - confusion_matrices.png/pdf")
    print("  - train_val_test_comparison.png/pdf")
    print("  - precision_recall.png/pdf")
    print("  - summary_table.txt")

if __name__ == '__main__':
    main()

