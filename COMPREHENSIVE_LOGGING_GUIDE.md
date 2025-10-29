# Comprehensive Logging Guide for Scientific Publications

This document explains all the metrics logged by the enhanced visual ablation experiment and how to use them for scientific papers.

## Overview

The experiment now logs comprehensive metrics suitable for publication-quality analysis:

1. **Performance Metrics** (accuracy, F1, precision, recall)
2. **Training Dynamics** (learning curves, convergence)
3. **Computational Cost** (training time, epochs to convergence)
4. **Per-Class Performance** (confusion matrices)
5. **Overfitting Analysis** (train/val/test gaps)
6. **Statistical Robustness** (multiple runs with mean ± std)

## Metrics Logged

### 1. Classification Performance

**Per-run metrics:**
- `train_acc`: Training set accuracy (for overfitting analysis)
- `val_acc`: Validation set accuracy
- `test_acc`: Test set accuracy
- `val_f1`: Validation F1 score (macro-averaged)
- `test_f1`: Test F1 score (macro-averaged)
- `val_precision`: Validation precision (macro-averaged)
- `test_precision`: Test precision (macro-averaged)
- `val_recall`: Validation recall (macro-averaged)
- `test_recall`: Test recall (macro-averaged)

**Aggregated metrics (across all runs):**
- `*_mean`: Mean of metric across runs
- `*_std`: Standard deviation across runs
- `*_all_runs`: List of values for all runs

### 2. Training Dynamics

**Learning curves:**
- `val_acc_history`: Validation accuracy at each epoch
- `train_loss_history`: Training loss at each epoch
- `best_epoch`: Epoch where best validation accuracy was achieved

**Convergence metrics:**
- `best_epoch_mean`: Average epoch of convergence
- `best_epoch_std`: Std dev of convergence epoch

### 3. Computational Cost

- `training_time`: Time taken for one complete run (seconds)
- `training_time_mean`: Average training time across runs
- `training_time_std`: Std dev of training time
- `training_time_total`: Total time for all runs

### 4. Model Complexity

- `feature_dim`: Dimensionality of feature representation
- `num_layers`: Number of LoRA layers used

### 5. Per-Class Performance

- `val_confusion_matrices`: Confusion matrix for each run (validation)
- `test_confusion_matrices`: Confusion matrix for each run (test)

### 6. Detailed Per-Run Results

Each run stores a complete record:
```json
{
  "run": 1,
  "train_acc": 0.95,
  "val_acc": 0.90,
  "test_acc": 0.88,
  "val_f1": 0.89,
  "test_f1": 0.87,
  "val_precision": 0.90,
  "test_precision": 0.88,
  "val_recall": 0.88,
  "test_recall": 0.86,
  "training_time": 45.2,
  "best_epoch": 87,
  "feature_dim": 2560
}
```

## Generated Visualizations

The `plot_ablation_results.py` script generates publication-ready figures:

### 1. Performance Comparison (`performance_comparison.png/pdf`)
- **Subfigure 1**: Test accuracy comparison (All Layers vs Visual-Only)
- **Subfigure 2**: Test F1 score comparison
- **Subfigure 3**: Performance drop when removing text conditioning
- **Subfigure 4**: Training time comparison

**Usage in paper:**
> "Figure 1 shows the performance comparison between models using all attention layers (attn1+attn2) and visual-only layers (attn1). We observe that..."

### 2. Learning Curves (`learning_curves.png/pdf`)
- Validation accuracy over epochs for best 3 representations
- Shows convergence behavior and training stability
- Vertical dashed lines indicate best epoch

**Usage in paper:**
> "Figure 2 illustrates the learning dynamics. The spectral representation converges in approximately 50 epochs, while..."

### 3. Confusion Matrices (`confusion_matrices.png/pdf`)
- Average confusion matrix across all runs
- Shows per-class performance
- Identifies which classes are confused

**Usage in paper:**
> "As shown in the confusion matrices (Figure 3), the model achieves perfect classification for classes X and Y, but shows confusion between..."

### 4. Train/Val/Test Comparison (`train_val_test_comparison.png/pdf`)
- Compares accuracy across train/val/test splits
- Identifies overfitting (large train-val gap)
- Identifies generalization issues (large val-test gap)

**Usage in paper:**
> "Figure 4 demonstrates the generalization capability. The small train-validation gap indicates minimal overfitting..."

### 5. Precision-Recall Plot (`precision_recall.png/pdf`)
- Scatter plot of precision vs recall
- Shows trade-off for different representations
- Helps identify balanced vs biased classifiers

**Usage in paper:**
> "Figure 5 presents the precision-recall characteristics. The stats representation achieves the best balance with..."

### 6. Summary Table (`summary_table.txt`)
- LaTeX-ready table with all metrics
- Performance drop analysis
- Suitable for direct inclusion in paper

**Usage in paper:**
> "Table 1 summarizes the quantitative results. The best performance is achieved by..."

## Example Analysis Workflow

### For Your Scientific Paper

1. **Run Experiments:**
```bash
# Main experiment
python3 visual_ablation_experiment.py --dataset 1k --num-runs 10 --output-dir results_1k

# Generate plots
python3 plot_ablation_results.py --results-json results_1k/visual_ablation_1k/results.json
```

2. **Key Questions to Answer:**

**Q1: Do visual features alone contain concept information?**
- Compare test accuracy: All Layers vs Visual-Only
- If drop < 5%: Concepts encoded in visual features
- If drop > 20%: Text conditioning is crucial

**Q2: Which representation method is best?**
- Rank by test accuracy (mean ± std)
- Consider computational cost (training time)
- Check overfitting (train-test gap)

**Q3: Is the model robust?**
- Low std across runs indicates stability
- Consistent learning curves show reproducibility
- Balanced precision-recall suggests no class bias

**Q4: Does it generalize?**
- Small val-test gap indicates good generalization
- Similar val and test F1 scores
- Confusion matrix shows uniform per-class performance

3. **Writing Results Section:**

```markdown
## Results

### Overall Performance
We evaluate 8 representation methods across 10 independent runs. 
Table 1 presents the comprehensive results. The spectral representation 
achieves the highest test accuracy of 0.92±0.02 when using all layers.

### Visual-Only Ablation
When restricting to visual self-attention layers only (attn1), we observe 
a modest performance drop of 3.5% (Figure 1c). This suggests that concept 
information is primarily encoded in visual features, with text conditioning 
providing supplementary refinement.

### Training Dynamics
Figure 2 shows the learning curves. All representations converge within 
100 epochs, with the stats representation showing the fastest convergence 
(52±8 epochs). The small standard deviations indicate stable training.

### Generalization Analysis
The train-validation-test comparison (Figure 4) reveals minimal overfitting 
across all representations. The average train-test gap is 2.1%, indicating 
excellent generalization capability.
```

## Statistical Reporting

### Proper Format for Papers

**❌ Bad:** "Accuracy = 0.923"
**✅ Good:** "Accuracy = 0.923 ± 0.018 (n=10 runs)"

**❌ Bad:** "Method A is better than Method B"
**✅ Good:** "Method A (0.92±0.02) outperforms Method B (0.88±0.03) with a 4.3% improvement"

### Reporting Checklist

- [ ] Report mean ± std for all metrics
- [ ] State number of runs (n=10)
- [ ] Include confidence intervals or p-values for comparisons
- [ ] Report both validation and test performance
- [ ] Include computational cost (time, epochs)
- [ ] Show confusion matrices for multi-class problems
- [ ] Report per-class metrics if classes are imbalanced
- [ ] Include learning curves for key results

## Advanced Analysis

### Custom Plots from JSON

You can create custom plots by loading the JSON:

```python
import json
import matplotlib.pyplot as plt

with open('results.json', 'r') as f:
    results = json.load(f)

# Plot performance vs training time
for res in results['all_layers_results']:
    plt.scatter(res['training_time_mean'], res['test_acc_mean'])
    plt.annotate(res['representation'], ...)
```

### Statistical Tests

```python
from scipy.stats import ttest_ind

# Compare two representations
rep1_scores = results['all_layers_results'][0]['test_acc_all_runs']
rep2_scores = results['all_layers_results'][1]['test_acc_all_runs']

t_stat, p_value = ttest_ind(rep1_scores, rep2_scores)
print(f"p-value = {p_value:.4f}")
```

## Best Practices

1. **Run Multiple Seeds**: Always use num_runs >= 10 for robust statistics
2. **Report All Metrics**: Don't cherry-pick, report comprehensive results
3. **Show Variability**: Always include error bars (std dev)
4. **Computational Cost**: Report training time alongside performance
5. **Generalization**: Show train/val/test splits to demonstrate no overfitting
6. **Reproducibility**: Save all hyperparameters in results JSON

## LaTeX Table Template

```latex
\begin{table}[h]
\centering
\caption{Performance comparison of representation methods}
\label{tab:results}
\begin{tabular}{lcccc}
\hline
\textbf{Representation} & \textbf{Test Acc} & \textbf{Test F1} & \textbf{Precision} & \textbf{Recall} \\
\hline
Spectral       & 0.92 ± 0.02 & 0.91 ± 0.02 & 0.92 ± 0.02 & 0.90 ± 0.03 \\
Stats          & 0.90 ± 0.01 & 0.89 ± 0.02 & 0.90 ± 0.02 & 0.88 ± 0.02 \\
Matrix Norms   & 0.88 ± 0.03 & 0.87 ± 0.03 & 0.88 ± 0.03 & 0.86 ± 0.04 \\
\hline
\end{tabular}
\end{table}
```

## Citation

When using these metrics in your paper, consider citing relevant methodology papers:
- Classification metrics: Sokolova & Lapalme (2009)
- Cross-validation: Kohavi (1995)
- Confusion matrices: Stehman (1997)

## Questions?

For specific analysis needs or custom visualizations, refer to:
- `visual_ablation_experiment.py` - Main experiment code
- `plot_ablation_results.py` - Visualization script
- `results.json` - Complete raw data

Happy publishing! 📊📈📉

