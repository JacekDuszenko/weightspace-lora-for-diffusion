# Ablation Experiment: Layer Importance and Novel Representations

## Overview

This experiment investigates the predictive power of individual LoRA layers and explores novel representation methods to maximize the interpretability of LoRA weight spaces. The key innovation is systematically identifying and removing the most predictive layers to understand if the remaining layers still contain useful information.

## Motivation

Current experiments use all LoRA layers together to predict the concept class. However, we don't know:
1. Which layers contribute most to prediction?
2. Is the signal distributed across layers or concentrated in a few key layers?
3. Can we achieve good prediction even after removing the most predictive layers?
4. What representation methods best capture the semantic information in LoRA weights?

## Experiment Design

### Phase 1: Layer Importance Identification
1. Evaluate each LoRA layer individually using a baseline representation (statistics)
2. Train an MLP classifier on each layer separately
3. Rank layers by their individual predictive power (test accuracy)
4. Identify the top-k most predictive layers

### Phase 2: Ablation Study
1. Remove the top-k most predictive layers
2. Re-evaluate all representation methods using only the remaining layers
3. Compare performance with and without ablation

### Phase 3: Representation Comparison
Test multiple representation methods:
- **Existing methods**: flat vector, statistics
- **Novel methods**: spectral features, matrix norms, distribution features, frequency features, information-theoretic features, ensemble

## Novel Representation Methods

### 1. Spectral Features (SVD-based)

**Motivation**: Singular value decomposition reveals the intrinsic structure of weight matrices. The singular values indicate the "effective rank" and information content.

**Features**:
- Top-10 singular values
- Normalized singular values (distribution)
- Effective rank (entropy of singular values)
- Condition number (ratio of largest to smallest singular value)
- Spectral gaps (differences between consecutive singular values)
- Nuclear norm (sum of singular values)

**Dimensionality**: 32 features per layer

**Theoretical Basis**:
- SVD captures the principal directions of variation in weight space
- Singular values indicate which directions contain most information
- Different concepts may have different singular value profiles

### 2. Matrix Norm Features

**Motivation**: Different matrix norms capture different geometric properties. Low-rank matrices (like LoRA) have characteristic norm relationships.

**Features**:
- Frobenius norm (L2 norm of all elements)
- Nuclear norm (sum of singular values)
- Spectral norm (largest singular value)
- L1 norm (max column sum)
- L-infinity norm (max row sum)
- Nuclear/Frobenius ratio (measures rank deficiency)
- Spectral/Frobenius ratio

**Dimensionality**: 7 features per layer

**Theoretical Basis**:
- Nuclear norm is related to matrix rank
- Norm ratios distinguish between full-rank and low-rank matrices
- Different concepts may require different effective ranks

### 3. Higher-Order Distribution Features

**Motivation**: Basic statistics (mean, std) don't capture the full weight distribution. Heavy tails, skewness, and quantiles may be informative.

**Features**:
- Quantiles: 5%, 10%, 25%, 50%, 75%, 90%, 95%
- Interquartile range (Q3 - Q1)
- Mean absolute deviation
- Coefficient of variation
- Skewness and kurtosis

**Dimensionality**: 12 features per layer

**Theoretical Basis**:
- Quantiles are robust to outliers
- Different initialization strategies may produce different distributions
- Concept complexity may correlate with distribution shape

### 4. Layer Coupling Features

**Motivation**: LoRA consists of paired down/up matrices. Their interaction (down @ up^T) reconstructs the full update. The relationship between these matrices may be informative.

**Features**:
- Correlation between down and up weights
- Cosine similarity of flattened matrices
- Frobenius norm of the product down @ up^T
- Spectral alignment (correlation of singular values)

**Dimensionality**: 4 features per layer pair

**Theoretical Basis**:
- Low-rank updates have correlated structure between down/up
- The product captures the actual weight update
- Different concepts may have different coupling patterns

### 5. Frequency Domain Features

**Motivation**: 2D FFT reveals spatial frequency patterns in weight matrices. Convolutional-like patterns would show up in frequency domain.

**Features**:
- Top-10 frequency components (magnitude)
- Mean, std, max of magnitude spectrum
- Energy concentration (ratio of top-k to total energy)

**Dimensionality**: 14 features per layer

**Theoretical Basis**:
- Spatial patterns in weights correspond to frequency components
- High-frequency components indicate fine-grained patterns
- Different concepts may have characteristic frequency signatures

### 6. Information-Theoretic Features

**Motivation**: Information theory provides tools to measure uncertainty and information content.

**Features**:
- Entropy of weight histogram (50 bins)
- Normalized entropy

**Dimensionality**: 2 features per layer

**Theoretical Basis**:
- High entropy indicates more uniform weight distribution
- Low entropy suggests concentrated, structured weights
- Information content may correlate with concept complexity

### 7. Ensemble Features

**Motivation**: Different representations capture complementary information. Combining them may provide the most complete picture.

**Method**: Concatenate all novel representations (spectral + norms + distribution + frequency + info-theoretic)

**Dimensionality**: ~70 features per layer

**Theoretical Basis**:
- Multiple views provide robustness
- Different features may be important for different concepts
- MLP can learn optimal feature weighting

### 8. Variational Autoencoder (VAE) Representation

**Motivation**: Learn a probabilistic latent space that captures the underlying distribution of weight matrices in an unsupervised manner.

**Architecture**:
- Encoder: Input → 512 → 256 → Latent (64-dim)
- Latent space: Mean (μ) and log-variance (log σ²) parameters
- Decoder: Latent → 256 → 512 → Input
- Loss: Reconstruction loss + KL divergence (β=0.1 for β-VAE)

**Dimensionality**: 64 features per layer (adaptive)

**Theoretical Basis**:
- VAEs learn smooth, continuous latent spaces
- KL divergence regularization ensures well-structured representations
- Disentanglement: Different latent dimensions may capture different semantic aspects
- Probabilistic framework provides uncertainty estimates

**Training**:
- 100 epochs with early stopping (patience=15)
- Adam optimizer (lr=0.001)
- Batch normalization for stable training
- Trained separately for each layer on flattened weights

### 9. Contrastive Autoencoder Representation

**Motivation**: Learn discriminative representations where similar samples cluster together and dissimilar samples are pushed apart, inspired by SimCLR contrastive learning.

**Architecture**:
- Encoder: Input → 512 → 512 → Latent (128-dim)
- Projection head: Latent → 64 → 64 (for contrastive loss)
- Decoder: Latent → 512 → 512 → Input
- Loss: Reconstruction loss + InfoNCE contrastive loss

**Dimensionality**: 128 features per layer (adaptive)

**Theoretical Basis**:
- Contrastive learning naturally creates discriminative features
- InfoNCE loss maximizes mutual information between representations
- Learns invariances to noise and irrelevant variations
- Self-supervised learning without requiring labels
- Encourages separation between different concept classes

**Training**:
- 100 epochs with early stopping
- Contrastive learning uses within-batch negatives
- Temperature scaling (τ=0.5) for contrastive loss
- Dropout (0.2) for regularization

### 10. Denoising Autoencoder Representation

**Motivation**: Force the model to learn robust, essential features by reconstructing clean data from corrupted inputs.

**Architecture**:
- Encoder: Input → 512 → 256 → Latent (64-dim)
- Noise corruption: Additive Gaussian noise (σ=0.2)
- Decoder: Latent → 256 → 512 → Input
- Loss: MSE reconstruction loss on clean data

**Dimensionality**: 64 features per layer (adaptive)

**Theoretical Basis**:
- Denoising forces learning of robust, essential structure
- Removes noise and captures underlying manifold
- Particularly useful for weight spaces with redundancy
- Learns features invariant to small perturbations
- Encourages learning of sparse, meaningful representations

**Training**:
- 100 epochs with early stopping
- Noise added only during training
- Dropout (0.2) for additional regularization
- Skip connections for better gradient flow

## Unsupervised vs Hand-Crafted Features

| Aspect | Hand-Crafted Features | Unsupervised Learning |
|--------|----------------------|----------------------|
| Design | Expert-designed | Data-driven |
| Flexibility | Fixed | Adapts to data |
| Interpretability | High | Lower |
| Computational Cost | Low | High (requires training) |
| Generalization | May miss patterns | Can discover novel patterns |
| Dimensionality | Typically lower | Controllable (latent dim) |

**When to use each**:
- **Hand-crafted**: When you understand the problem domain, need interpretability, or have limited compute
- **Unsupervised**: When the optimal features are unknown, data is complex, or you want to discover novel patterns

## Implementation Details

### Model Architecture
- 3-layer MLP with 512 hidden units
- ReLU activation
- Dropout (0.3) for regularization
- Output: softmax over concept classes

### Training Configuration
- Optimizer: Adam (lr=0.001)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)
- Loss: Cross-entropy
- Batch size: 32
- Max epochs: 1000
- Early stopping: patience=20
- Multiple runs: 10 (for averaging)

### Evaluation Metrics
- Accuracy (train, validation, test)
- F1-score (macro average)
- All metrics reported as mean ± std over multiple runs

## Usage

### Basic Usage
```bash
python ablation_experiment.py --dataset=1k --top-k=5 --num-runs=10 --output-dir=ablation-results
```

### Parameters
- `--dataset`: Dataset size ['1k', '10k', '50k', '100k', 'small']
- `--top-k`: Number of top predictive layers to ablate (default: 5)
- `--num-runs`: Number of training runs for averaging (default: 10)
- `--output-dir`: Directory to save results (default: 'ablation-results')
- `--quick`: Quick mode with fewer runs (useful for testing)

### Quick Test
```bash
python ablation_experiment.py --dataset=1k --top-k=3 --quick
```

## Expected Outputs

The experiment creates a directory structure:
```
ablation-results/
└── ablation_1k/
    ├── top_layers.json          # Top-k most predictive layers
    ├── results.json              # Complete results (JSON)
    └── results.txt               # Human-readable summary
```

### Output Files

**top_layers.json**: Lists the most predictive layers with their accuracies
```json
[
  {"layer": "lora.down.weight.80", "accuracy": 0.8542},
  {"layer": "lora.up.weight.80", "accuracy": 0.8231},
  ...
]
```

**results.json**: Complete experimental results including:
- Dataset configuration
- Top ablated layers
- Results with ablation (all representations)
- Baseline results without ablation

**results.txt**: Human-readable table comparing all methods

## Research Questions

1. **Layer Importance**: Which LoRA layers are most predictive?
   - Hypothesis: Later layers (closer to output) may be more predictive

2. **Robustness**: Does prediction still work after removing top layers?
   - Hypothesis: If signal is distributed, performance should degrade gracefully

3. **Best Representation**: Which representation method maximizes predictability?
   - Hypothesis: Ensemble or spectral features may outperform simple statistics

4. **Complementarity**: Do different layers encode different information?
   - Hypothesis: Top layers may capture coarse features, others fine details

5. **Rank Structure**: How does low-rank structure relate to concept learning?
   - Hypothesis: Effective rank may correlate with concept complexity

## Theoretical Framework

### Weight Space Hypothesis
The fundamental hypothesis is that **LoRA weight matrices encode semantic information** about the fine-tuning concept. This experiment tests:

1. **Localization**: Is semantic information localized to specific layers?
2. **Distribution**: How is information distributed across the weight space?
3. **Representation**: What mathematical structures best capture this information?

### Expected Outcomes

**Scenario 1: Information is Localized**
- Removing top layers drastically reduces performance
- Few layers are highly predictive
- Suggests concept learning happens in specific architectural components

**Scenario 2: Information is Distributed**
- Removing top layers has moderate impact
- Many layers contribute to prediction
- Suggests concept learning is distributed across the network

**Scenario 3: Redundancy**
- Removing top layers has little impact
- Many layers contain redundant information
- Suggests concept is encoded robustly across architecture

## Comparison with Original Experiments

| Aspect | Original Experiments | Ablation Experiment |
|--------|---------------------|---------------------|
| Focus | All layers together | Individual layer importance |
| Goal | Maximize accuracy | Understand layer contributions |
| Representations | 5 methods | 8 methods (3 new) |
| Analysis | Overall performance | Layer-wise + ablation study |
| Insights | Feasibility | Mechanism |

## Extensions and Future Work

### Possible Extensions
1. **Progressive ablation**: Remove layers one by one to see degradation curve
2. **Layer clustering**: Group similar layers based on their representations
3. **Attention visualization**: Visualize which layers the model attends to
4. **Concept-specific analysis**: Do different concepts use different layers?
5. **Transfer learning**: Use learned representations for other tasks
6. **Interpretability**: Visualize what specific layers encode

### Integration with Other Analyses
- Combine with gradient-based attribution methods
- Analyze layer importance vs. depth
- Compare with activation-based analysis (requires inference)
- Study temporal dynamics during training

## Technical Notes

### Memory Efficiency
- Uses incremental processing for large datasets
- GPU acceleration for SVD and PCA
- Efficient caching of intermediate results

### Computational Cost
Estimated runtime on A100 GPU:
- 1k dataset: ~30 minutes
- 10k dataset: ~2 hours
- 50k dataset: ~8 hours

Most expensive operations:
1. Layer-by-layer evaluation (Phase 1)
2. SVD for spectral features
3. 2D FFT for frequency features

### Reproducibility
- All random seeds are set
- Multiple runs for statistical significance
- Results include standard deviations

## References

This experiment builds on:
1. The original weight-space interpretation paper
2. Classical interpretability techniques (ablation studies)
3. Representation learning theory
4. Matrix analysis and spectral theory

## Contact

For questions or issues, please refer to the main repository README.
