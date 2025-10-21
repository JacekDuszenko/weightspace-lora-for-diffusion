# Comprehensive Guide to LoRA Representation Methods

This document provides a detailed overview of all representation methods implemented for LoRA weight-space interpretation, including both hand-crafted features and advanced unsupervised learning approaches.

## Overview

The goal of these representation methods is to extract meaningful features from LoRA weight matrices that maximize the predictability of the concept class used for training. Each method captures different aspects of the weight space structure.

## Categories of Representations

### 1. Hand-Crafted Statistical Features
Simple statistical summaries that are fast to compute and highly interpretable.

### 2. Spectral and Algebraic Features
Features derived from matrix decompositions and linear algebra properties.

### 3. Distribution and Information-Theoretic Features
Features capturing the shape and information content of weight distributions.

### 4. Frequency Domain Features
Features extracted from the spatial frequency patterns in weight matrices.

### 5. Unsupervised Learned Representations
Deep learning models that learn optimal representations directly from data.

---

## Detailed Method Descriptions

## Hand-Crafted Methods

### 1. Flat Vector (Baseline)
**Type**: Raw representation
**Dimensionality**: ~100,000 (all weights concatenated)
**Compute Time**: O(n) - very fast
**Interpretability**: Low

**Description**: Simply flattens all weight matrices into a single vector. Serves as a baseline.

**Pros**:
- No information loss
- Simple and fast
- Works surprisingly well in practice

**Cons**:
- Very high dimensional
- No feature engineering
- Requires large datasets
- Sensitive to weight initialization

**When to use**: As a baseline comparison or when you have abundant data and computational resources.

---

### 2. Basic Statistics
**Type**: Statistical features
**Dimensionality**: 7 per layer
**Compute Time**: O(n) - very fast
**Interpretability**: High

**Features**:
- Mean, Standard deviation
- Median, Min, Max
- Skewness, Kurtosis

**Description**: Computes basic statistical moments of weight distributions.

**Pros**:
- Very fast to compute
- Highly interpretable
- Low dimensional
- Robust baseline

**Cons**:
- May miss complex patterns
- Assumes weights are IID
- Discards spatial structure

**When to use**: When you need fast, interpretable results or as a strong baseline.

---

## Advanced Hand-Crafted Methods

### 3. Spectral Features (SVD-based)
**Type**: Spectral/algebraic
**Dimensionality**: 32 per layer
**Compute Time**: O(min(m,n)²) - moderate
**Interpretability**: Medium-High

**Features**:
- Top-10 singular values
- Normalized singular values
- Effective rank (entropy of singular values)
- Condition number
- Spectral gaps (between consecutive singular values)
- Nuclear norm

**Description**: Uses Singular Value Decomposition to extract features about the intrinsic dimensionality and structure of weight matrices.

**Mathematical Foundation**:
```
W = UΣV^T
Effective Rank = exp(-Σ(σᵢ log σᵢ))  where σᵢ are normalized singular values
Condition Number = σ₁ / σₙ
```

**Pros**:
- Captures low-rank structure
- Theoretically motivated
- Rotation invariant
- Reveals intrinsic dimensionality

**Cons**:
- More expensive to compute
- Requires numerical stability
- May be sensitive to noise

**When to use**: When the low-rank structure of LoRA is expected to be informative. Particularly useful since LoRA explicitly uses low-rank decomposition.

**Research Insight**: Since LoRA learns W = BA where B and A are low-rank, the singular value spectrum directly relates to how the rank-r update is distributed across dimensions. Different concepts may have different rank structures.

---

### 4. Matrix Norm Features
**Type**: Algebraic
**Dimensionality**: 7 per layer
**Compute Time**: O(n) to O(min(m,n)²)
**Interpretability**: Medium

**Features**:
- Frobenius norm (√Σwᵢⱼ²)
- Nuclear norm (Σσᵢ)
- Spectral norm (max σᵢ)
- L1 norm (max column sum)
- L∞ norm (max row sum)
- Nuclear/Frobenius ratio
- Spectral/Frobenius ratio

**Description**: Different matrix norms capture different geometric properties. Norm ratios are particularly informative for low-rank matrices.

**Mathematical Foundation**:
- For rank-r matrix: Nuclear norm ≥ √r × Frobenius norm
- Ratio approaches 1 as matrix becomes more "spread out" across singular values
- Useful for detecting rank deficiency

**Pros**:
- Fast to compute (except nuclear norm)
- Theoretically grounded
- Complementary to spectral features

**Cons**:
- Overlaps with spectral features
- Less interpretable individually

**When to use**: In combination with spectral features to get a complete picture of matrix structure.

---

### 5. Higher-Order Distribution Features
**Type**: Statistical
**Dimensionality**: 12 per layer
**Compute Time**: O(n log n)
**Interpretability**: High

**Features**:
- Quantiles: 5%, 10%, 25%, 50%, 75%, 90%, 95%
- Interquartile range (IQR)
- Mean absolute deviation (MAD)
- Coefficient of variation (CV)
- Skewness, Kurtosis

**Description**: Goes beyond basic statistics to capture the full shape of weight distributions.

**Pros**:
- Robust to outliers (quantiles)
- Captures distribution shape
- Highly interpretable
- Fast to compute

**Cons**:
- Still assumes independence
- May miss spatial patterns

**When to use**: When you want robust statistical features that are less sensitive to outliers than mean/std.

**Research Insight**: Different initialization strategies (Xavier, He, etc.) produce different weight distributions. Fine-tuning may shift these distributions in concept-specific ways.

---

### 6. Frequency Domain Features
**Type**: Signal processing
**Dimensionality**: 14 per layer
**Compute Time**: O(n log n)
**Interpretability**: Medium

**Features**:
- Top-10 frequency components (FFT magnitude)
- Mean, std, max of magnitude spectrum
- Energy concentration ratio

**Description**: Applies 2D Fast Fourier Transform to weight matrices to extract frequency domain features.

**Mathematical Foundation**:
```
F(u,v) = Σₘ Σₙ W(m,n) exp(-2πi(um/M + vn/N))
Magnitude Spectrum = |F(u,v)|
```

**Pros**:
- Captures spatial patterns
- Reveals repetitive structures
- Different from other methods

**Cons**:
- Less interpretable
- Assumes spatial structure matters
- Sensitive to matrix size

**When to use**: When you suspect weight matrices have spatial/periodic patterns. More relevant for convolutional layers.

**Research Insight**: If LoRA learns structured patterns (e.g., repeated motifs across channels), these would show up as peaks in frequency domain.

---

### 7. Information-Theoretic Features
**Type**: Information theory
**Dimensionality**: 2 per layer
**Compute Time**: O(n)
**Interpretability**: Medium

**Features**:
- Entropy of weight histogram
- Normalized entropy

**Description**: Measures information content and uncertainty in weight distributions.

**Mathematical Foundation**:
```
H(W) = -Σ p(w) log p(w)
Normalized Entropy = H(W) / log(n_bins)
```

**Pros**:
- Principled measure of information
- Complementary to other features
- Fast to compute

**Cons**:
- Requires binning (parameter choice)
- Limited features
- Overlaps with distribution features

**When to use**: As supplementary features in an ensemble.

---

### 8. Layer Coupling Features
**Type**: Structural
**Dimensionality**: 4 per layer pair
**Compute Time**: O(mn)
**Interpretability**: Medium

**Features**:
- Correlation between down and up weights
- Cosine similarity
- Frobenius norm of product (down @ up^T)
- Spectral alignment (correlation of singular values)

**Description**: Captures the relationship between paired down and up matrices in LoRA.

**Mathematical Foundation**:
LoRA update: ΔW = BA where B is down, A is up
The interaction between B and A determines the actual weight update

**Pros**:
- Specifically designed for LoRA structure
- Captures paired structure
- Theoretically motivated

**Cons**:
- Only works for paired layers
- More expensive to compute

**When to use**: When evaluating paired down/up layers together. Particularly relevant for LoRA since the product BA is what matters.

---

### 9. Ensemble Features
**Type**: Meta-feature
**Dimensionality**: ~70 per layer
**Compute Time**: Sum of all methods
**Interpretability**: Low

**Description**: Concatenates multiple hand-crafted representations:
- Spectral features
- Matrix norms
- Distribution features
- Frequency features
- Information-theoretic features

**Pros**:
- Combines multiple views
- Robust to any single feature failing
- Often achieves best performance

**Cons**:
- Higher dimensional
- Less interpretable
- More expensive to compute

**When to use**: When you want maximum performance and don't need interpretability.

---

## Unsupervised Learning Methods

### 10. Variational Autoencoder (VAE)
**Type**: Unsupervised deep learning
**Dimensionality**: 64 (latent dim, adaptive)
**Compute Time**: O(n) inference after O(epochs × n) training
**Interpretability**: Low

**Architecture**:
```
Encoder: Input → Dense(512) → Dense(256) → [μ, log σ²]
Latent: z = μ + σ * ε, where ε ~ N(0,1)
Decoder: z → Dense(256) → Dense(512) → Output
```

**Loss Function**:
```
L = L_reconstruction + β * L_KL
L_reconstruction = MSE(output, input)
L_KL = -0.5 * Σ(1 + log σ² - μ² - σ²)
```

**Description**: Learns a probabilistic latent space that captures the underlying distribution of weight matrices.

**Pros**:
- Learns optimal features from data
- Probabilistic framework
- Can capture complex patterns
- Latent space is smooth and continuous
- β-VAE encourages disentanglement

**Cons**:
- Requires training (100 epochs)
- Less interpretable
- More hyperparameters
- May require more data

**When to use**: When you have sufficient data and compute, and want to discover novel patterns not captured by hand-crafted features.

**Research Insight**: VAEs may learn to disentangle different aspects of concepts (e.g., object shape vs. texture) into different latent dimensions. The smooth latent space enables interpolation between concepts.

---

### 11. Contrastive Autoencoder
**Type**: Unsupervised deep learning
**Dimensionality**: 128 (latent dim, adaptive)
**Compute Time**: O(n²) training (due to contrastive loss), O(n) inference
**Interpretability**: Low

**Architecture**:
```
Encoder: Input → Dense(512) → Dense(512) → Dense(128)
Projection: Latent(128) → Dense(64) → Dense(64)
Decoder: Latent(128) → Dense(512) → Dense(512) → Output
```

**Loss Function**:
```
L = L_reconstruction + α * L_contrastive
L_contrastive = InfoNCE loss using within-batch negatives
```

**Description**: Combines autoencoding with contrastive learning (SimCLR-style) to create discriminative representations.

**Pros**:
- Naturally creates discriminative features
- Learns to separate different concepts
- Robust to noise
- Self-supervised learning
- May achieve better class separation

**Cons**:
- More expensive training (O(n²) per batch)
- Requires careful tuning
- Less interpretable
- Needs large batches for good negatives

**When to use**: When you prioritize discriminative power over reconstruction quality. Best for classification tasks.

**Research Insight**: Contrastive learning has achieved state-of-the-art results in vision tasks. Applying it to weight spaces may reveal that certain LoRA weights are "similar" in a learned metric that's more meaningful than Euclidean distance.

---

### 12. Denoising Autoencoder
**Type**: Unsupervised deep learning
**Dimensionality**: 64 (latent dim, adaptive)
**Compute Time**: O(n) inference, O(epochs × n) training
**Interpretability**: Low

**Architecture**:
```
Encoder: Input + Noise(σ=0.2) → Dense(512) → Dense(256) → Dense(64)
Decoder: Latent(64) → Dense(256) → Dense(512) → Output
```

**Loss Function**:
```
L = MSE(output, clean_input)
Input is corrupted with Gaussian noise during training
```

**Description**: Forces the model to learn robust features by reconstructing clean data from corrupted inputs.

**Pros**:
- Learns robust features
- Removes noise automatically
- Captures essential structure
- Good for noisy data
- Simpler than VAE

**Cons**:
- Requires choosing noise level
- Less principled than VAE
- May discard important details

**When to use**: When weight spaces may have noise or redundancy that should be filtered out. Useful for finding the "core" structure.

**Research Insight**: LoRA weights may have redundancy or noise from SGD optimization. Denoising autoencoders can extract the essential structure that determines concept learning.

---

## Comparison Matrix

| Method | Dimensionality | Speed | Interpretability | Best For |
|--------|---------------|-------|------------------|----------|
| Flat Vector | ~100k | ⚡⚡⚡ | ⭐ | Baseline, abundant data |
| Basic Stats | 7 | ⚡⚡⚡ | ⭐⭐⭐ | Quick analysis, interpretability |
| Spectral | 32 | ⚡⚡ | ⭐⭐⭐ | Low-rank structure |
| Matrix Norms | 7 | ⚡⚡⚡ | ⭐⭐ | Complementing spectral |
| Distribution | 12 | ⚡⚡ | ⭐⭐⭐ | Robust statistics |
| Frequency | 14 | ⚡⚡ | ⭐⭐ | Spatial patterns |
| Info-Theoretic | 2 | ⚡⚡⚡ | ⭐⭐ | Supplementary |
| Layer Coupling | 4 | ⚡⚡ | ⭐⭐ | LoRA-specific |
| Ensemble | ~70 | ⚡ | ⭐ | Maximum performance |
| VAE | 64 | ⚡ | ⭐ | Complex patterns, smooth space |
| Contrastive | 128 | ⚡ | ⭐ | Discrimination, classification |
| Denoising | 64 | ⚡ | ⭐ | Robust features, noisy data |

Legend:
- Speed: ⚡⚡⚡ (fast) to ⚡ (slow)
- Interpretability: ⭐ (low) to ⭐⭐⭐ (high)

---

## Recommended Strategies

### Strategy 1: Quick Exploration
**Goal**: Fast initial results
**Methods**: Basic Stats, Matrix Norms, Distribution
**Time**: ~10 minutes on 1k dataset

### Strategy 2: Comprehensive Hand-Crafted
**Goal**: Maximum interpretability
**Methods**: Ensemble (all hand-crafted features)
**Time**: ~30 minutes on 1k dataset

### Strategy 3: Deep Learning Approach
**Goal**: Maximum performance
**Methods**: VAE, Contrastive, Denoising
**Time**: ~2 hours on 1k dataset

### Strategy 4: Hybrid Approach
**Goal**: Best of both worlds
**Methods**: Spectral + Distribution + Contrastive
**Time**: ~1 hour on 1k dataset

---

## Implementation Notes

### Memory Considerations
- Flat vector: Requires storing ~100k features per sample
- Autoencoders: Require GPU memory for batch processing
- Spectral methods: May require SVD workspace

### Computational Bottlenecks
1. SVD computation for spectral features: O(min(m,n)²)
2. Autoencoder training: 100 epochs × dataset size
3. Contrastive loss: O(batch_size²) per batch

### Hyperparameter Tuning
- Autoencoder latent dims: Adaptive based on input size
- VAE β parameter: 0.1 (can be tuned for more/less disentanglement)
- Contrastive temperature: 0.5 (controls hardness of negatives)
- Denoising noise factor: 0.2 (should match expected noise level)

---

## Theoretical Insights

### Why Weight Spaces Are Informative

**Hypothesis**: LoRA weight matrices encode semantic information about the fine-tuning concept through:

1. **Magnitude Patterns**: Different concepts may require different update magnitudes
2. **Rank Structure**: Complex concepts may need higher effective rank
3. **Distribution Shape**: Different initializations + concept learning = characteristic distributions
4. **Spatial Patterns**: Convolutional-like patterns may emerge in attention weights
5. **Latent Structure**: Deep autoencoders may discover non-linear manifolds

### Connection to Representation Learning Theory

This work bridges two areas:
1. **Model Interpretability**: Understanding what neural networks learn
2. **Meta-Learning**: Learning about learning (weights that encode learning)

LoRA weight-space interpretation is essentially:
- **First-order meta-learning**: Using weights from one model to predict properties of that model
- **Transfer learning in weight space**: Can we transfer insights across different LoRAs?

### Future Directions

1. **Graph Neural Networks**: Treat weight matrices as graphs
2. **Transformer Autoencoders**: Use attention mechanisms in latent space
3. **Metric Learning**: Learn optimal distance metrics in weight space
4. **Causal Analysis**: Identify causal relationships between weight patterns and concepts
5. **Zero-Shot Prediction**: Predict concepts for unseen LoRAs

---

## Datasets

All experiments use HuggingFace datasets:
- **1k samples per class**: `jacekduszenko/lora-ws-1k`
- **10k samples per class**: `jacekduszenko/lora-ws-10k`
- **50k samples per class**: `jacekduszenko/lora-ws-50k`
- **100k samples per class**: `jacekduszenko/lora-ws-100k`

Classes: airplane, bird, cat, car, dog, fish, fruit, ship, snake, vegetable

---

## Citation

If you use these representation methods, please cite:
```bibtex
@article{weightspace-lora,
  title={Towards weight-space interpretation of Low-Rank Adapters for Diffusion Models},
  author={...},
  year={2024}
}
```

---

## Conclusion

This comprehensive suite of representation methods provides multiple ways to analyze LoRA weight spaces:

- **Hand-crafted features**: Fast, interpretable, theoretically grounded
- **Unsupervised learning**: Powerful, data-driven, potentially discovers novel patterns

The ablation experiment framework allows systematic comparison of all these methods to identify which representations best capture the semantic information encoded in LoRA weights.
