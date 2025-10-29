"""
Ablation Experiment: Discarding Most Predictive Layers

This experiment identifies the LoRA layers with the highest predictive power,
discards them, and re-evaluates all representation methods on the remaining layers.
Additionally implements novel representation methods to maximize predictability.

Author: Claude
Date: 2025-10-21
"""

import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import json
from typing import List, Tuple, Dict, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm
from pca_gpu import IncrementalPCAonGPU
from scipy import stats as scipy_stats
from scipy.fft import fft2
from scipy.spatial.distance import cdist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# Model Architecture
# ============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron for classification"""
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================================
# Dataset Utilities
# ============================================================================

def load_and_split_dataset(dataset_name='1k'):
    """Load dataset from HuggingFace and split into train/val/test"""
    dataset_map = {
        '1k': 'jacekduszenko/lora-ws-1k',
        '10k': 'jacekduszenko/lora-ws-10k',
        '50k': 'jacekduszenko/lora-ws-50k',
        '100k': 'jacekduszenko/lora-ws-100k',
        'small': 'jacekduszenko/weights-dataset-small'
    }

    dataset = load_dataset(dataset_map[dataset_name])['train']

    # Split: 70% train, 10% val, 20% test
    initial_split = dataset.train_test_split(test_size=0.3, seed=42)
    validation_test_split = initial_split["test"].train_test_split(test_size=0.67, seed=42)

    training_set = initial_split["train"]
    validation_set = validation_test_split["train"]
    testing_set = validation_test_split["test"]

    return training_set, validation_set, testing_set


def get_lora_layers(dataset):
    """Extract list of LoRA layer keys from dataset"""
    lora_layers = [key for key in dataset.features.keys()
                  if 'lora.down.weight' in key or 'lora.up.weight' in key]
    return sorted(lora_layers)


def get_layer_pairs(lora_layers):
    """Group down and up weights into pairs"""
    down_layers = sorted([l for l in lora_layers if 'down' in l])
    up_layers = sorted([l for l in lora_layers if 'up' in l])

    pairs = []
    for down in down_layers:
        # Find corresponding up layer
        layer_id = down.replace('lora.down.weight.', '')
        up = f'lora.up.weight.{layer_id}'
        if up in up_layers:
            pairs.append((down, up))

    return pairs


# ============================================================================
# Novel Representation Methods
# ============================================================================

def spectral_features(layer_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract spectral features using SVD

    Features:
    - Top-k singular values
    - Singular value ratios
    - Effective rank (entropy-based)
    - Condition number
    - Spectral gap
    """
    batch_size = layer_tensor.size(0)
    features_list = []

    for i in range(batch_size):
        matrix = layer_tensor[i]

        # Compute SVD
        try:
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

            # Top-10 singular values (pad if necessary)
            top_k = 10
            if len(S) < top_k:
                S_padded = torch.zeros(top_k, device=device)
                S_padded[:len(S)] = S
                S = S_padded
            else:
                S = S[:top_k]

            # Singular value ratios
            S_norm = S / (S.sum() + 1e-8)

            # Effective rank (entropy of normalized singular values)
            entropy = -(S_norm * torch.log(S_norm + 1e-8)).sum()

            # Condition number
            cond_num = S[0] / (S[-1] + 1e-8)

            # Spectral gap (relative difference between consecutive singular values)
            spectral_gaps = (S[:-1] - S[1:]) / (S[:-1] + 1e-8)

            # Nuclear norm (sum of singular values)
            nuclear_norm = S.sum()

            # Concatenate features
            features = torch.cat([
                S,  # Top-k singular values
                S_norm,  # Normalized singular values
                spectral_gaps,  # Spectral gaps
                torch.tensor([entropy, cond_num, nuclear_norm], device=device)
            ])

        except Exception as e:
            # Fallback to zeros if SVD fails
            features = torch.zeros(10 + 10 + 9 + 3, device=device)

        features_list.append(features)

    return torch.stack(features_list)


def matrix_norm_features(layer_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract various matrix norm features

    Features:
    - Frobenius norm
    - Nuclear norm (sum of singular values)
    - Spectral norm (largest singular value)
    - L1 norm
    - L-infinity norm
    - Norm ratios
    """
    batch_size = layer_tensor.size(0)
    features_list = []

    for i in range(batch_size):
        matrix = layer_tensor[i]

        # Frobenius norm
        frob_norm = torch.linalg.norm(matrix, ord='fro')

        # Nuclear norm (sum of singular values)
        nuclear_norm = torch.linalg.norm(matrix, ord='nuc')

        # Spectral norm (largest singular value)
        spectral_norm = torch.linalg.norm(matrix, ord=2)

        # L1 norm (max column sum)
        l1_norm = torch.max(torch.sum(torch.abs(matrix), dim=0))

        # L-infinity norm (max row sum)
        linf_norm = torch.max(torch.sum(torch.abs(matrix), dim=1))

        # Norm ratios (measure of rank deficiency)
        nuclear_frob_ratio = nuclear_norm / (frob_norm + 1e-8)
        spectral_frob_ratio = spectral_norm / (frob_norm + 1e-8)

        features = torch.tensor([
            frob_norm, nuclear_norm, spectral_norm,
            l1_norm, linf_norm,
            nuclear_frob_ratio, spectral_frob_ratio
        ], device=device)

        features_list.append(features)

    return torch.stack(features_list)


def distribution_features(layer_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract higher-order distribution features

    Features:
    - Quantiles (5%, 10%, 25%, 50%, 75%, 90%, 95%)
    - Interquartile range
    - Mean absolute deviation
    - Coefficient of variation
    - Skewness and kurtosis (already in stats, but included for completeness)
    """
    batch_size = layer_tensor.size(0)
    features_list = []

    for i in range(batch_size):
        matrix = layer_tensor[i].flatten()

        # Quantiles
        quantiles = torch.quantile(matrix, torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], device=device))

        # Interquartile range
        iqr = quantiles[4] - quantiles[2]  # Q3 - Q1

        # Mean absolute deviation
        mad = torch.mean(torch.abs(matrix - matrix.mean()))

        # Coefficient of variation
        cv = matrix.std() / (matrix.mean().abs() + 1e-8)

        # Skewness and kurtosis
        mean = matrix.mean()
        std = matrix.std()
        skewness = torch.mean(((matrix - mean) / (std + 1e-8)) ** 3)
        kurtosis = torch.mean(((matrix - mean) / (std + 1e-8)) ** 4)

        features = torch.cat([
            quantiles,
            torch.tensor([iqr, mad, cv, skewness, kurtosis], device=device)
        ])

        features_list.append(features)

    return torch.stack(features_list)


def layer_coupling_features(down_tensor: torch.Tensor, up_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract features capturing interaction between down and up matrices

    Features:
    - Correlation between down and up weights
    - Frobenius norm of the product
    - Alignment score (cosine similarity of flattened matrices)
    - Spectral alignment (correlation of singular values)
    """
    batch_size = down_tensor.size(0)
    features_list = []

    for i in range(batch_size):
        down = down_tensor[i]
        up = up_tensor[i]

        # Flatten for correlation
        down_flat = down.flatten()
        up_flat = up.flatten()

        # Pad to same length if necessary
        if len(down_flat) < len(up_flat):
            down_flat = torch.cat([down_flat, torch.zeros(len(up_flat) - len(down_flat), device=device)])
        elif len(up_flat) < len(down_flat):
            up_flat = torch.cat([up_flat, torch.zeros(len(down_flat) - len(up_flat), device=device)])

        # Correlation (Pearson)
        correlation = torch.corrcoef(torch.stack([down_flat, up_flat]))[0, 1]
        if torch.isnan(correlation):
            correlation = torch.tensor(0.0, device=device)

        # Cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            down_flat.unsqueeze(0), up_flat.unsqueeze(0)
        )[0]

        # Product norm
        try:
            product = torch.matmul(down, up.T)
            product_norm = torch.linalg.norm(product, ord='fro')
        except:
            product_norm = torch.tensor(0.0, device=device)

        # Spectral alignment (correlation of singular values)
        try:
            _, S_down, _ = torch.linalg.svd(down, full_matrices=False)
            _, S_up, _ = torch.linalg.svd(up, full_matrices=False)

            # Pad to same length
            min_len = min(len(S_down), len(S_up))
            S_down = S_down[:min_len]
            S_up = S_up[:min_len]

            spectral_corr = torch.corrcoef(torch.stack([S_down, S_up]))[0, 1]
            if torch.isnan(spectral_corr):
                spectral_corr = torch.tensor(0.0, device=device)
        except:
            spectral_corr = torch.tensor(0.0, device=device)

        features = torch.tensor([
            correlation, cosine_sim, product_norm, spectral_corr
        ], device=device)

        features_list.append(features)

    return torch.stack(features_list)


def frequency_features(layer_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract frequency domain features using 2D FFT

    Features:
    - Mean/std of magnitude spectrum
    - Dominant frequency statistics
    - Energy concentration
    """
    batch_size = layer_tensor.size(0)
    features_list = []

    for i in range(batch_size):
        matrix = layer_tensor[i].cpu().numpy()

        # 2D FFT
        fft_matrix = fft2(matrix)
        magnitude = np.abs(fft_matrix)

        # Statistics of magnitude spectrum
        mag_mean = np.mean(magnitude)
        mag_std = np.std(magnitude)
        mag_max = np.max(magnitude)

        # Energy concentration (ratio of top-k components to total)
        magnitude_flat = magnitude.flatten()
        magnitude_sorted = np.sort(magnitude_flat)[::-1]
        total_energy = np.sum(magnitude_flat ** 2)
        top_k_energy = np.sum(magnitude_sorted[:10] ** 2)
        energy_concentration = top_k_energy / (total_energy + 1e-8)

        # Dominant frequency components
        top_k_freqs = magnitude_sorted[:10]

        features = torch.tensor(
            np.concatenate([
                top_k_freqs,
                [mag_mean, mag_std, mag_max, energy_concentration]
            ]),
            device=device,
            dtype=torch.float32
        )

        features_list.append(features)

    return torch.stack(features_list)


def information_theoretic_features(layer_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract information-theoretic features

    Features:
    - Entropy of weight histogram
    - Normalized entropy
    """
    batch_size = layer_tensor.size(0)
    features_list = []

    for i in range(batch_size):
        matrix = layer_tensor[i].flatten()

        # Create histogram
        hist, _ = torch.histogram(matrix, bins=50)
        hist = hist.float()
        hist = hist / (hist.sum() + 1e-8)  # Normalize

        # Entropy
        entropy = -(hist * torch.log(hist + 1e-8)).sum()

        # Normalized entropy (divide by log of number of bins)
        normalized_entropy = entropy / np.log(50)

        features = torch.tensor([entropy, normalized_entropy], device=device)
        features_list.append(features)

    return torch.stack(features_list)


def ensemble_features(layer_tensor: torch.Tensor, down_up_pair=None) -> torch.Tensor:
    """
    Combine multiple representation methods

    This creates a comprehensive feature set by combining:
    - Spectral features
    - Matrix norms
    - Distribution features
    - Information-theoretic features
    - Frequency features
    """
    spectral_feats = spectral_features(layer_tensor)
    norm_feats = matrix_norm_features(layer_tensor)
    dist_feats = distribution_features(layer_tensor)
    freq_feats = frequency_features(layer_tensor)
    info_feats = information_theoretic_features(layer_tensor)

    features = torch.cat([
        spectral_feats,
        norm_feats,
        dist_feats,
        freq_feats,
        info_feats
    ], dim=1)

    return features


# ============================================================================
# Advanced Unsupervised Representation Learning Methods
# ============================================================================

class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for learning latent representations

    Motivation:
    - VAEs learn a probabilistic latent space that captures the underlying
      distribution of weight matrices
    - The latent space is regularized to be smooth and well-structured
    - Disentangled representations may separate different aspects of concepts
    """
    def __init__(self, input_dim, latent_dim=64, hidden_dim=512):
        super(VariationalAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def get_latent(self, x):
        """Get latent representation for use as features"""
        mu, _ = self.encode(x)
        return mu


class ContrastiveAutoencoder(nn.Module):
    """
    Contrastive Autoencoder using SimCLR-inspired contrastive loss

    Motivation:
    - Contrastive learning creates representations where similar samples
      are close and dissimilar samples are far apart
    - Learns robust features without requiring labels
    - Encourages learning of invariant and discriminative features
    """
    def __init__(self, input_dim, latent_dim=128, hidden_dim=512, projection_dim=64):
        super(ContrastiveAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Projection head (for contrastive learning)
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        proj = self.projection(z)
        recon = self.decoder(z)
        return recon, z, proj

    def get_latent(self, x):
        """Get latent representation for use as features"""
        return self.encoder(x)


class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder with added corruption

    Motivation:
    - Forces the model to learn robust representations by reconstructing
      from corrupted inputs
    - Learns features that capture the essential structure of the data
    - Particularly useful for noisy or redundant input dimensions
    """
    def __init__(self, input_dim, latent_dim=64, hidden_dim=512, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__()

        self.noise_factor = noise_factor

        # Encoder with skip connections
        self.encoder_1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.encoder_2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.encoder_3 = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder_1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.decoder_2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.decoder_3 = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise

    def encode(self, x):
        h1 = self.relu(self.bn1(self.encoder_1(x)))
        h1 = self.dropout(h1)
        h2 = self.relu(self.bn2(self.encoder_2(h1)))
        h2 = self.dropout(h2)
        z = self.encoder_3(h2)
        return z

    def decode(self, z):
        h1 = self.relu(self.bn3(self.decoder_1(z)))
        h2 = self.relu(self.bn4(self.decoder_2(h1)))
        recon = self.decoder_3(h2)
        return recon

    def forward(self, x):
        x_noisy = self.add_noise(x) if self.training else x
        z = self.encode(x_noisy)
        recon = self.decode(z)
        return recon, z

    def get_latent(self, x):
        """Get latent representation for use as features"""
        return self.encode(x)


def train_autoencoder(model, train_loader, val_loader, epochs=100, lr=0.001, model_type='vae'):
    """
    Train an autoencoder model

    Args:
        model: Autoencoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        model_type: Type of autoencoder ('vae', 'contrastive', 'denoising')

    Returns:
        Trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()

            if model_type == 'vae':
                recon, mu, logvar = model(batch_x)
                # VAE loss: reconstruction + KL divergence
                recon_loss = nn.functional.mse_loss(recon, batch_x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.1 * kl_loss  # Beta=0.1 for beta-VAE

            elif model_type == 'contrastive':
                recon, z, proj = model(batch_x)
                # Reconstruction loss
                recon_loss = nn.functional.mse_loss(recon, batch_x)

                # Contrastive loss (simplified, using within-batch negatives)
                # Normalize projections
                proj_norm = nn.functional.normalize(proj, dim=1)
                # Compute similarity matrix
                sim_matrix = torch.matmul(proj_norm, proj_norm.T)
                # Create labels (diagonal should be high)
                labels = torch.arange(batch_x.size(0), device=device)
                # InfoNCE-style loss
                contrastive_loss = nn.functional.cross_entropy(sim_matrix / 0.5, labels)

                loss = recon_loss + 0.5 * contrastive_loss

            elif model_type == 'denoising':
                recon, z = model(batch_x)
                # Simple reconstruction loss
                loss = nn.functional.mse_loss(recon, batch_x)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)

                if model_type == 'vae':
                    recon, mu, logvar = model(batch_x)
                    recon_loss = nn.functional.mse_loss(recon, batch_x, reduction='sum')
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + 0.1 * kl_loss
                elif model_type == 'contrastive':
                    recon, z, proj = model(batch_x)
                    loss = nn.functional.mse_loss(recon, batch_x)
                elif model_type == 'denoising':
                    recon, z = model(batch_x)
                    loss = nn.functional.mse_loss(recon, batch_x)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    model.load_state_dict(best_model_state)
    return model


def vae_representation(layer_tensor: torch.Tensor, pretrained_vae=None, train_data=None, val_data=None) -> torch.Tensor:
    """
    Learn VAE representation from LoRA weights

    Args:
        layer_tensor: Input layer weights
        pretrained_vae: Pre-trained VAE model (if available)
        train_data: Training data for fitting VAE (if pretrained_vae is None)
        val_data: Validation data for fitting VAE (if pretrained_vae is None)

    Returns:
        Latent representations from VAE
    """
    # Flatten input
    X = layer_tensor.reshape(layer_tensor.size(0), -1).to(device)

    if pretrained_vae is None and train_data is not None:
        # Train a new VAE
        print("Training VAE...")
        input_dim = X.size(1)
        latent_dim = min(64, input_dim // 10)  # Adaptive latent dimension

        vae = VariationalAutoencoder(input_dim, latent_dim=latent_dim).to(device)

        # Create data loaders
        train_dataset = TensorDataset(train_data, torch.zeros(train_data.size(0)))
        val_dataset = TensorDataset(val_data, torch.zeros(val_data.size(0)))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        vae = train_autoencoder(vae, train_loader, val_loader, epochs=100, model_type='vae')
        pretrained_vae = vae

    # Extract latent representations
    pretrained_vae.eval()
    with torch.no_grad():
        features = pretrained_vae.get_latent(X)

    return features


def contrastive_representation(layer_tensor: torch.Tensor, pretrained_model=None, train_data=None, val_data=None) -> torch.Tensor:
    """
    Learn contrastive representation from LoRA weights

    Motivation:
    - Contrastive learning naturally creates discriminative features
    - Learns to separate different concepts in latent space
    - Robust to noise and invariant to irrelevant variations
    """
    # Flatten input
    X = layer_tensor.reshape(layer_tensor.size(0), -1).to(device)

    if pretrained_model is None and train_data is not None:
        print("Training Contrastive Autoencoder...")
        input_dim = X.size(1)
        latent_dim = min(128, input_dim // 5)

        model = ContrastiveAutoencoder(input_dim, latent_dim=latent_dim).to(device)

        train_dataset = TensorDataset(train_data, torch.zeros(train_data.size(0)))
        val_dataset = TensorDataset(val_data, torch.zeros(val_data.size(0)))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = train_autoencoder(model, train_loader, val_loader, epochs=100, model_type='contrastive')
        pretrained_model = model

    pretrained_model.eval()
    with torch.no_grad():
        features = pretrained_model.get_latent(X)

    return features


def denoising_representation(layer_tensor: torch.Tensor, pretrained_model=None, train_data=None, val_data=None) -> torch.Tensor:
    """
    Learn denoising autoencoder representation from LoRA weights

    Motivation:
    - Denoising forces learning of robust, essential features
    - Removes noise and captures underlying structure
    - Particularly useful for weight spaces which may have redundancy
    """
    # Flatten input
    X = layer_tensor.reshape(layer_tensor.size(0), -1).to(device)

    if pretrained_model is None and train_data is not None:
        print("Training Denoising Autoencoder...")
        input_dim = X.size(1)
        latent_dim = min(64, input_dim // 10)

        model = DenoisingAutoencoder(input_dim, latent_dim=latent_dim).to(device)

        train_dataset = TensorDataset(train_data, torch.zeros(train_data.size(0)))
        val_dataset = TensorDataset(val_data, torch.zeros(val_data.size(0)))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = train_autoencoder(model, train_loader, val_loader, epochs=100, model_type='denoising')
        pretrained_model = model

    pretrained_model.eval()
    with torch.no_grad():
        features = pretrained_model.get_latent(X)

    return features


# ============================================================================
# Existing Representation Methods (from experiment.py)
# ============================================================================

def flat_vec_representation(layer_tensor: torch.Tensor) -> torch.Tensor:
    """Flatten weight matrices into 1D vectors"""
    return layer_tensor.reshape(layer_tensor.size(0), -1).to(device)


def stats_representation(layer_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute statistical features: mean, std, median, min, max, skewness, kurtosis
    """
    batch_size = layer_tensor.size(0)
    features_list = []

    for i in range(batch_size):
        matrix = layer_tensor[i].flatten()

        mean = matrix.mean()
        std = matrix.std()
        median = matrix.median()
        min_val = matrix.min()
        max_val = matrix.max()

        # Skewness and kurtosis
        mean_val = mean.item()
        std_val = std.item()
        skewness = torch.mean(((matrix - mean_val) / (std_val + 1e-8)) ** 3)
        kurtosis = torch.mean(((matrix - mean_val) / (std_val + 1e-8)) ** 4)

        features = torch.tensor([
            mean, std, median, min_val, max_val, skewness, kurtosis
        ], device=device)

        features_list.append(features)

    return torch.stack(features_list)


# ============================================================================
# Layer Evaluation and Ablation
# ============================================================================

def evaluate_single_layer(
    train_data, val_data, test_data,
    train_labels, val_labels, test_labels,
    layer_name: str,
    representation_fn: Callable,
    num_classes: int,
    num_runs: int = 5,
    epochs: int = 500,
    patience: int = 20
) -> Dict[str, float]:
    """
    Evaluate a single layer's predictive power

    Returns:
        Dictionary with validation and test accuracies (mean and std)
    """
    val_accs = []
    test_accs = []

    for run in range(num_runs):
        # Set random seed
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)

        # Extract layer data (convert lists to tensors)
        train_layer = torch.stack([torch.tensor(sample[layer_name], dtype=torch.float32) for sample in train_data])
        val_layer = torch.stack([torch.tensor(sample[layer_name], dtype=torch.float32) for sample in val_data])
        test_layer = torch.stack([torch.tensor(sample[layer_name], dtype=torch.float32) for sample in test_data])

        # Apply representation
        X_train = representation_fn(train_layer)
        X_val = representation_fn(val_layer)
        X_test = representation_fn(test_layer)

        # Normalize
        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train.cpu().numpy()), device=device, dtype=torch.float32)
        X_val = torch.tensor(scaler.transform(X_val.cpu().numpy()), device=device, dtype=torch.float32)
        X_test = torch.tensor(scaler.transform(X_test.cpu().numpy()), device=device, dtype=torch.float32)

        # Create data loaders
        train_dataset = TensorDataset(X_train, train_labels)
        val_dataset = TensorDataset(X_val, val_labels)
        test_dataset = TensorDataset(X_test, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Create model
        model = MLP(X_train.size(1), num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

        # Training loop
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_preds = []
            val_true = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    preds = torch.argmax(outputs, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(y_batch.cpu().numpy())

            val_acc = accuracy_score(val_true, val_preds)
            scheduler.step(val_acc)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Load best model and evaluate on test
        model.load_state_dict(best_model_state)
        model.eval()
        test_preds = []
        test_true = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_true.extend(y_batch.cpu().numpy())

        test_acc = accuracy_score(test_true, test_preds)

        val_accs.append(best_val_acc)
        test_accs.append(test_acc)

    return {
        'val_mean': np.mean(val_accs),
        'val_std': np.std(val_accs),
        'test_mean': np.mean(test_accs),
        'test_std': np.std(test_accs)
    }


def identify_top_layers(
    train_data, val_data, test_data,
    train_labels, val_labels, test_labels,
    lora_layers: List[str],
    representation_fn: Callable,
    num_classes: int,
    top_k: int = 5,
    num_runs: int = 3
) -> List[Tuple[str, float]]:
    """
    Identify the top-k most predictive layers

    Returns:
        List of (layer_name, test_accuracy) tuples, sorted by accuracy
    """
    print(f"\nIdentifying top {top_k} most predictive layers...")
    layer_results = []

    for layer_name in tqdm(lora_layers, desc="Evaluating layers"):
        results = evaluate_single_layer(
            train_data, val_data, test_data,
            train_labels, val_labels, test_labels,
            layer_name,
            representation_fn,
            num_classes,
            num_runs=num_runs
        )

        layer_results.append((layer_name, results['test_mean']))
        print(f"{layer_name}: Val={results['val_mean']:.4f}±{results['val_std']:.4f}, Test={results['test_mean']:.4f}±{results['test_std']:.4f}")

    # Sort by test accuracy
    layer_results.sort(key=lambda x: x[1], reverse=True)

    return layer_results[:top_k]


def evaluate_with_ablation(
    train_data, val_data, test_data,
    train_labels, val_labels, test_labels,
    lora_layers: List[str],
    ablated_layers: List[str],
    representation_fn: Callable,
    num_classes: int,
    representation_name: str,
    num_runs: int = 10
) -> Dict[str, any]:
    """
    Evaluate a representation method with specified layers ablated (removed)

    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    print(f"\nEvaluating {representation_name} with {len(ablated_layers)} layers ablated...")

    # Filter out ablated layers
    remaining_layers = [l for l in lora_layers if l not in ablated_layers]
    print(f"Using {len(remaining_layers)} out of {len(lora_layers)} layers")

    val_accs = []
    test_accs = []
    val_f1s = []
    test_f1s = []

    for run in range(num_runs):
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)

        # Extract and concatenate remaining layers
        train_features_list = []
        val_features_list = []
        test_features_list = []

        for layer_name in remaining_layers:
            train_layer = torch.stack([torch.tensor(sample[layer_name], dtype=torch.float32) for sample in train_data])
            val_layer = torch.stack([torch.tensor(sample[layer_name], dtype=torch.float32) for sample in val_data])
            test_layer = torch.stack([torch.tensor(sample[layer_name], dtype=torch.float32) for sample in test_data])

            train_features_list.append(representation_fn(train_layer))
            val_features_list.append(representation_fn(val_layer))
            test_features_list.append(representation_fn(test_layer))

        # Concatenate features from all remaining layers
        X_train = torch.cat(train_features_list, dim=1)
        X_val = torch.cat(val_features_list, dim=1)
        X_test = torch.cat(test_features_list, dim=1)

        # Normalize
        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train.cpu().numpy()), device=device, dtype=torch.float32)
        X_val = torch.tensor(scaler.transform(X_val.cpu().numpy()), device=device, dtype=torch.float32)
        X_test = torch.tensor(scaler.transform(X_test.cpu().numpy()), device=device, dtype=torch.float32)

        # Create data loaders
        train_dataset = TensorDataset(X_train, train_labels)
        val_dataset = TensorDataset(X_val, val_labels)
        test_dataset = TensorDataset(X_test, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Create model
        model = MLP(X_train.size(1), num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

        # Training loop
        best_val_acc = 0
        patience_counter = 0
        epochs = 1000
        patience = 20

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_preds = []
            val_true = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    preds = torch.argmax(outputs, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(y_batch.cpu().numpy())

            val_acc = accuracy_score(val_true, val_preds)
            scheduler.step(val_acc)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Load best model and evaluate
        model.load_state_dict(best_model_state)
        model.eval()

        # Validation metrics
        val_preds = []
        val_true = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='macro')

        # Test metrics
        test_preds = []
        test_true = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_true.extend(y_batch.cpu().numpy())

        test_acc = accuracy_score(test_true, test_preds)
        test_f1 = f1_score(test_true, test_preds, average='macro')

        val_accs.append(val_acc)
        test_accs.append(test_acc)
        val_f1s.append(val_f1)
        test_f1s.append(test_f1)

    results = {
        'representation': representation_name,
        'num_layers': len(remaining_layers),
        'num_ablated': len(ablated_layers),
        'val_acc_mean': np.mean(val_accs),
        'val_acc_std': np.std(val_accs),
        'test_acc_mean': np.mean(test_accs),
        'test_acc_std': np.std(test_accs),
        'val_f1_mean': np.mean(val_f1s),
        'val_f1_std': np.std(val_f1s),
        'test_f1_mean': np.mean(test_f1s),
        'test_f1_std': np.std(test_f1s),
    }

    print(f"Results: Val Acc={results['val_acc_mean']:.4f}±{results['val_acc_std']:.4f}, "
          f"Test Acc={results['test_acc_mean']:.4f}±{results['test_acc_std']:.4f}")

    return results


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ablation experiment for LoRA representations')
    parser.add_argument('--dataset', type=str, default='1k', choices=['1k', '10k', '50k', '100k', 'small'],
                       help='Dataset size to use')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top predictive layers to ablate')
    parser.add_argument('--num-runs', type=int, default=10,
                       help='Number of training runs for averaging')
    parser.add_argument('--output-dir', type=str, default='ablation-results',
                       help='Directory to save results')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer runs and epochs')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'ablation_{args.dataset}')
    os.makedirs(output_path, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    train_data, val_data, test_data = load_and_split_dataset(args.dataset)

    # Get labels
    train_labels = torch.tensor([sample['leaf_id'] for sample in train_data], device=device)
    val_labels = torch.tensor([sample['leaf_id'] for sample in val_data], device=device)
    test_labels = torch.tensor([sample['leaf_id'] for sample in test_data], device=device)

    num_classes = len(set(train_labels.cpu().numpy()))
    print(f"Number of classes: {num_classes}")

    # Get LoRA layers
    lora_layers = get_lora_layers(train_data)
    print(f"Number of LoRA layers: {len(lora_layers)}")

    # Define representation methods to test
    representations = {
        'flat_vec': flat_vec_representation,
        'stats': stats_representation,
        'spectral': spectral_features,
        'matrix_norms': matrix_norm_features,
        'distribution': distribution_features,
        'frequency': frequency_features,
        'info_theoretic': information_theoretic_features,
        'ensemble': ensemble_features,
    }

    num_runs = 3 if args.quick else args.num_runs

    # Step 1: Identify top-k most predictive layers using stats representation
    print("\n" + "="*80)
    print("STEP 1: Identifying most predictive layers")
    print("="*80)

    top_layers = identify_top_layers(
        train_data, val_data, test_data,
        train_labels, val_labels, test_labels,
        lora_layers,
        stats_representation,  # Use stats as baseline
        num_classes,
        top_k=args.top_k,
        num_runs=3  # Fewer runs for layer identification
    )

    print(f"\nTop {args.top_k} most predictive layers:")
    for i, (layer_name, acc) in enumerate(top_layers, 1):
        print(f"{i}. {layer_name}: {acc:.4f}")

    # Save top layers
    with open(os.path.join(output_path, 'top_layers.json'), 'w') as f:
        json.dump([{'layer': layer, 'accuracy': float(acc)} for layer, acc in top_layers], f, indent=2)

    # Step 2: Evaluate all representations with top layers ablated
    print("\n" + "="*80)
    print("STEP 2: Evaluating representations with ablation")
    print("="*80)

    ablated_layers = [layer for layer, _ in top_layers]

    all_results = []

    for rep_name, rep_fn in representations.items():
        try:
            results = evaluate_with_ablation(
                train_data, val_data, test_data,
                train_labels, val_labels, test_labels,
                lora_layers,
                ablated_layers,
                rep_fn,
                num_classes,
                rep_name,
                num_runs=num_runs
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error evaluating {rep_name}: {e}")
            import traceback
            traceback.print_exc()

    # Step 3: Also evaluate without ablation for comparison
    print("\n" + "="*80)
    print("STEP 3: Evaluating representations WITHOUT ablation (baseline)")
    print("="*80)

    baseline_results = []

    for rep_name, rep_fn in representations.items():
        try:
            results = evaluate_with_ablation(
                train_data, val_data, test_data,
                train_labels, val_labels, test_labels,
                lora_layers,
                [],  # No ablation
                rep_fn,
                num_classes,
                rep_name + '_baseline',
                num_runs=num_runs
            )
            baseline_results.append(results)
        except Exception as e:
            print(f"Error evaluating {rep_name} baseline: {e}")
            import traceback
            traceback.print_exc()

    # Step 4: Save and visualize results
    print("\n" + "="*80)
    print("STEP 4: Saving results")
    print("="*80)

    # Save results to JSON
    all_results_dict = {
        'dataset': args.dataset,
        'top_k_ablated': args.top_k,
        'num_runs': num_runs,
        'top_layers': [layer for layer, _ in top_layers],
        'ablation_results': all_results,
        'baseline_results': baseline_results
    }

    with open(os.path.join(output_path, 'results.json'), 'w') as f:
        json.dump(all_results_dict, f, indent=2)

    # Create comparison table
    with open(os.path.join(output_path, 'results.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Ablation Experiment Results - Dataset: {args.dataset}\n")
        f.write("="*80 + "\n\n")

        f.write(f"Top {args.top_k} ablated layers:\n")
        for i, (layer, acc) in enumerate(top_layers, 1):
            f.write(f"{i}. {layer}: {acc:.4f}\n")
        f.write("\n")

        f.write("-"*80 + "\n")
        f.write("Results WITH ablation:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Representation':<20} {'Val Acc':<20} {'Test Acc':<20} {'Test F1':<20}\n")
        for res in all_results:
            f.write(f"{res['representation']:<20} "
                   f"{res['val_acc_mean']:.4f}±{res['val_acc_std']:.4f}     "
                   f"{res['test_acc_mean']:.4f}±{res['test_acc_std']:.4f}     "
                   f"{res['test_f1_mean']:.4f}±{res['test_f1_std']:.4f}\n")

        f.write("\n")
        f.write("-"*80 + "\n")
        f.write("Results WITHOUT ablation (baseline):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Representation':<20} {'Val Acc':<20} {'Test Acc':<20} {'Test F1':<20}\n")
        for res in baseline_results:
            f.write(f"{res['representation']:<20} "
                   f"{res['val_acc_mean']:.4f}±{res['val_acc_std']:.4f}     "
                   f"{res['test_acc_mean']:.4f}±{res['test_acc_std']:.4f}     "
                   f"{res['test_f1_mean']:.4f}±{res['test_f1_std']:.4f}\n")

    print(f"\nResults saved to {output_path}/")
    print("Experiment complete!")


if __name__ == '__main__':
    main()
