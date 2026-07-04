"""
Node 3 -- Phase 3: Train the Context Autoencoder

Trains the ContextAutoencoder on the 6-dim telemetry columns from the
synthetic training data. After training:
    1. Saves the full autoencoder   -> node3/models/context_autoencoder.pth
    2. Extracts and saves encoder   -> node3/models/context_encoder.pth
    3. Prints validation reconstruction loss

The encoder is then used as a frozen feature extractor in Phase 4 (MLP training).

Training config:
    Loss:       MSE
    Optimizer:  Adam (lr=1e-3)
    Batch size: 64
    Epochs:     100 (early stopping, patience=10)
    Input:      6-dim normalized telemetry
    Bottleneck: 128-dim

Usage:
    python node3/train_autoencoder.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from node3.models import ContextAutoencoder, ContextEncoder

# =========================================================================
# Config
# =========================================================================

DATA_PATH = "data/synthetic_training_data.csv"
MODEL_DIR = "node3/models"
AUTOENCODER_PATH = os.path.join(MODEL_DIR, "context_autoencoder.pth")
ENCODER_PATH = os.path.join(MODEL_DIR, "context_encoder.pth")

# Telemetry columns (first 6 in the CSV)
TELEMETRY_COLS = ["speed_norm", "lat_norm", "lon_norm", "time_sin", "time_cos", "v_rel_norm"]

# Hyperparameters
INPUT_DIM = 6
BOTTLENECK_DIM = 128
LEARNING_RATE = 3e-3
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 15           # Early stopping patience
TRAIN_SPLIT = 0.8       # 80% train, 20% validation
RANDOM_SEED = 42


# =========================================================================
# Data Loading
# =========================================================================

def load_telemetry_data(csv_path: str):
    """
    Load and split telemetry columns from the synthetic dataset.

    Returns:
        (train_tensor, val_tensor): both of shape (N, 6), dtype float32
    """
    import csv

    # Read CSV manually (avoids pandas dependency for lightweight edge deployment)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Extract telemetry columns
    data = []
    for row in rows:
        sample = [float(row[col]) for col in TELEMETRY_COLS]
        data.append(sample)

    data = np.array(data, dtype=np.float32)

    # Shuffle and split
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(data))
    split_idx = int(len(data) * TRAIN_SPLIT)

    train_data = data[indices[:split_idx]]
    val_data = data[indices[split_idx:]]

    return torch.tensor(train_data), torch.tensor(val_data)


# =========================================================================
# Training Loop
# =========================================================================

def train():
    """Train the Context Autoencoder with early stopping."""

    print("=" * 60)
    print("  Node 3 -- Phase 3: Context Autoencoder Training")
    print("=" * 60)

    # Setup
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Load data
    print(f"  Loading data from: {DATA_PATH}")
    train_data, val_data = load_telemetry_data(DATA_PATH)
    print(f"  Train samples: {len(train_data):,}")
    print(f"  Val samples:   {len(val_data):,}")

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Model
    model = ContextAutoencoder(input_dim=INPUT_DIM, bottleneck_dim=BOTTLENECK_DIM)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"\n  Model Parameters:")
    print(f"    Total:   {total_params:,}")
    print(f"    Encoder: {encoder_params:,}")
    print(f"    Decoder: {decoder_params:,}")

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Early stopping state
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    print(f"\n  Training config:")
    print(f"    Loss:       MSE")
    print(f"    Optimizer:  Adam (lr={LEARNING_RATE})")
    print(f"    Batch size: {BATCH_SIZE}")
    print(f"    Max epochs: {MAX_EPOCHS}")
    print(f"    Patience:   {PATIENCE}")
    print(f"\n  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'Status':>10}")
    print(f"  {'-' * 48}")

    for epoch in range(1, MAX_EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            x_hat = model(batch_x)
            loss = criterion(x_hat, batch_x)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / train_batches

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                x_hat = model(batch_x)
                loss = criterion(x_hat, batch_x)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches

        # --- Early Stopping Check ---
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            status = "* BEST"

            # Save best model
            torch.save(model.state_dict(), AUTOENCODER_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                status = "STOP"
            else:
                status = f"wait {patience_counter}/{PATIENCE}"

        # Print progress (every 5 epochs, or first, or best, or last)
        if epoch <= 3 or epoch % 5 == 0 or status == "* BEST" or status == "STOP":
            print(f"  {epoch:>6}  {avg_train_loss:>12.6f}  {avg_val_loss:>12.6f}  {status:>10}")

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            print(f"  Best epoch: {best_epoch} with val loss: {best_val_loss:.6f}")
            break

    # =========================================================================
    # Extract Encoder
    # =========================================================================

    print(f"\n  Loading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(AUTOENCODER_PATH, weights_only=True))
    model.eval()

    # Extract encoder-only model
    encoder = ContextEncoder.from_autoencoder(model)
    encoder.eval()
    torch.save(encoder.state_dict(), ENCODER_PATH)

    print(f"  Saved full autoencoder -> {AUTOENCODER_PATH}")
    print(f"  Saved encoder-only    -> {ENCODER_PATH}")

    # =========================================================================
    # Validation: Reconstruction Quality
    # =========================================================================

    print(f"\n  Validation Reconstruction Analysis:")

    model.to(device)
    all_errors = []

    with torch.no_grad():
        for (batch_x,) in val_loader:
            batch_x = batch_x.to(device)
            x_hat = model(batch_x)
            errors = torch.abs(batch_x - x_hat)  # Per-element absolute error
            all_errors.append(errors.cpu().numpy())

    all_errors = np.concatenate(all_errors, axis=0)
    mean_errors = all_errors.mean(axis=0)

    print(f"    Overall MSE:          {best_val_loss:.6f}")
    print(f"    Overall MAE:          {all_errors.mean():.6f}")
    print(f"    Max single error:     {all_errors.max():.6f}")
    print(f"\n    Per-feature MAE:")
    for i, col in enumerate(TELEMETRY_COLS):
        print(f"      {col:<15}  {mean_errors[i]:.6f}")

    # =========================================================================
    # Validation: Embedding Diversity
    # =========================================================================

    print(f"\n  Embedding Diversity Check:")

    encoder.to(device)
    all_embeddings = []

    with torch.no_grad():
        for (batch_x,) in val_loader:
            batch_x = batch_x.to(device)
            emb = encoder(batch_x)
            all_embeddings.append(emb.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    emb_mean = all_embeddings.mean()
    emb_std = all_embeddings.std()
    emb_min = all_embeddings.min()
    emb_max = all_embeddings.max()

    # Check that embeddings aren't collapsed (all same)
    sample_cosine_sims = []
    for _ in range(100):
        i, j = np.random.choice(len(all_embeddings), 2, replace=False)
        a, b = all_embeddings[i], all_embeddings[j]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        sample_cosine_sims.append(cos_sim)

    avg_cos_sim = np.mean(sample_cosine_sims)

    print(f"    Embedding shape:      (N, {all_embeddings.shape[1]})")
    print(f"    Mean value:           {emb_mean:.4f}")
    print(f"    Std deviation:        {emb_std:.4f}")
    print(f"    Range:                [{emb_min:.4f}, {emb_max:.4f}]")
    print(f"    Avg cosine sim (100 random pairs): {avg_cos_sim:.4f}")

    if avg_cos_sim > 0.99:
        print(f"    [!] WARNING: Embeddings may have collapsed (too similar)")
    else:
        print(f"    [OK] Embeddings show healthy diversity")

    # =========================================================================
    # Final Summary
    # =========================================================================

    # Plan §3.4: target validation MSE < 0.01 (inputs are [0,1] normalized,
    # meaning <1% reconstruction error).
    success = best_val_loss < 0.01
    print(f"\n{'=' * 60}")
    print(f"  Phase 3 Training Complete!")
    print(f"{'=' * 60}")
    print(f"  Best val MSE:    {best_val_loss:.6f}  {'PASS (<0.01)' if success else 'FAIL (>=0.01)'}")
    print(f"  Best epoch:      {best_epoch}")
    print(f"  Encoder output:  {ENCODER_PATH}")
    print(f"{'=' * 60}\n")

    return success


if __name__ == "__main__":
    success = train()
    sys.exit(0 if success else 1)
