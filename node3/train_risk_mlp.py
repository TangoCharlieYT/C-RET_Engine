"""
Node 3 -- Phase 4: Train the Safety Risk MLP

Trains the SafetyRiskMLP on the synthetic training data using a *frozen*
ContextEncoder as the feature extractor for the telemetry portion.

Pipeline per sample:
    1. Pass the 6-dim telemetry vector through the frozen encoder  -> (128,)
    2. Concatenate with detection confidences (5), top label one-hot (4),
       and anomaly flags + crash_prob (3).
    3. Feed the 140-dim vector into SafetyRiskMLP and train against
       the risk_label target with BCELoss.

After training:
    - Saves node3/models/safety_risk_mlp.pth
    - Prints validation MSE, MAE, accuracy@0.5, accuracy@0.15

Training config (per Plan §4.2):
    Loss:       BCELoss
    Optimizer:  Adam (lr=5e-4)
    Batch size: 64
    Epochs:     150 with early stopping (patience=15)
    Dropout:    0.3 / 0.2
    Encoder:    Frozen (requires_grad=False)

Usage:
    python node3/train_risk_mlp.py
"""

import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from node3.models import ContextEncoder, SafetyRiskMLP

# =========================================================================
# Config
# =========================================================================

DATA_PATH = "data/synthetic_training_data.csv"
MODEL_DIR = "node3/models"
ENCODER_PATH = os.path.join(MODEL_DIR, "context_encoder.pth")
MLP_PATH = os.path.join(MODEL_DIR, "safety_risk_mlp.pth")

# CSV columns
TELEMETRY_COLS = ["speed_norm", "lat_norm", "lon_norm", "time_sin", "time_cos", "v_rel_norm"]
DETECTION_COLS = ["conf_1", "conf_2", "conf_3", "conf_4", "conf_5"]
LABEL_COLS = ["label_car", "label_person", "label_truck", "label_other"]
ANOMALY_COLS = ["low_conf_flag", "glare_flag", "crash_prob"]
TARGET_COL = "risk_label"

# Hyperparameters
INPUT_DIM_TELEMETRY = 6
BOTTLENECK_DIM = 128
MLP_INPUT_DIM = 140  # 128 + 5 + 4 + 3
LEARNING_RATE = 5e-4
BATCH_SIZE = 64
MAX_EPOCHS = 150
PATIENCE = 15
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# Accuracy thresholds for reporting
ACC_THRESHOLDS = [0.10, 0.15, 0.20, 0.25]


# =========================================================================
# Data Loading
# =========================================================================

def load_dataset(csv_path: str):
    """
    Load and split the synthetic dataset into train/val tensors.

    Returns:
        (train_telemetry, train_mlp_extra, train_target,
         val_telemetry,   val_mlp_extra,   val_target)

        train_telemetry: (N_train, 6)   float32
        train_mlp_extra: (N_train, 12)  float32  -- [5 confs + 4 label + 3 anomaly]
        train_target:    (N_train, 1)   float32

        val_* analogous with N_val = 2000
    """
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    telemetry = np.array(
        [[float(r[c]) for c in TELEMETRY_COLS] for r in rows], dtype=np.float32
    )
    detection = np.array(
        [[float(r[c]) for c in DETECTION_COLS] for r in rows], dtype=np.float32
    )
    label_oh = np.array(
        [[float(r[c]) for c in LABEL_COLS] for r in rows], dtype=np.float32
    )
    anomaly = np.array(
        [[float(r[c]) for c in ANOMALY_COLS] for r in rows], dtype=np.float32
    )
    target = np.array(
        [[float(r[TARGET_COL])] for r in rows], dtype=np.float32
    )

    mlp_extra = np.concatenate([detection, label_oh, anomaly], axis=1)  # (N, 12)
    assert mlp_extra.shape[1] == 12, f"Expected 12 MLP-extra cols, got {mlp_extra.shape[1]}"

    # Shuffle + split
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(rows))
    split_idx = int(len(rows) * TRAIN_SPLIT)

    tr = indices[:split_idx]
    va = indices[split_idx:]

    return (
        torch.tensor(telemetry[tr]),
        torch.tensor(mlp_extra[tr]),
        torch.tensor(target[tr]),
        torch.tensor(telemetry[va]),
        torch.tensor(mlp_extra[va]),
        torch.tensor(target[va]),
    )


# =========================================================================
# Helper: Build full MLP input by running frozen encoder over telemetry
# =========================================================================

def build_mlp_inputs(encoder: ContextEncoder, telemetry: torch.Tensor, extra: torch.Tensor, device) -> torch.Tensor:
    """
    Run the (frozen) ContextEncoder over telemetry, then concat with extra.

    Args:
        encoder:   frozen ContextEncoder (eval mode)
        telemetry: (N, 6)
        extra:     (N, 12)  -- detection + label + anomaly
        device:    torch.device

    Returns:
        (N, 140) tensor
    """
    encoder.eval()
    with torch.no_grad():
        telemetry = telemetry.to(device)
        embeddings = encoder(telemetry)            # (N, 128)
        extra = extra.to(device)
        mlp_input = torch.cat([embeddings, extra], dim=1)
    return mlp_input


# =========================================================================
# Training
# =========================================================================

def train():
    """Train the Safety Risk MLP with a frozen ContextEncoder front-end."""

    print("=" * 60)
    print("  Node 3 -- Phase 4: Safety Risk MLP Training")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ------------------------------------------------------------------
    # Load frozen encoder
    # ------------------------------------------------------------------
    if not os.path.exists(ENCODER_PATH):
        print(f"  [!] Encoder not found at {ENCODER_PATH}. Run train_autoencoder.py first.")
        sys.exit(1)

    encoder = ContextEncoder(input_dim=INPUT_DIM_TELEMETRY, bottleneck_dim=BOTTLENECK_DIM)
    encoder.load_state_dict(torch.load(ENCODER_PATH, weights_only=True))
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"  Loaded frozen encoder from {ENCODER_PATH}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"  Loading data from: {DATA_PATH}")
    (tr_tel, tr_extra, tr_y,
     va_tel, va_extra, va_y) = load_dataset(DATA_PATH)
    print(f"  Train samples: {len(tr_tel):,}")
    print(f"  Val samples:   {len(va_tel):,}")

    # Pre-compute embeddings (encoder is frozen)
    print("  Encoding telemetry through frozen encoder...")
    tr_X = build_mlp_inputs(encoder, tr_tel, tr_extra, device)
    va_X = build_mlp_inputs(encoder, va_tel, va_extra, device)
    assert tr_X.shape[1] == MLP_INPUT_DIM, f"Expected {MLP_INPUT_DIM}-dim MLP input, got {tr_X.shape[1]}"
    assert va_X.shape[1] == MLP_INPUT_DIM
    print(f"  MLP input shape: {tuple(tr_X.shape)}  (per plan: 128 + 5 + 4 + 3 = {MLP_INPUT_DIM})")

    tr_y = tr_y.to(device)
    va_y = va_y.to(device)

    train_loader = DataLoader(
        TensorDataset(tr_X, tr_y), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(va_X, va_y), batch_size=BATCH_SIZE, shuffle=False
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = SafetyRiskMLP(input_dim=MLP_INPUT_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model Parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Loss / optimizer
    # ------------------------------------------------------------------
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ------------------------------------------------------------------
    # Early-stopping state
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"\n  Training config:")
    print(f"    Loss:       BCELoss")
    print(f"    Optimizer:  Adam (lr={LEARNING_RATE})")
    print(f"    Batch size: {BATCH_SIZE}")
    print(f"    Max epochs: {MAX_EPOCHS}")
    print(f"    Patience:   {PATIENCE}")
    print(f"    Encoder:    FROZEN")
    print(f"\n  {'Epoch':>6}  {'Train BCE':>10}  {'Val BCE':>10}  {'Val MSE':>10}  {'Acc@0.5':>8}  {'Status':>10}")
    print(f"  {'-' * 64}")

    for epoch in range(1, MAX_EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / train_batches

        # ---- Validate ----
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                val_loss_sum += loss.item()
                val_batches += 1
                all_preds.append(pred.cpu())
                all_targets.append(batch_y.cpu())

        avg_val_loss = val_loss_sum / val_batches

        preds_cat = torch.cat(all_preds, dim=0).numpy().flatten()
        targets_cat = torch.cat(all_targets, dim=0).numpy().flatten()

        # Metrics
        val_mse = float(np.mean((preds_cat - targets_cat) ** 2))
        val_mae = float(np.mean(np.abs(preds_cat - targets_cat)))
        acc_at_05 = float(np.mean((preds_cat > 0.5) == (targets_cat > 0.5)))
        acc_at_015 = float(np.mean(np.abs(preds_cat - targets_cat) < 0.15))

        # ---- Early-stopping check ----
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            status = "* BEST"
            torch.save(model.state_dict(), MLP_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                status = "STOP"
            else:
                status = f"wait {patience_counter}/{PATIENCE}"

        # ---- Print progress ----
        if epoch <= 3 or epoch % 5 == 0 or status in ("* BEST", "STOP"):
            print(
                f"  {epoch:>6}  {avg_train_loss:>10.6f}  {avg_val_loss:>10.6f}  "
                f"{val_mse:>10.6f}  {acc_at_05:>8.4f}  {status:>10}"
            )

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            print(f"  Best epoch: {best_epoch} with val BCE: {best_val_loss:.6f}")
            break

    # ------------------------------------------------------------------
    # Final evaluation with best model
    # ------------------------------------------------------------------
    print(f"\n  Loading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(MLP_PATH, weights_only=True))
    model.eval()

    final_preds = []
    final_targets = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            pred = model(batch_X)
            final_preds.append(pred.cpu().numpy())
            final_targets.append(batch_y.cpu().numpy())

    final_preds = np.concatenate(final_preds, axis=0).flatten()
    final_targets = np.concatenate(final_targets, axis=0).flatten()

    final_mse = float(np.mean((final_preds - final_targets) ** 2))
    final_mae = float(np.mean(np.abs(final_preds - final_targets)))

    print(f"\n  Final Validation Metrics (best model):")
    print(f"    Val BCE:  {best_val_loss:.6f}")
    print(f"    Val MSE:  {final_mse:.6f}")
    print(f"    Val MAE:  {final_mae:.6f}")

    # Accuracy at multiple tolerance thresholds
    print(f"\n  Accuracy (predicted within tolerance):")
    for tol in ACC_THRESHOLDS:
        acc = float(np.mean(np.abs(final_preds - final_targets) < tol))
        print(f"    within +/-{tol:.2f}: {acc * 100:.2f}%")

    # Binary accuracy (above/below 0.5)
    bin_acc = float(np.mean((final_preds > 0.5) == (final_targets > 0.5)))
    print(f"    binary (threshold 0.5):       {bin_acc * 100:.2f}%")

    # Action-bucket accuracy
    print(f"\n  Per-Action-Bucket Accuracy (predicted matches target bucket):")
    bucket_edges = [0.25, 0.50, 0.70, 0.85]
    bucket_names = ['all_clear', 'stay_alert', 'reduce_speed', 'slow_down', 'brake_and_record']

    def bucket(v):
        for i, e in enumerate(bucket_edges):
            if v < e:
                return i
        return 4

    pred_buckets = np.array([bucket(p) for p in final_preds])
    targ_buckets = np.array([bucket(t) for t in final_targets])
    for i, name in enumerate(bucket_names):
        mask = targ_buckets == i
        if mask.sum() == 0:
            continue
        correct = (pred_buckets[mask] == i).sum()
        total = int(mask.sum())
        print(f"    {name:<20} {correct:>4}/{total:<4} ({100.0 * correct / total:>5.1f}%)")

    # ------------------------------------------------------------------
    # Success criterion (Plan §Verification: |pred - target| < 0.15 for >90%)
    # ------------------------------------------------------------------
    success = acc_at_015 >= 0.90
    print(f"\n{'=' * 60}")
    print(f"  Phase 4 Training Complete!")
    print(f"{'=' * 60}")
    print(f"  Best val BCE:           {best_val_loss:.6f}")
    print(f"  Best epoch:             {best_epoch}")
    print(f"  Model output:           {MLP_PATH}")
    print(f"  Accuracy +/-0.15:       {acc_at_015 * 100:.2f}%  "
          f"{'PASS (>=90%)' if success else 'FAIL (<90%)'}")
    print(f"{'=' * 60}\n")

    return success


if __name__ == "__main__":
    success = train()
    sys.exit(0 if success else 1)