"""
Node 3 — Neural Network Architectures

Contains the PyTorch model definitions for:
    1. ContextAutoencoder — learns to compress telemetry into a 128-dim embedding
    2. ContextEncoder     — encoder-only half (extracted after training)
    3. SafetyRiskMLP      — predicts risk score from context + perception data

These are training-time definitions. For inference, models are exported to ONNX
and run via onnxruntime-directml (Phase 5).
"""

import torch
import torch.nn as nn


class ContextAutoencoder(nn.Module):
    """
    Autoencoder that compresses 6-dim telemetry into a 128-dim bottleneck.

    Architecture:
        Encoder: 6 → 64 (ReLU) → 128 (ReLU) + BatchNorm
        Decoder: 128 → 64 (ReLU) → 6 (Sigmoid)

    After training, the decoder is discarded and only the encoder
    is used as a feature extractor for the Safety Risk MLP.

    Input:  (batch, 6)   — normalized telemetry vector
    Output: (batch, 6)   — reconstructed telemetry (training only)
    """

    def __init__(self, input_dim=6, bottleneck_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Tanh(),  # Output bounded [-1, 1] to cover both [0,1] and [-1,1] inputs
        )

    def forward(self, x):
        """Full forward pass (encoder + decoder) for training."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        """Encoder-only forward pass for inference."""
        return self.encoder(x)


class ContextEncoder(nn.Module):
    """
    Standalone encoder extracted from a trained ContextAutoencoder.

    This is the production model — takes 6-dim telemetry and outputs
    a 128-dim context embedding.

    Input:  (batch, 6)   — normalized telemetry vector
    Output: (batch, 128) — context embedding
    """

    def __init__(self, input_dim=6, bottleneck_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_dim),
        )

    def forward(self, x):
        return self.encoder(x)

    @staticmethod
    def from_autoencoder(autoencoder: ContextAutoencoder) -> "ContextEncoder":
        """
        Extract the encoder from a trained autoencoder.

        Copies the encoder weights from the autoencoder into a standalone
        ContextEncoder model.
        """
        encoder = ContextEncoder(
            input_dim=autoencoder.encoder[0].in_features,
            bottleneck_dim=autoencoder.encoder[2].out_features,
        )
        encoder.encoder.load_state_dict(autoencoder.encoder.state_dict())
        return encoder


class SafetyRiskMLP(nn.Module):
    """
    Multi-Layer Perceptron that predicts a risk score from:
        - Context embedding (128-dim, from frozen ContextEncoder)
        - YOLO detection confidences (5-dim)
        - Top label one-hot (4-dim)
        - Anomaly flags + crash_prob (3-dim)

    Total input: 128 + 5 + 4 + 3 = 140 dimensions

    Architecture:
        140 → 256 (ReLU, Dropout 0.3)
            → 128 (ReLU, Dropout 0.2)
            → 64  (ReLU)
            → 1   (Sigmoid)

    Input:  (batch, 140)
    Output: (batch, 1)  — risk score in [0.0, 1.0]
    """

    def __init__(self, input_dim=140):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)
