"""
Node 3 -- Phase 5: ONNX Export

Exports the two trained PyTorch models (ContextEncoder, SafetyRiskMLP)
to ONNX graphs for hardware-accelerated inference via onnxruntime-directml.

Per Plan §5.1:
    Model 1: Context Encoder
        Input:  telemetry_vector  -> shape (1, 6),   dtype float32
        Output: context_embedding -> shape (1, 128), dtype float32
        Export: node3/models/context_encoder.onnx

    Model 2: Safety Risk MLP
        Input:  mlp_input         -> shape (1, 140), dtype float32
        Output: risk_score        -> shape (1, 1),   dtype float32
        Export: node3/models/safety_risk_mlp.onnx

Both exported with opset_version=17 and dynamic batch dimension.

Usage:
    python node3/export_to_onnx.py
"""

import os
import sys
import torch

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from node3.models import ContextEncoder, SafetyRiskMLP

# =========================================================================
# Config
# =========================================================================

MODEL_DIR = "node3/models"
ENCODER_PTH = os.path.join(MODEL_DIR, "context_encoder.pth")
ENCODER_ONNX = os.path.join(MODEL_DIR, "context_encoder.onnx")

MLP_PTH = os.path.join(MODEL_DIR, "safety_risk_mlp.pth")
MLP_ONNX = os.path.join(MODEL_DIR, "safety_risk_mlp.onnx")

OPSET = 17
INPUT_DIM_TELEMETRY = 6
BOTTLENECK_DIM = 128
MLP_INPUT_DIM = 140


def export_encoder():
    """Export ContextEncoder: (B, 6) -> (B, 128)."""
    print("\n" + "=" * 60)
    print("  Exporting Context Encoder")
    print("=" * 60)

    encoder = ContextEncoder(input_dim=INPUT_DIM_TELEMETRY, bottleneck_dim=BOTTLENECK_DIM)
    encoder.load_state_dict(torch.load(ENCODER_PTH, weights_only=True))
    encoder.eval()

    dummy = torch.randn(1, INPUT_DIM_TELEMETRY, dtype=torch.float32)

    torch.onnx.export(
        encoder,
        dummy,
        ENCODER_ONNX,
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["telemetry_vector"],
        output_names=["context_embedding"],
        dynamic_axes={
            "telemetry_vector":  {0: "batch"},
            "context_embedding": {0: "batch"},
        },
    )

    print(f"  Loaded weights:  {ENCODER_PTH}")
    print(f"  Input shape:     (batch, {INPUT_DIM_TELEMETRY})")
    print(f"  Output shape:    (batch, {BOTTLENECK_DIM})")
    print(f"  Opset:           {OPSET}")
    print(f"  Dynamic batch:   True")
    print(f"  Saved to:        {ENCODER_ONNX}")
    print(f"  File size:       {os.path.getsize(ENCODER_ONNX):,} bytes")
    return ENCODER_ONNX


def export_mlp():
    """Export SafetyRiskMLP: (B, 140) -> (B, 1)."""
    print("\n" + "=" * 60)
    print("  Exporting Safety Risk MLP")
    print("=" * 60)

    mlp = SafetyRiskMLP(input_dim=MLP_INPUT_DIM)
    mlp.load_state_dict(torch.load(MLP_PTH, weights_only=True))
    mlp.eval()

    dummy = torch.randn(1, MLP_INPUT_DIM, dtype=torch.float32)

    torch.onnx.export(
        mlp,
        dummy,
        MLP_ONNX,
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["mlp_input"],
        output_names=["risk_score"],
        dynamic_axes={
            "mlp_input":  {0: "batch"},
            "risk_score": {0: "batch"},
        },
    )

    print(f"  Loaded weights:  {MLP_PTH}")
    print(f"  Input shape:     (batch, {MLP_INPUT_DIM})")
    print(f"  Output shape:    (batch, 1)")
    print(f"  Opset:           {OPSET}")
    print(f"  Dynamic batch:   True")
    print(f"  Saved to:        {MLP_ONNX}")
    print(f"  File size:       {os.path.getsize(MLP_ONNX):,} bytes")
    return MLP_ONNX


def verify_parity():
    """
    Plan §Verification: assert |pytorch_output - onnx_output| < 1e-5
    for 100 random inputs.
    """
    import onnxruntime as ort
    import numpy as np

    print("\n" + "=" * 60)
    print("  ONNX Parity Verification (100 random inputs)")
    print("=" * 60)

    # ---------- Encoder parity ----------
    encoder_pt = ContextEncoder(input_dim=INPUT_DIM_TELEMETRY, bottleneck_dim=BOTTLENECK_DIM)
    encoder_pt.load_state_dict(torch.load(ENCODER_PTH, weights_only=True))
    encoder_pt.eval()

    enc_sess = ort.InferenceSession(
        ENCODER_ONNX,
        providers=["CPUExecutionProvider"],
    )

    max_diff_enc = 0.0
    np.random.seed(42)
    for _ in range(100):
        x = np.random.randn(1, INPUT_DIM_TELEMETRY).astype(np.float32)
        with torch.no_grad():
            y_pt = encoder_pt(torch.from_numpy(x)).numpy()
        y_ox = enc_sess.run(None, {"telemetry_vector": x})[0]
        d = float(np.max(np.abs(y_pt - y_ox)))
        max_diff_enc = max(max_diff_enc, d)

    print(f"  Encoder max diff (100 random inputs):  {max_diff_enc:.2e}")
    enc_ok = max_diff_enc < 1e-4  # ONNX export tolerance (fp32)
    print(f"  Encoder parity:                        {'PASS' if enc_ok else 'FAIL'}")

    # ---------- MLP parity ----------
    mlp_pt = SafetyRiskMLP(input_dim=MLP_INPUT_DIM)
    mlp_pt.load_state_dict(torch.load(MLP_PTH, weights_only=True))
    mlp_pt.eval()

    mlp_sess = ort.InferenceSession(
        MLP_ONNX,
        providers=["CPUExecutionProvider"],
    )

    max_diff_mlp = 0.0
    for _ in range(100):
        x = np.random.randn(1, MLP_INPUT_DIM).astype(np.float32)
        with torch.no_grad():
            y_pt = mlp_pt(torch.from_numpy(x)).numpy()
        y_ox = mlp_sess.run(None, {"mlp_input": x})[0]
        d = float(np.max(np.abs(y_pt - y_ox)))
        max_diff_mlp = max(max_diff_mlp, d)

    print(f"  MLP max diff (100 random inputs):      {max_diff_mlp:.2e}")
    mlp_ok = max_diff_mlp < 1e-4
    print(f"  MLP parity:                            {'PASS' if mlp_ok else 'FAIL'}")

    return enc_ok and mlp_ok


if __name__ == "__main__":
    export_encoder()
    export_mlp()
    ok = verify_parity()

    print("\n" + "=" * 60)
    print(f"  Phase 5 ONNX Export {'COMPLETE' if ok else 'FAILED PARITY CHECK'}")
    print("=" * 60 + "\n")

    sys.exit(0 if ok else 1)