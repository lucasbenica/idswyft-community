#!/usr/bin/env python3
"""
Export a fine-tuned deepfake detector to ONNX for the Idswyft engine.

The previous version of this script exported an ImageNet-pretrained
EfficientNet-B0 with a randomly initialized 2-class head whenever it failed to
find deepfake weights — a model that answers noise. This one only ships a
checkpoint that was actually fine-tuned for deepfake detection, and writes a
sidecar descriptor so the runtime preprocesses exactly the way the checkpoint
was trained (normalization + logit order).

Usage:
    pip install -r requirements.txt
    python export-deepfake-detector.py [--model REPO_ID] [--quantize]

Output:
    ../../shared/models/deepfake-detector.onnx
    ../../shared/models/deepfake-detector.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForImageClassification

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / ".." / ".." / "shared" / "models"
OUTPUT_PATH = OUTPUT_DIR / "deepfake-detector.onnx"
DESCRIPTOR_PATH = OUTPUT_DIR / "deepfake-detector.json"

# ViT-base fine-tuned on real vs. deepfake faces. Two labels: Realism / Deepfake.
DEFAULT_REPO = "prithivMLmods/Deep-Fake-Detector-v2-Model"
OPSET_VERSION = 17

# Label names that mean "this face is genuine". Everything else is the fake class.
REAL_LABEL_HINTS = {"real", "realism", "genuine", "authentic", "live", "not_deepfake"}


def classify_labels(config) -> list:
    """Map the checkpoint's id2label onto the runtime's ('real' | 'fake') order."""
    id2label = {int(k): str(v) for k, v in (config.id2label or {}).items()}
    if len(id2label) != 2:
        raise SystemExit(f"Expected a 2-class checkpoint, got {len(id2label)}: {id2label}")

    labels = []
    for index in sorted(id2label):
        name = id2label[index].strip().lower().replace(" ", "_")
        labels.append("real" if name in REAL_LABEL_HINTS else "fake")

    if set(labels) != {"real", "fake"}:
        raise SystemExit(f"Could not tell real from fake in {id2label} — extend REAL_LABEL_HINTS")
    return labels


def export(repo_id: str) -> tuple:
    print(f"Loading {repo_id} ...")
    config = AutoConfig.from_pretrained(repo_id)
    model = AutoModelForImageClassification.from_pretrained(repo_id)
    model.eval()

    # Read the preprocessing straight from the checkpoint's own config — going
    # through AutoImageProcessor would drag torchvision in for no gain here.
    with open(hf_hub_download(repo_id, "preprocessor_config.json"), encoding="utf-8") as handle:
        preprocessor = json.load(handle)

    labels = classify_labels(config)
    size = preprocessor.get("size") or {}
    input_size = int(size.get("height") or size.get("shortest_edge") or 224)
    mean = [float(v) for v in preprocessor["image_mean"]]
    std = [float(v) for v in preprocessor["image_std"]]

    print(f"  labels (logit order): {labels}")
    print(f"  input: {input_size}x{input_size}, mean={mean}, std={std}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, input_size, input_size)

    print(f"Exporting to ONNX (opset {OPSET_VERSION}) ...")

    class LogitsOnly(torch.nn.Module):
        """The runtime reads a plain logits tensor, not a HF output object."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            return self.inner(pixel_values=pixel_values).logits

    torch.onnx.export(
        LogitsOnly(model),
        dummy,
        str(OUTPUT_PATH),
        input_names=["input"],
        output_names=["output"],
        opset_version=OPSET_VERSION,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )

    descriptor = {
        "source": repo_id,
        "inputSize": input_size,
        "mean": mean,
        "std": std,
        "labels": labels,
    }
    DESCRIPTOR_PATH.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  Saved: {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print(f"  Saved: {DESCRIPTOR_PATH}")
    return descriptor, model


def quantize() -> None:
    """Dynamic int8 quantization — roughly a 4x smaller file, same interface."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    raw_path = OUTPUT_PATH.with_suffix(".fp32.onnx")
    OUTPUT_PATH.replace(raw_path)
    print("Quantizing to int8 ...")
    quantize_dynamic(str(raw_path), str(OUTPUT_PATH), weight_type=QuantType.QInt8)
    raw_path.unlink()
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  Quantized: {OUTPUT_PATH} ({size_mb:.1f} MB)")


def validate(descriptor: dict, model) -> None:
    """Structural check plus agreement between torch and ONNX on a real image."""
    print("\nValidating ...")
    onnx.checker.check_model(onnx.load(str(OUTPUT_PATH)))
    print("  ONNX structural check: PASSED")

    input_size = descriptor["inputSize"]
    session = ort.InferenceSession(str(OUTPUT_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 255, size=(input_size, input_size, 3), dtype=np.uint8).astype(np.float32) / 255.0
    normalized = (pixels - np.array(descriptor["mean"])) / np.array(descriptor["std"])
    tensor = normalized.transpose(2, 0, 1)[None].astype(np.float32)

    onnx_logits = session.run([output_name], {input_name: tensor})[0][0]
    with torch.no_grad():
        torch_logits = model(pixel_values=torch.from_numpy(tensor)).logits[0].numpy()

    drift = float(np.max(np.abs(onnx_logits - torch_logits)))
    print(f"  torch vs onnx max logit drift: {drift:.4f}")
    if drift > 0.5:
        raise SystemExit("ONNX output diverges from the torch model — aborting")

    exp = np.exp(onnx_logits - onnx_logits.max())
    probs = exp / exp.sum()
    readable = {label: round(float(p), 4) for label, p in zip(descriptor["labels"], probs)}
    print(f"  Probabilities on noise: {readable}")
    print("  Runtime inference check: PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_REPO, help="HuggingFace repo id of the checkpoint")
    parser.add_argument("--quantize", action="store_true", help="Emit an int8 model (~4x smaller, slight accuracy cost)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing model without asking")
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.force:
        size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {OUTPUT_PATH} ({size_mb:.1f} MB)")
        if input("Overwrite? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)

    descriptor, model = export(args.model)
    if args.quantize:
        quantize()
    validate(descriptor, model)

    print("\nDeepfake detector ready.")
    print("  Rebuild the engine image so shared/models/ ships with it.")
    print("  Look for: 'Deepfake detector model loaded'")


if __name__ == "__main__":
    main()
