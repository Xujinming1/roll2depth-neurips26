"""Predict one depth map from the bundled stripe cue and relative depth."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from roll2depth.checkpoint import load_weights
from roll2depth.config import PROJECT_ROOT
from roll2depth.model import Roll2DepthModel, predict_depth


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_depth_image(depth, path):
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("Prediction contains no finite positive depths.")
    low, high = np.percentile(depth[valid], (2, 98))
    normalized = np.clip((np.where(valid, depth, low) - low) / max(float(high - low), 1e-6), 0, 1)
    colored = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    if not cv2.imwrite(str(path), colored):
        raise OSError(f"Cannot save depth image: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "demo")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints/roll2depth_v45.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/demo")
    parser.add_argument("--device", default="cpu", help="cpu (default), or CUDA device index")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory to keep previous results.")
    if args.device != "cpu":
        if (not args.device.isdigit() or not torch.cuda.is_available()
                or int(args.device) >= torch.cuda.device_count()):
            parser.error("CUDA is unavailable or the device index is invalid; use --device cpu.")
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    image = read_rgb(args.input_dir / "rgb.png")
    stripe = read_rgb(args.input_dir / "stripe.png")
    relative = np.load(args.input_dir / "relative_depth.npy", allow_pickle=False)
    if relative.ndim != 2 or not np.isfinite(relative).all():
        raise ValueError("relative_depth.npy must contain a finite 2D array.")
    if image.shape != stripe.shape or relative.shape != image.shape[:2]:
        raise ValueError("RGB, stripe and relative depth must have matching image dimensions.")
    # RGB provides the scene reference; the network consumes stripe and relative depth.
    relative_tensor = torch.from_numpy(np.ascontiguousarray(relative, dtype=np.float32))[None, None]
    stripe_tensor = torch.from_numpy(np.ascontiguousarray(stripe.transpose(2, 0, 1))).float()[None] / 255
    torch.set_float32_matmul_precision("highest")
    model = Roll2DepthModel(pretrained=False)
    load_weights(model, args.checkpoint)
    model.to(device).eval()
    with torch.inference_mode():
        prediction, _ = predict_depth(model, relative_tensor.to(device), stripe_tensor.to(device))
    depth = prediction[0, 0].cpu().numpy()
    if not np.isfinite(depth).all():
        raise ValueError("Prediction contains nonfinite depths.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "depth.png"
    save_depth_image(depth, output)
    print(f"Saved depth image: {output}")


if __name__ == "__main__":
    main()
