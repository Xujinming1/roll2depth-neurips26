"""Evaluate Roll2Depth on SUNRGBD, iBims, DIODE or Hypersim."""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from roll2depth.checkpoint import load_weights
from roll2depth.config import PROJECT_ROOT, load_config
from roll2depth.data import DepthDataset
from roll2depth.metrics import METRIC_NAMES, MetricAverages, compute_metrics
from roll2depth.model import Roll2DepthModel, predict_depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints/roll2depth_v45.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0", help="CUDA device index, or cpu")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--save-predictions", action="store_true", help="Save metric depth .npy maps")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive.")
    config = load_config(args.config)
    dataset = DepthDataset(config["data"], config["synthesis"], max_samples=args.max_samples)
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{int(args.device)}")
    torch.set_float32_matmul_precision("highest")
    model = Roll2DepthModel(pretrained=False)
    loaded = load_weights(model, args.checkpoint)
    print(json.dumps(loaded))
    model.to(device).eval()
    workers = args.num_workers if args.num_workers is not None else config.get("num_workers", 8)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prediction_dir = args.output_dir / "predictions"
    if args.save_predictions:
        prediction_dir.mkdir()
    averages = MetricAverages(device)
    count, skipped = 0, 0
    with (args.output_dir / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["index", "sample_id", "num_valid", "scale", "shift", *METRIC_NAMES])
        writer.writeheader()
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                pred, values = predict_depth(model, batch["relative_depth"].to(device), batch["stripe"].to(device))
                depth = batch["depth"].to(device)
                if pred.shape[-2:] != depth.shape[-2:]:
                    pred = F.interpolate(pred, depth.shape[-2:], mode="bilinear", align_corners=True)
                result = compute_metrics(pred, depth, config["evaluation"]["max_depth"], config["evaluation"]["crop"])
                if result is None:
                    skipped += 1
                    print(f"Skipped sample {index}: no valid depth pixels")
                    continue
                row = {name: float(result[name]) for name in METRIC_NAMES}
                averages.update(result)
                count += 1
                writer.writerow(dict(index=index, sample_id=batch["sample_id"][0],
                                     num_valid=result["num_valid"], scale=float(values[0, 0]),
                                     shift=float(values[0, 1]), **row))
                stream.flush()
                if args.save_predictions:
                    import numpy as np
                    np.save(prediction_dir / f"{index:06d}.npy", pred[0, 0].cpu().numpy())
                print(f"{index + 1}/{len(dataset)} abs_rel={row['abs_rel']:.6f}")
    if count == 0:
        raise RuntimeError("No samples with valid evaluation pixels.")
    summary = {
        "dataset": config["data"]["dataset"], "evaluated_images": count, "skipped_images": skipped,
        "aggregation": "arithmetic mean over valid images (legacy batch size 1)",
        "relative_depth_source": config["data"]["relative_depth_source"],
        "evaluation": config["evaluation"], "metrics": averages.compute(),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
