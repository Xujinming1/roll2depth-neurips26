"""The version45 crop, depth ranges and nine depth-estimation metrics."""

import torch

METRIC_NAMES = ("silog", "abs_rel", "log10", "rmse", "sq_rel", "rmse_log", "d1", "d2", "d3")


def silog_loss(prediction, target):
    valid = (target > 0) & torch.isfinite(target)
    if not valid.any():
        return None
    # Keep training/evaluation hooks finite for an untrained affine head.
    difference = torch.log(target[valid]) - torch.log(prediction[valid].clamp_min(1e-6))
    return (difference.square().mean() - 0.85 * difference.mean().square()).clamp_min(0).sqrt()


def compute_metrics(prediction, target, max_depth=10.0, crop="eigen"):
    # Work on copies: metrics must not modify maps reused for saved predictions.
    prediction = prediction.clone().clamp(1e-3, max_depth)
    target = torch.nan_to_num(target, nan=1e-3, posinf=max_depth, neginf=0)
    valid = (target > 1e-3) & (target < max_depth)
    if crop == "eigen":
        crop_mask = torch.zeros_like(valid)
        crop_mask[..., 45:471, 41:601] = True
        valid &= crop_mask
    elif crop != "none":
        raise ValueError("Supported evaluation crops: 'eigen', 'none'.")
    if not valid.any():
        return None
    pred, gt = prediction[valid], target[valid]
    if not torch.isfinite(pred).all():
        raise ValueError("Nonfinite predicted depths in the evaluation mask.")
    threshold = torch.maximum(gt / pred, pred / gt)
    diff = pred - gt
    log_diff = torch.log(pred) - torch.log(gt)
    return {
        "silog": (log_diff.square().mean() - log_diff.mean().square()).clamp_min(0).sqrt() * 100,
        "abs_rel": (diff.abs() / gt).mean(),
        "log10": (torch.log10(pred) - torch.log10(gt)).abs().mean(),
        "rmse": diff.square().mean().sqrt(),
        "sq_rel": (diff.square() / gt).mean(),
        "rmse_log": log_diff.square().mean().sqrt(),
        "d1": (threshold < 1.25).float().mean(),
        "d2": (threshold < 1.25 ** 2).float().mean(),
        "d3": (threshold < 1.25 ** 3).float().mean(),
        "num_valid": int(valid.sum()),
    }


class MetricAverages:
    """Match Lightning's sequential, dtype-preserving epoch mean for batch size 1."""

    def __init__(self, device="cpu"):
        self.sums = {name: torch.zeros((), device=device, dtype=torch.float32)
                     for name in METRIC_NAMES}
        self.count = 0

    def update(self, metrics):
        for name in METRIC_NAMES:
            self.sums[name] = self.sums[name] + metrics[name]
        self.count += 1

    def compute(self):
        if self.count == 0:
            raise ValueError("No valid images to average.")
        return {name: float(value / self.count) for name, value in self.sums.items()}
