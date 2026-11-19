"""Version45 affine-parameter supervision and AdamW parameter groups."""

import lightning as L
import torch
import torch.nn.functional as F

from .metrics import METRIC_NAMES, compute_metrics, silog_loss
from .model import predict_depth


class Roll2DepthTask(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters({"config": config})

    def training_step(self, batch, batch_idx):
        rd = F.interpolate(batch["relative_depth"].float(), (384, 384),
                           mode="bilinear", align_corners=True)
        stripe = F.interpolate(batch["stripe"], (384, 384), mode="bilinear", align_corners=True)
        _, values = self.model(rd, stripe)
        target = torch.stack((batch["scale"], batch["shift"]), dim=1)
        loss = F.mse_loss(values, target)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite affine-parameter training loss.")
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=len(values))
        return loss

    def validation_step(self, batch, batch_idx):
        pred, _ = predict_depth(self.model, batch["relative_depth"], batch["stripe"])
        depth = batch["depth"]
        if pred.shape[-2:] != depth.shape[-2:]:
            pred = F.interpolate(pred, depth.shape[-2:], mode="bilinear", align_corners=True)
        settings = self.config["validation"]
        loss = silog_loss(pred, depth)
        if loss is not None:
            self.log("val_loss", loss, on_epoch=True, batch_size=1)
        metrics = compute_metrics(pred, depth, settings["max_depth"], settings["crop"])
        if metrics is None:
            raise ValueError("Validation sample has no valid evaluation pixels.")
        self.log_dict({name: metrics[name] for name in METRIC_NAMES}, on_epoch=True, batch_size=1)

    def configure_optimizers(self):
        settings = self.config["training"]
        groups = {}
        no_decay = ("relative_position_bias_table", "rpe_mlp", "logit_scale")
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            decay = settings["weight_decay"]
            if (parameter.ndim == 1 or name.endswith(".bias") or "absolute_pos_embed" in name
                    or any(token in name for token in no_decay)):
                decay = 0.0
            groups.setdefault(decay, []).append(parameter)
        return torch.optim.AdamW([
            {"params": params, "lr": settings["lr"], "weight_decay": decay}
            for decay, params in groups.items()
        ])
