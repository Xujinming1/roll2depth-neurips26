"""Train the version45 Roll2Depth model on NYU and validate on SUNRGBD."""

import argparse
import json
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader

from roll2depth.checkpoint import load_weights
from roll2depth.config import PROJECT_ROOT, load_config
from roll2depth.data import DepthDataset
from roll2depth.model import Roll2DepthModel
from roll2depth.training import Roll2DepthTask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/train.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/train")
    parser.add_argument("--device", default="0", help="CUDA device index, or cpu")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--init-checkpoint", type=Path, help="Initialize model weights; optimizer starts fresh")
    parser.add_argument("--resume", type=Path, help="Resume a checkpoint produced by this training script")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--fast-dev-run", action="store_true", help="One training and one validation batch; no checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.init_checkpoint and args.resume:
        parser.error("Choose either --init-checkpoint or --resume.")
    for value in (args.max_train_samples, args.max_val_samples):
        if value is not None and value < 1:
            parser.error("Sample limits must be positive.")
    config = load_config(args.config)
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    settings = config["training"]
    workers = args.num_workers if args.num_workers is not None else settings["num_workers"]
    train_data = DepthDataset(config["train_data"], config["synthesis"], True, args.max_train_samples)
    val_data = DepthDataset(config["val_data"], config["synthesis"], False, args.max_val_samples)
    model_config = config.get("model", {})
    model = Roll2DepthModel(
        pretrained=not bool(args.init_checkpoint or args.resume),
        backbone_weights=None if args.init_checkpoint or args.resume else model_config.get("backbone_weights"),
    )
    if args.init_checkpoint:
        print(load_weights(model, args.init_checkpoint))
    task = Roll2DepthTask(model, config)
    logger = TensorBoardLogger(save_dir=str(args.output_dir), name="roll2depth")
    checkpoint_dir = Path(logger.log_dir) / "checkpoints"
    callbacks = [
        ModelCheckpoint(dirpath=checkpoint_dir, filename="best-val-{epoch:02d}",
                        monitor="val_loss", mode="min", save_top_k=1, save_last=True,
                        save_weights_only=False),
        ModelCheckpoint(dirpath=checkpoint_dir, filename="best-d1-{epoch:02d}",
                        monitor="d1", mode="max", save_top_k=1, save_weights_only=True),
    ]
    trainer = L.Trainer(
        accelerator="cpu" if args.device == "cpu" else "gpu",
        devices=1 if args.device == "cpu" else [int(args.device)],
        max_epochs=settings["epochs"], max_steps=args.max_steps,
        val_check_interval=settings["val_check_interval"],
        precision="32-true", logger=logger, callbacks=callbacks,
        fast_dev_run=args.fast_dev_run, log_every_n_steps=50,
        default_root_dir=str(args.output_dir),
    )
    trainer.fit(
        task,
        train_dataloaders=DataLoader(train_data, batch_size=settings["batch_size"],
                                     num_workers=workers, shuffle=True),
        val_dataloaders=DataLoader(val_data, batch_size=1, num_workers=workers),
        ckpt_path=str(args.resume) if args.resume else None,
    )
    summary = {"optimizer_steps": trainer.global_step}
    for key in ("train_loss", "val_loss", "d1"):
        if key in trainer.callback_metrics:
            summary[key] = float(trainer.callback_metrics[key])
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
