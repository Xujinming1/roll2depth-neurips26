# Roll2Depth

Training on NYU and evaluation on SUNRGBD, iBims, DIODE and Hypersim.
See the paper for the method and experimental results.

## Installation

Run the following commands from the repository root. Create a Python 3.9
environment, then install PyTorch and the remaining dependencies:

```bash
conda create -n roll2depth python=3.9 -y
conda activate roll2depth
python -m pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

The command above installs the CUDA 11.8 build. For other CUDA builds or CPU-only
installation, use the matching command from the
[PyTorch installation page](https://pytorch.org/get-started/previous-versions/#v250).

## Model assets

The following large files are not included in this repository. Place them at the
paths below before running the corresponding commands:

| File path | Size | Required for |
|---|---|---|
| `checkpoints/roll2depth_v45.pt` | 1.22 GB | Demo, evaluation, and fine-tuning the supplied model |
| `assets/stripes.npy` | 115 MB | Training and evaluation stripe synthesis |

```bash
mkdir -p checkpoints assets
```

The demo uses its prepared stripe cue and does not need `assets/stripes.npy`.
Training from scratch uses DINOv2 initialization and does not require the final
Roll2Depth checkpoint. Download links for the two large files are not available yet.

## Quick demo

The `demo/` folder contains one iBims example (`livingroom_11`) with three files:

- `rgb.png`: the scene image for reference.
- `stripe.png`: the prepared three-channel stripe cue.
- `relative_depth.npy`: the normalized Marigold relative depth, ready for inference.

After placing the checkpoint as described above, run on CPU:

```bash
python demo.py --output-dir runs/demo
```

Add `--device 0` for GPU inference. Use `--input-dir path/to/your/demo` to supply
another set of the same three files with matching image dimensions. The network
uses the stripe cue and relative depth. The script saves only `depth.png`, a
colorized prediction; its colors visualize depth and do not encode meter values.
Choose a new output directory for each run.

## Prepare data

Prepare the datasets and preprocessing outputs described in [DATA.md](DATA.md).
Evaluation requires RGB/depth data, inverse-rendered albedos and precomputed
Marigold relative-depth arrays. These external files are not included or generated
automatically. NYU training also requires dense depth maps.

Replace the `path/to/your/...` values in the configuration files:

| Task | Configuration | Paths to set |
|---|---|---|
| Training | `configs/train.json` | `train_data.root`, `val_data.root`, `val_data.albedo_root`, `val_data.relative_depth_root` |
| Evaluation | `configs/eval/{sun,ibims,diode,hypersim}.json` | `data.root`, `data.albedo_root`, `data.relative_depth_root` |

Use absolute paths for external data. Keep the supplied split lists in order.
Place the required model assets as described above.

## Training

```bash
python train.py --config configs/train.json --device 0 --output-dir runs/train
```

The default configuration uses 100 epochs, batch size 2, AdamW with learning rate
`1e-5` and weight decay `0.1`, and SUNRGBD validation every 3,000 training batches.

A new training run downloads DINOv2 pretrained weights through timm. For offline
training, set `model.backbone_weights` in the training configuration to a local
`vit_large_patch14_dinov2` state dictionary or safetensors file with patch14 weights.

To run one training batch and one validation batch:

```bash
python train.py --config configs/train.json --device 0 --num-workers 0 \
  --fast-dev-run --max-train-samples 2 --max-val-samples 1 \
  --output-dir runs/train_smoke
```

To fine-tune the supplied model, add
`--init-checkpoint checkpoints/roll2depth_v45.pt`. To resume a training run,
add `--resume runs/train/roll2depth/version_0/checkpoints/last.ckpt`, replacing the
path with your run's checkpoint. Use one of these options at a time.

Logs and checkpoints are saved under the selected output directory. `last.ckpt`
and the best-validation checkpoint include the training state; the best-d1
checkpoint contains model weights only.

## Evaluation

```bash
python evaluate.py --config configs/eval/sun.json --device 0 --output-dir runs/sun
python evaluate.py --config configs/eval/ibims.json --device 0 --output-dir runs/ibims
python evaluate.py --config configs/eval/diode.json --device 0 --output-dir runs/diode
python evaluate.py --config configs/eval/hypersim.json --device 0 --output-dir runs/hypersim
```

Evaluation loads `checkpoints/roll2depth_v45.pt` by default and does not download
model weights. Use `--checkpoint path/to/your/model.pt` to select another checkpoint.
Each evaluation requires a new output directory.

To check the setup on two images before a full evaluation:

```bash
python evaluate.py --config configs/eval/sun.json --device 0 \
  --max-samples 2 --num-workers 0 --output-dir runs/sun_smoke
```

Each run saves `per_image.csv` and `metrics.json`. Add `--save-predictions` to save
depth maps in meters as `.npy` files under `predictions/`.
Use `--device cpu` to run on CPU, or change `--device 0` to another GPU index.
Use `--num-workers 0` if multiprocessing is unavailable.

## Stripe synthesis

Training and evaluation synthesize stripe cues automatically using
`roll2depth/synthesis.py` and the supplied `assets/stripes.npy`; no separate step is
needed. To generate a custom FSK template bank:

```bash
python tools/generate_stripes.py --output assets/custom_stripes.npy --count 1000
```

Set `synthesis.bank` in the relevant configuration to use a custom bank.
Keep the supplied bank for the paper configuration. Run a script with `--help`
to see its available options.
