# Data preparation

Download the datasets and prepare the dense depth, inverse-rendered albedo and
Marigold outputs required below, following the paper's preprocessing setup.
The training and evaluation scripts read these files from disk.

Keep the supplied `splits/*.txt` files in order. Paths in each split are relative
to the configured dataset root. All depths used by the model are in meters after
the loader's unit conversion.

## NYU training

Set `train_data.root` in `configs/train.json` to the directory containing the NYU
scene folders. For example, the first split entry requires:

```text
<train_data.root>/kitchen_0028b/rgb_00045.jpg
<train_data.root>/kitchen_0028b/sync_depth_00045.png
<train_data.root>/kitchen_0028b/dense/sync_depth_dense_00045.png
```

The two depth PNGs must store millimeters; the loader divides them by 1,000.
Dense maps are required in addition to synchronized depth. NYU training uses RGB
as the albedo proxy and constructs relative depth from dense depth, so it does
not require separate albedo or Marigold files.

Training also validates on SUNRGBD. Configure `val_data.root`,
`val_data.albedo_root` and `val_data.relative_depth_root` with the SUNRGBD assets
described below.

## Evaluation datasets

Set `data.root` in each `configs/eval/*.json` to the directory containing:

| Dataset | Required data under `data.root` |
|---|---|
| SUNRGBD | `kv1/...`, `kv2/...`, including RGB, depth and `depth_bfx` dense depth files named in `splits/sun.txt` |
| iBims | `ibims1_core_mat/*.mat`, containing RGB, depth, invalid and transparent masks |
| DIODE | `scene_.../scan_.../` with RGB, metric-depth `.npy` and depth-mask `.npy` files |
| Hypersim | `ai_.../images/` with RGB previews and geometry depth HDF5 files |

For iBims, a split RGB path such as `ibims1_core_raw/rgb/corridor_01.png`
maps to `<data.root>/ibims1_core_mat/corridor_01.mat`. RGB and depth are read from
that MAT record.

SUNRGBD depth PNG values are divided by 10,000. DIODE depth arrays must be in
meters. Hypersim geometry must retain its 768 x 1024 resolution; the loader
converts stored ray distance to optical-axis depth.

### Albedos

Set `data.albedo_root` to the inverse-rendering output directory. Retain the RGB
relative directory and use the suffix `_albedoBS1.png`:

```text
<data.root>/<relative-directory>/<rgb-stem>.jpg
<data.albedo_root>/<relative-directory>/<rgb-stem>_albedoBS1.png
```

The input RGB extension may also be `.png`. Supply RGB albedo PNGs in [0,255].
Normal maps and roughness estimates are not required.

For example, the first iBims albedo must be at:

```text
<data.albedo_root>/ibims1_core_raw/rgb/corridor_01_albedoBS1.png
```

### Marigold relative depth

Set `data.relative_depth_root` to the directory containing one `.npy` prediction
per split entry. Arrays must contain depth-like values that increase with
distance. Do not supply inverse depth. Singleton channel dimensions are accepted.

| Dataset | Filename under `data.relative_depth_root` |
|---|---|
| SUNRGBD | `idx-0.npy`, `idx-1.npy`, ... |
| iBims | RGB stem plus `.npy`, e.g. `corridor_01.npy` |
| DIODE | RGB stem plus `.npy`, e.g. `00019_00183_indoors_000_010.npy` |
| Hypersim | `idx-0.npy`, `idx-1.npy`, ... |

Indices are zero-based and follow the supplied split order. Preserve each
image's evaluation resolution. Leave `relative_depth_source` set to `marigold`;
the loader performs relative-depth normalization automatically.
