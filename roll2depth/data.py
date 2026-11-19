"""NYU training and SUNRGBD / iBims / DIODE / Hypersim evaluation adapters."""

from functools import lru_cache
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

from .synthesis import synthesize_stripes


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read RGB/albedo: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_depth(path, factor=1.0):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth: {path}")
    return depth.astype(np.float32) / factor


def resize(array, size):
    return cv2.resize(array, tuple(size), interpolation=cv2.INTER_CUBIC)


def compute_relative_depth(depth, random_range=False, range_jitter=0.2):
    """Percentile normalization with its inverse affine parameters."""
    depth = np.asarray(depth)
    valid = depth[(depth > 0) & np.isfinite(depth)]
    if valid.size == 0:
        raise ValueError("No finite positive depths for relative-depth normalization.")
    low, high = np.percentile(valid, (2, 98))
    if high <= low:
        raise ValueError("Relative-depth normalization requires a nonconstant depth map.")
    if random_range:
        upper = np.random.uniform(1 - range_jitter, 1 + range_jitter)
        lower = np.random.uniform(-1 - range_jitter, -1 + range_jitter)
    else:
        upper, lower = 1, -1
    depth = np.nan_to_num(depth, nan=1e-3, posinf=80, neginf=0)
    relative = (depth - low) / (high - low) * (upper - lower) + lower
    relative[depth <= 0] = 0
    scale = (high - low) / (upper - lower)
    shift = (low * upper - high * lower) / (upper - lower)
    return relative, (scale, shift)


@lru_cache(maxsize=1)
def hypersim_ray_norm():
    width, height, focal = 1024, 768, 886.81
    x = np.linspace(-0.5 * width + 0.5, 0.5 * width - 0.5, width)
    y = np.linspace(-0.5 * height + 0.5, 0.5 * height - 0.5, height)
    x = x.reshape(1, width).repeat(height, 0).astype(np.float32)[:, :, None]
    y = y.reshape(height, 1).repeat(width, 1).astype(np.float32)[:, :, None]
    z = np.full((height, width, 1), focal, np.float32)
    return np.linalg.norm(np.concatenate((x, y, z), axis=2), 2, 2)


def as_tensor(array, rgb=False):
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[:, :, None]
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    return tensor.float() / 255.0 if rgb else tensor


class DepthDataset(Dataset):
    def __init__(self, config, synthesis, training=False, max_samples=None):
        self.config = config
        self.synthesis = synthesis
        self.training = training
        self.name = config["dataset"]
        if self.name not in {"nyu", "sun", "ibims", "diode", "hypersim"}:
            raise ValueError(f"Unsupported dataset: {self.name}")
        if training and self.name != "nyu":
            raise ValueError("The version45 training recipe uses NYU only.")
        self.root = Path(config["root"])
        self.albedo_root = Path(config.get("albedo_root", "."))
        self.bank = np.load(synthesis["bank"], mmap_mode="r")
        if self.bank.ndim != 2 or min(self.bank.shape) < 1:
            raise ValueError("Stripe bank must be a nonempty [templates, rows] array.")
        self.rows = [line.split() for line in Path(config["split"]).read_text().splitlines()
                     if line.strip() and "None" not in line and not line.lstrip().startswith("#")]
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError("The selected data split is empty.")

    def __len__(self):
        return len(self.rows)

    def _load(self, row):
        rgb_file, depth_file = row[:2]
        if self.name == "nyu":
            size = self.config.get("size", [384, 384])  # width, height
            image = resize(read_rgb(self.root / rgb_file), size)
            factor = self.config.get("depth_factor", 1000.0)
            depth = resize(read_depth(self.root / depth_file, factor), size)
            dense_file = depth_file.replace("sync_depth", "dense/sync_depth_dense")
            dense = resize(read_depth(self.root / dense_file, factor), size)
            # Version45 deliberately uses RGB as the training albedo proxy.
            return image, depth, image, dense, image

        if self.name == "ibims":
            mat_file = rgb_file.replace("raw/rgb", "mat").replace("png", "mat")
            record = loadmat(self.root / mat_file)["data"]
            image = resize(record["rgb"][0][0], (640, 480))
            depth = resize(record["depth"][0][0], (640, 480))
            invalid = record["mask_invalid"][0][0]
            transparent = record["mask_transp"][0][0]
            if invalid.shape != depth.shape:
                invalid = cv2.resize(invalid, (640, 480), interpolation=cv2.INTER_NEAREST)
                transparent = cv2.resize(transparent, (640, 480), interpolation=cv2.INTER_NEAREST)
            depth = depth * invalid * transparent * (depth != 0)
        else:
            image = read_rgb(self.root / rgb_file)
            if self.name == "sun":
                depth = read_depth(self.root / depth_file, 10000.0)
            elif self.name == "diode":
                depth = np.load(self.root / depth_file).squeeze()
                if self.config.get("apply_valid_mask", False):
                    depth = depth * np.load(self.root / row[2]).astype(bool).squeeze()
            else:
                with h5py.File(self.root / depth_file, "r") as handle:
                    distance = np.array(handle["dataset"])
                if distance.shape != (768, 1024):
                    raise ValueError("Expected the original 768 x 1024 Hypersim geometry.")
                depth = distance / hypersim_ray_norm() * 886.81

        synthesis_image = resize(image, (640, 480))
        if self.name == "sun":
            # The legacy path resizes the integer-valued dense depth before /10000.
            dense = resize(read_depth(self.root / row[2]), (640, 480)) / 10000.0
        else:
            dense = resize(depth, (640, 480))
        albedo_file = str(Path(rgb_file).with_suffix("")) + "_albedoBS1.png"
        albedo = resize(read_rgb(self.albedo_root / albedo_file), (640, 480))
        return image, depth, synthesis_image, dense, albedo

    def __getitem__(self, index):
        try:
            row = self.rows[index]
            image, depth, synthesis_image, dense, albedo = self._load(row)
            physical = {key: self.synthesis[key] for key in
                        ("k0", "gamma", "fx", "fy", "cx", "cy") if key in self.synthesis}
            _, _, cue = synthesize_stripes(synthesis_image, albedo, dense, self.bank, index, **physical)
            if cue.shape[:2] != image.shape[:2]:
                cue = resize(cue, image.shape[:2][::-1])

            reference = dense if self.name in {"nyu", "sun"} else depth
            relative, (scale, shift) = compute_relative_depth(
                reference, random_range=self.training and self.config.get("random_range", True),
                range_jitter=self.config.get("range_jitter", 0.2),
            )
            source = self.config.get("relative_depth_source", "marigold")
            if source == "marigold":
                relative_root = Path(self.config["relative_depth_root"])
                filename = (f"idx-{index}.npy" if self.name in {"sun", "hypersim"}
                            else Path(row[0]).stem + ".npy")
                relative, _ = compute_relative_depth(np.load(relative_root / filename).squeeze())
            elif source != "gt":
                raise ValueError("relative_depth_source must be 'gt' or 'marigold'.")
            return {
                "image": as_tensor(image, rgb=True),
                "depth": as_tensor(depth),
                "stripe": as_tensor(cue, rgb=True),
                "relative_depth": as_tensor(relative).float(),
                "scale": torch.tensor(scale, dtype=torch.float32),
                "shift": torch.tensor(shift, dtype=torch.float32),
                "sample_id": row[0],
            }
        except (OSError, ValueError, KeyError, IndexError) as error:
            raise RuntimeError(f"{self.name} sample {index}: {error}") from error
