"""Version45 indoor stripe synthesis and rolling-shutter FSK templates."""

import cv2
import numpy as np


def synthesize_stripes(image, albedo, depth, bank, sample_index, *, k0=1200.0,
                       gamma=0.45, fx=585.0, fy=585.0, cx=324.5, cy=253.7):
    """Return (1D pattern, striped RGB, O_norm), all with legacy numerical rules.

    The bank entry is seeded by the original split index during both training
    and evaluation, matching the boolean mode passed by the version45 dataset.
    O_norm intentionally retains uint8 conversion BEFORE clipping for parity.
    """
    height, width = depth.shape
    if image.shape != (height, width, 3) or albedo.shape != image.shape:
        raise ValueError("RGB, albedo and depth must have matching spatial dimensions.")
    depth[np.isnan(depth)] = 0.5
    depth[np.isinf(depth)] = 100
    mask = (depth <= 0.5).astype(np.uint8)
    z = cv2.inpaint(depth.astype(np.float32), mask, 8, cv2.INPAINT_TELEA)
    z[z <= 0.5] = 0.5

    rng = np.random.RandomState(sample_index)
    pattern = bank[rng.randint(0, bank.shape[0]), :]
    pattern = cv2.resize(pattern[:, None], (1, height), interpolation=cv2.INTER_LINEAR)
    pattern = np.clip(pattern, 0, None)
    template = np.tile(pattern, (1, width))
    template = np.tile(template[:, :, None], (1, 1, 3))

    # Match the original zero-albedo replacement, including shared RGB arrays.
    albedo[albedo == 0] = 1
    shading = (image.astype(np.float64) / albedo.astype(np.float64)) ** (1 / gamma)
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    points = np.dstack(((xx - cx) * z / fx, (yy - cy) * z / fy, z))
    distance = np.linalg.norm(points, axis=2, keepdims=True)
    irradiance = template / (distance ** 2 + 1e-6)

    with np.errstate(invalid="ignore", over="ignore"):
        onorm = (np.power(np.mean(irradiance * k0, axis=2), gamma)[:, :, None]
                 * albedo).astype(np.uint8)
    onorm = np.clip(onorm, 0, 255)
    striped = albedo.astype(np.float64) * np.power(shading + irradiance * k0, gamma)
    striped = np.clip(striped, 0, 255).astype(np.uint8)
    return pattern, striped, onorm


def synthesize_fsk_stripe(rng, exposure=1 / 7200, frequency=1202.0,
                          camera_factor=1.32, rows=1080):
    """Integrate a 50% duty-cycle source over rolling-shutter exposure windows."""
    period = 1 / frequency
    row_delay = (1 / (60 * camera_factor) - exposure) / (rows - 1)
    light_duration = period * 0.5
    full_cycles = np.floor(exposure / period)
    residual = exposure - full_cycles * period
    start = residual * rng.rand()
    stripes = np.zeros(rows)
    for i in range(rows):
        end = start + residual
        a, b = np.floor(start / period), np.floor(end / period)
        if a == b:
            amount = max(min(a * period + light_duration - start, residual), 0)
        elif b == a + 1:
            amount = max(a * period + light_duration - start, 0)
            amount += min(light_duration, end - b * period)
        else:
            amount = 0
        stripes[i] = amount + full_cycles * light_duration
        start += row_delay
    return stripes
