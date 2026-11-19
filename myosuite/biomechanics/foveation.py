# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Eccentricity-dependent foveated acuity and dual-sensor visual sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class FoveationConfig:
    """Retinal acuity falloff parameters (Geisler & Perry, 1998).

    Blur standard deviation scales linearly with angular eccentricity ``theta`` [deg]:
    ``sigma_deg(theta) = fovea_sigma0_deg * (1 + theta / half_resolution_deg)``.

    Args:
        fovea_sigma0_deg: Baseline foveal blur standard deviation [deg].
        half_resolution_deg: Eccentricity ``E_2`` [deg] where blur doubles.
        n_levels: Number of Gaussian blur pyramid levels (``>= 2``).
    """

    fovea_sigma0_deg: float = 0.03
    half_resolution_deg: float = 2.3
    n_levels: int = 5

    def __post_init__(self) -> None:
        if self.fovea_sigma0_deg <= 0.0 or not math.isfinite(self.fovea_sigma0_deg):
            raise ValueError("fovea_sigma0_deg must be finite and positive")
        if self.half_resolution_deg <= 0.0 or not math.isfinite(
            self.half_resolution_deg
        ):
            raise ValueError("half_resolution_deg must be finite and positive")
        if int(self.n_levels) < 2:
            raise ValueError("n_levels must be >= 2")


def intrinsic_from_fovy(height: int, width: int, fovy_deg: float) -> np.ndarray:
    """Build a 3x3 pinhole intrinsic matrix matching MuJoCo's vertical ``fovy`` [deg]."""
    if height < 1 or width < 1:
        raise ValueError(f"height and width must be >= 1, got {(height, width)}")
    if not (0.0 < float(fovy_deg) < 180.0):
        raise ValueError(f"fovy_deg must lie in (0, 180), got {fovy_deg}")
    fy = 0.5 * float(height) / math.tan(0.5 * math.radians(float(fovy_deg)))
    cx = 0.5 * (float(width) - 1.0)
    cy = 0.5 * (float(height) - 1.0)
    return np.array([[fy, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def eccentricity_deg(
    height: int,
    width: int,
    intrinsic: np.ndarray,
    fixation_px: tuple[float, float] | np.ndarray | None = None,
) -> np.ndarray:
    """Compute exact ray-angle eccentricity ``(H, W)`` [deg] from a pinhole intrinsic matrix.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        intrinsic: Calibrated ``(3, 3)`` pinhole matrix.
        fixation_px: Fixation pixel ``(u, v)`` (col, row); defaults to optical centre.

    Returns:
        Non-negative angular eccentricity array of shape ``(height, width)`` in degrees.
    """
    k = np.asarray(intrinsic, dtype=np.float64)
    if k.shape != (3, 3) or k[0, 0] <= 0.0 or k[1, 1] <= 0.0:
        raise ValueError("intrinsic must be a valid (3, 3) pinhole matrix")
    if fixation_px is None:
        u0, v0 = float(k[0, 2]), float(k[1, 2])
    else:
        fix = np.asarray(fixation_px, dtype=np.float64).ravel()
        if fix.size != 2 or not np.all(np.isfinite(fix)):
            raise ValueError("fixation_px must be a finite (u, v) pair")
        u0, v0 = float(fix[0]), float(fix[1])

    k_inv = np.linalg.inv(k)
    us, vs = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    homog = np.stack([us, vs, np.ones_like(us)], axis=-1)
    rays = homog @ k_inv.T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    fovea_ray = k_inv @ np.array([u0, v0, 1.0], dtype=np.float64)
    fovea_ray /= np.linalg.norm(fovea_ray)
    dots = np.clip(rays @ fovea_ray, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def _torch_gaussian_blur_2d(tensor: Any, sigma: float, torch_mod: Any) -> Any:
    """Separable depthwise 2D Gaussian blur on ``(B, C, H, W)`` tensors."""
    if sigma <= 1e-6:
        return tensor
    F = torch_mod.nn.functional
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch_mod.arange(
        -radius, radius + 1, dtype=tensor.dtype, device=tensor.device
    )
    kernel_1d = torch_mod.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    channels = tensor.shape[1]
    kx = kernel_1d.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    ky = kernel_1d.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    padded_x = F.pad(tensor, (radius, radius, 0, 0), mode="replicate")
    blurred_x = F.conv2d(padded_x, kx, groups=channels)
    padded_y = F.pad(blurred_x, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(padded_y, ky, groups=channels)


def foveate_image(
    image: Any,
    intrinsic: np.ndarray,
    fixation_px: tuple[float, float] | np.ndarray | None = None,
    *,
    cfg: FoveationConfig | None = None,
) -> Any:
    """Apply eccentricity-dependent retinal acuity blur to a NumPy or Torch image.

    Supports NumPy arrays ``(H, W)`` / ``(H, W, C)`` and PyTorch tensors
    ``(C, H, W)`` / ``(B, C, H, W)`` without importing ``torch`` unless passed a tensor.

    Args:
        image: Input RGB/grayscale frame(s).
        intrinsic: ``(3, 3)`` camera intrinsic matrix.
        fixation_px: Fixation ``(u, v)`` pixel coordinates (default: principal point).
        cfg: Optional :class:`FoveationConfig`.

    Returns:
        Foveated image with the same type and shape as *image* (floating dtype).
    """
    params = FoveationConfig() if cfg is None else cfg
    k = np.asarray(intrinsic, dtype=np.float64)
    ppd = math.radians(1.0) * 0.5 * (float(k[0, 0]) + float(k[1, 1]))
    sigmas = np.array(
        [0.0, *(0.5 * (2.0**i) for i in range(params.n_levels - 1))],
        dtype=np.float64,
    )

    is_torch = type(image).__module__.startswith("torch")
    if is_torch:
        import importlib  # noqa: PLC0415

        torch_mod = importlib.import_module("torch")
        squeeze_batch = image.ndim == 3
        tensor = image.unsqueeze(0) if squeeze_batch else image
        if tensor.ndim != 4:
            raise ValueError(
                f"Torch image must have shape (C, H, W) or (B, C, H, W), got {tuple(image.shape)}"
            )
        tensor_f = tensor.to(
            dtype=torch_mod.float32 if not tensor.is_floating_point() else tensor.dtype
        )
        _, _, h, w = tensor_f.shape
        theta = eccentricity_deg(h, w, k, fixation_px=fixation_px)
        sigma_px = (
            params.fovea_sigma0_deg * (1.0 + theta / params.half_resolution_deg) * ppd
        )
        level = np.interp(sigma_px, sigmas, np.arange(len(sigmas)))
        low = np.floor(level).astype(np.int64)
        high = np.minimum(low + 1, len(sigmas) - 1)
        weight = torch_mod.as_tensor(
            (level - low)[None, None, :, :],
            dtype=tensor_f.dtype,
            device=tensor_f.device,
        )
        low_t = torch_mod.as_tensor(
            low[None, None, :, :], dtype=torch_mod.long, device=tensor_f.device
        ).expand_as(tensor_f)
        high_t = torch_mod.as_tensor(
            high[None, None, :, :], dtype=torch_mod.long, device=tensor_f.device
        ).expand_as(tensor_f)

        pyramid = torch_mod.stack(
            [_torch_gaussian_blur_2d(tensor_f, float(s), torch_mod) for s in sigmas],
            dim=0,
        )
        low_gather = torch_mod.gather(pyramid, 0, low_t.unsqueeze(0)).squeeze(0)
        high_gather = torch_mod.gather(pyramid, 0, high_t.unsqueeze(0)).squeeze(0)
        out = (1.0 - weight) * low_gather + weight * high_gather
        return out.squeeze(0) if squeeze_batch else out

    arr = np.asarray(image, dtype=np.float64)
    squeeze_chan = arr.ndim == 2
    if squeeze_chan:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(
            f"NumPy image must have shape (H, W) or (H, W, C), got {arr.shape}"
        )
    h, w, _ = arr.shape
    theta = eccentricity_deg(h, w, k, fixation_px=fixation_px)
    sigma_px = (
        params.fovea_sigma0_deg * (1.0 + theta / params.half_resolution_deg) * ppd
    )
    pyramid_np = np.stack(
        [
            arr if s == 0.0 else gaussian_filter(arr, sigma=(s, s, 0.0), mode="nearest")
            for s in sigmas
        ],
        axis=0,
    )
    level = np.interp(sigma_px, sigmas, np.arange(len(sigmas)))
    low = np.floor(level).astype(np.intp)
    high = np.minimum(low + 1, len(sigmas) - 1)
    weight = (level - low)[:, :, None]
    rows, cols = np.ogrid[:h, :w]
    out_np = (1.0 - weight) * pyramid_np[low, rows, cols] + weight * pyramid_np[
        high, rows, cols
    ]
    return out_np[:, :, 0] if squeeze_chan else out_np


def crop_foveal_patch(
    image: np.ndarray,
    patch_size: tuple[int, int] = (64, 64),
    fixation_px: tuple[float, float] | np.ndarray | None = None,
) -> np.ndarray:
    """Extract a clamped high-acuity foveal crop ``(ph, pw, ...)`` around *fixation_px*."""
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got shape {arr.shape}")
    h, w = arr.shape[:2]
    ph, pw = int(patch_size[0]), int(patch_size[1])
    if ph < 1 or pw < 1 or ph > h or pw > w:
        raise ValueError(f"patch_size {(ph, pw)} invalid for image {(h, w)}")
    if fixation_px is None:
        cx, cy = w // 2, h // 2
    else:
        fix = np.asarray(fixation_px, dtype=np.float64).ravel()
        cx, cy = int(round(float(fix[0]))), int(round(float(fix[1])))
    x0 = min(max(0, cx - pw // 2), w - pw)
    y0 = min(max(0, cy - ph // 2), h - ph)
    return arr[y0 : y0 + ph, x0 : x0 + pw].copy()


__all__ = [
    "FoveationConfig",
    "crop_foveal_patch",
    "eccentricity_deg",
    "foveate_image",
    "intrinsic_from_fovy",
]
