"""Zero-GPU image density heuristics for router fallback."""

from dataclasses import dataclass
from typing import Union

import numpy as np
from PIL import Image

from config.settings import (
    DENSITY_ANALYSIS_MAX_DIM,
    DENSITY_CONTRAST_THRESHOLD,
    DENSITY_EDGE_MID,
    DENSITY_EDGE_THRESHOLD,
    DENSITY_EDGE_VERY_HIGH,
    DENSITY_RESOLUTION_THRESHOLD,
)


@dataclass
class DensityReport:
    edge_density: float
    resolution_flag: bool
    low_contrast_flag: bool
    density_override: bool


def _to_grayscale_array(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(image, Image.Image):
        img = image.convert("L")
        arr = np.array(img)
    else:
        arr = image
        if arr.ndim == 3:
            arr = np.mean(arr, axis=2)
    return arr.astype(np.float32)


def _resize_for_analysis(arr: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = arr.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return arr
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    pil = Image.fromarray(arr.astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return np.array(pil).astype(np.float32)


def _canny_edge_density(arr: np.ndarray) -> float:
    try:
        import cv2
        edges = cv2.Canny(arr.astype(np.uint8), 50, 150)
        return float(np.count_nonzero(edges)) / float(edges.size)
    except ImportError:
        # PIL/numpy Sobel fallback
        gx = np.zeros_like(arr)
        gy = np.zeros_like(arr)
        gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
        gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
        mag = np.sqrt(gx ** 2 + gy ** 2)
        threshold = np.percentile(mag, 85)
        return float(np.count_nonzero(mag > threshold)) / float(mag.size)


def analyze_image_density(image: Image.Image) -> DensityReport:
    """Analyze document image for dense micro-text / low contrast."""
    orig_w, orig_h = image.size
    arr = _to_grayscale_array(image)
    arr = _resize_for_analysis(arr, DENSITY_ANALYSIS_MAX_DIM)

    contrast_std = float(np.std(arr))
    low_contrast_flag = contrast_std < DENSITY_CONTRAST_THRESHOLD
    resolution_flag = (
        orig_w > DENSITY_RESOLUTION_THRESHOLD or orig_h > DENSITY_RESOLUTION_THRESHOLD
    )
    edge_density = _canny_edge_density(arr)

    density_override = (
        edge_density > DENSITY_EDGE_VERY_HIGH
        or (
            edge_density > DENSITY_EDGE_THRESHOLD
            and low_contrast_flag
        )
        or (
            low_contrast_flag
            and edge_density > DENSITY_EDGE_MID
        )
    )

    return DensityReport(
        edge_density=edge_density,
        resolution_flag=resolution_flag,
        low_contrast_flag=low_contrast_flag,
        density_override=density_override,
    )
