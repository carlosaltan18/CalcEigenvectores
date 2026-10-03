"""Session-owned image factors: decompose once, reconstruct on rank changes."""
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class ChannelFactors:
    u: np.ndarray
    singular: np.ndarray
    vt: np.ndarray
    mean: np.ndarray | None


def preview_image(image: Image.Image, max_side: int | None) -> np.ndarray:
    """Keep the full frame; only the explicitly selected preview is resampled."""
    if max_side is not None and max_side < 2:
        raise ValueError("Preview size must be at least 2.")
    image = image.convert("RGB")
    if max_side is not None:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(image)


class ImageWorkspace:
    """Own at most two decompositions and one current result per method.

    Store in session_state, not a global serialized cache: factors are independent
    of k and must not be copied or accumulated for every slider position.
    """

    def __init__(self, image: np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
            raise ValueError("Expected an RGB uint8 image of at least 2 × 2 pixels.")
        self.image = image
        self.factors: dict[str, list[ChannelFactors]] = {}
        self.results: dict[str, tuple[int, dict]] = {}

    def reconstruct(self, method: str, k: int) -> dict:
        if method not in ("PCA", "SVD"):
            raise ValueError("Unknown method.")
        if not isinstance(k, (int, np.integer)) or not 1 <= k <= min(self.image.shape[:2]):
            raise ValueError("Invalid rank.")
        cached = self.results.get(method)
        if cached is not None and cached[0] == k:
            return cached[1]
        if method not in self.factors:
            factors = []
            for channel in range(3):
                source = self.image[:, :, channel].astype(np.float32)
                mean = source.mean(axis=0) if method == "PCA" else None
                if mean is not None:
                    source -= mean
                # Centered SVD yields the same PCA subspaces without a W × W covariance.
                u, singular, vt = np.linalg.svd(source, full_matrices=False)
                factors.append(ChannelFactors(u, singular, vt, mean))
            self.factors[method] = factors

        reconstructed = np.empty_like(self.image)
        total_energy = retained_energy = squared_error = 0.0
        for channel, factor in enumerate(self.factors[method]):
            plane = (factor.u[:, :k] * factor.singular[:k]) @ factor.vt[:k]
            if factor.mean is not None:
                plane += factor.mean
            np.clip(plane, 0, 255, out=plane)
            # Round, rather than biasing nearly exact reconstructions down a byte.
            np.rint(plane, out=plane)
            reconstructed[:, :, channel] = plane.astype(np.uint8)
            delta = self.image[:, :, channel].astype(np.float32)
            delta -= reconstructed[:, :, channel]
            np.square(delta, out=delta)
            squared_error += float(delta.sum(dtype=np.float64))
            energy = np.square(factor.singular.astype(np.float64))
            total_energy += float(energy.sum())
            retained_energy += float(energy[:k].sum())
        height, width = self.image.shape[:2]
        mse = squared_error / self.image.size
        stored = (height + width) * k + (width if method == "PCA" else k)
        result = {
            "reconstructed_image": reconstructed,
            "mse": mse,
            "psnr": float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse)),
            "retained_variance": retained_energy / total_energy if total_energy else 0.0,
            "total_variance": total_energy,
            "storage_ratio": 100 * stored / (height * width),
        }
        self.results[method] = (k, result)
        return result
