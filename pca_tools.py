"""Utilidades numéricas reutilizables para PCA y compresión de imágenes."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def fit_pca(data: np.ndarray, n_components: int, standardize: bool = False) -> Dict[str, Any]:
    """Ajusta PCA con ``eigh`` sobre una matriz de observaciones por variables.

    Devuelve tanto la proyección como la reconstrucción para que la interfaz pueda
    explicar qué se conserva y qué se pierde al reducir la dimensionalidad.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("PCA requiere al menos 2 observaciones y 2 variables.")

    observations, features = values.shape
    if not 1 <= n_components <= min(observations, features):
        raise ValueError("El número de componentes no es válido.")

    mean = values.mean(axis=0)
    centered = values - mean
    scale = np.ones(features)
    if standardize:
        scale = values.std(axis=0, ddof=1)
        scale[scale < 1e-12] = 1.0
        centered = centered / scale

    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0, None)
    eigenvectors = eigenvectors[:, order]
    components = eigenvectors[:, :n_components]
    scores = centered @ components
    reconstructed_centered = scores @ components.T
    reconstructed = reconstructed_centered * scale + mean if standardize else reconstructed_centered + mean

    total_variance = float(eigenvalues.sum())
    explained_ratio = (
        eigenvalues / total_variance if total_variance > 1e-12 else np.zeros_like(eigenvalues)
    )
    reconstruction_error = float(np.mean((values - reconstructed) ** 2))

    return {
        "mean": mean,
        "scale": scale,
        "centered": centered,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "components": components,
        "scores": scores,
        "reconstructed": reconstructed,
        "explained_ratio": explained_ratio,
        "cumulative_ratio": np.cumsum(explained_ratio),
        "reconstruction_error": reconstruction_error,
        "total_variance": total_variance,
    }


def image_pca(image: np.ndarray, n_components: int) -> Dict[str, Any]:
    """Comprime una imagen gris cuyos píxeles de cada fila son observaciones."""
    image_data = np.asarray(image, dtype=float)
    result = fit_pca(image_data, n_components=n_components, standardize=False)
    reconstructed = np.clip(result["reconstructed"], 0, 255).astype(np.uint8)
    mse = float(np.mean((image_data - reconstructed.astype(float)) ** 2))
    psnr = float("inf") if mse < 1e-12 else 20 * np.log10(255.0 / np.sqrt(mse))
    height, width = image_data.shape
    pca_values = height * n_components + width * n_components + width
    result.update(
        {
            "reconstructed_image": reconstructed,
            "mse": mse,
            "psnr": psnr,
            "storage_ratio": 100 * pca_values / (height * width),
        }
    )
    return result
