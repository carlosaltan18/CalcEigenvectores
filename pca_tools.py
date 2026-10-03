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
        "all_scores": centered @ eigenvectors,
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


def _quality_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, Any]:
    """Calcula métricas compartidas y un mapa de diferencias visible."""
    difference = np.abs(original.astype(float) - reconstructed.astype(float))
    mse = float(np.mean(difference**2))
    psnr = float("inf") if mse < 1e-12 else 20 * np.log10(255.0 / np.sqrt(mse))
    return {
        "mse": mse,
        "psnr": psnr,
        "difference_image": np.clip(difference * 4, 0, 255).astype(np.uint8),
    }


def color_image_pca(image: np.ndarray, n_components: int) -> Dict[str, Any]:
    """Aplica PCA independiente a R, G y B y reúne sus métricas."""
    values = np.asarray(image)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("La imagen debe tener tres canales RGB.")

    channels = [image_pca(values[:, :, channel], n_components) for channel in range(3)]
    reconstructed = np.stack([channel["reconstructed_image"] for channel in channels], axis=2)
    variances = np.array([channel["total_variance"] for channel in channels], dtype=float)
    retained = np.array(
        [channel["cumulative_ratio"][n_components - 1] for channel in channels], dtype=float
    )
    total_variance = float(variances.sum())
    weighted_retained = float(np.average(retained, weights=variances)) if total_variance > 1e-12 else 0.0
    metrics = _quality_metrics(values, reconstructed)

    return {
        "reconstructed_image": reconstructed,
        "channels": channels,
        "retained_variance": weighted_retained,
        "total_variance": total_variance,
        "storage_ratio": float(np.mean([channel["storage_ratio"] for channel in channels])),
        **metrics,
    }


def color_image_svd(image: np.ndarray, n_components: int) -> Dict[str, Any]:
    """Reconstruye cada canal RGB con una aproximación SVD de rango k."""
    values = np.asarray(image)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("La imagen debe tener tres canales RGB.")

    height, width, _ = values.shape
    reconstructions = []
    retained_energy = []
    energies = []
    for channel in range(3):
        source = values[:, :, channel].astype(float)
        u_matrix, singular_values, vt_matrix = np.linalg.svd(source, full_matrices=False)
        reconstruction = (u_matrix[:, :n_components] * singular_values[:n_components]) @ vt_matrix[:n_components, :]
        reconstructions.append(np.clip(reconstruction, 0, 255).astype(np.uint8))
        total_energy = float(np.sum(singular_values**2))
        energies.append(total_energy)
        retained_energy.append(float(np.sum(singular_values[:n_components] ** 2) / total_energy) if total_energy > 1e-12 else 0.0)

    reconstructed = np.stack(reconstructions, axis=2)
    total_energy = float(np.sum(energies))
    weighted_retained = float(np.average(retained_energy, weights=energies)) if total_energy > 1e-12 else 0.0
    stored_values = height * n_components + n_components + n_components * width
    metrics = _quality_metrics(values, reconstructed)

    return {
        "reconstructed_image": reconstructed,
        "retained_variance": weighted_retained,
        "total_variance": total_energy,
        "storage_ratio": 100 * stored_values / (height * width),
        **metrics,
    }
