"""Pixel-aligned RGB comparison, independent of Streamlit's rerun lifecycle."""
from base64 import b64encode
from html import escape
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def aligned_pair(original: np.ndarray, reconstructed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reject mismatched grids instead of silently resizing or cropping pixels."""
    pair = tuple(np.asarray(image) for image in (original, reconstructed))
    for image in pair:
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("Expected a nonempty RGB image.")
        if image.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB pixels.")
    if pair[0].shape != pair[1].shape:
        raise ValueError("Comparison images must have identical dimensions.")
    return pair


def difference_map(original: np.ndarray, reconstructed: np.ndarray, amplification: float = 1) -> np.ndarray:
    """Absolute per-channel error; multiply before clipping, never uint8-wrap."""
    original, reconstructed = aligned_pair(original, reconstructed)
    if not np.isfinite(amplification) or amplification < 1:
        raise ValueError("Amplification must be finite and at least 1.")
    difference = np.subtract(original, reconstructed, dtype=np.int16)
    np.abs(difference, out=difference)
    return amplify_difference(difference.astype(np.uint8), amplification)


def amplify_difference(difference: np.ndarray, amplification: float = 1) -> np.ndarray:
    """Amplify a byte error map; cap the factor since every nonzero byte saturates."""
    if not np.isfinite(amplification) or amplification < 1:
        raise ValueError("Amplification must be finite and at least 1.")
    scaled = difference.astype(np.float32)
    scaled *= min(amplification, 255)
    np.clip(scaled, 0, 255, out=scaled)
    return scaled.astype(np.uint8)


def png_url(image: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return "data:image/png;base64," + b64encode(buffer.getvalue()).decode("ascii")


def comparison_html(original: np.ndarray, reconstructed: np.ndarray, method: str, k: int, state_key: str, left_label: str = "Original", original_url: str | None = None, reconstructed_url: str | None = None) -> str:
    original, reconstructed = aligned_pair(original, reconstructed)
    template = Path(__file__).with_name("image_comparison.html").read_text()
    difference = difference_map(original, reconstructed)
    values = {
        "LEFT": original_url or png_url(original), "RIGHT": reconstructed_url or png_url(reconstructed),
        "DIFF": png_url(difference),
        "AMP": png_url(amplify_difference(difference, 4)),
        "TITLE": escape(f"{method} · k={k}"), "LABEL": escape(left_label),
        "KEY": escape(state_key, quote=True), "WIDTH": str(original.shape[1]),
        "HEIGHT": str(original.shape[0]),
    }
    for key, value in values.items():
        template = template.replace("@@" + key + "@@", value)
    return template


def render_comparison(original, reconstructed, method, k, state_key, left_label="Original"):
    render_comparison_html(comparison_html(original, reconstructed, method, k, state_key, left_label))


def render_comparison_html(html: str):
    import streamlit as st

    if hasattr(st, "iframe"):
        st.iframe(html, height=650)
    else:
        import streamlit.components.v1 as components

        components.html(html, height=650, scrolling=False)
