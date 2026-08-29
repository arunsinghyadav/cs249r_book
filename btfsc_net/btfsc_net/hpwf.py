"""Hybrid Probabilistic Wiener Filter (HPWF).

Implements Table 2 ("HPWF Approach") and Equations (1)-(7) of:

    Yadav et al., "A Feature Extraction Using Probabilistic Neural Network
    and BTFSC-Net Model with Deep Learning for Brain Tumor Classification",
    J. Imaging 2023, 9, 10.

The filter is a spatially-adaptive denoiser: it estimates a Gaussian-noise
deviation from the image using a Laplacian-of-Gaussian style mask
(Eq. 2), picks a mean-filter window size from that deviation (Eq. 3),
mean-filters the image (Eq. 4), builds a difference mask between the raw
and mean-filtered image (Eq. 5), keeps only the pixels whose difference is
below the image mean (Eq. 6), and finally denoises by weighted-averaging
the surviving pixel set with the mean-filtered image (Eq. 7).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, uniform_filter

# 3x3 discrete Laplacian-of-Gaussian style mask used to probe high-frequency
# (noise-like) energy in the image, as referenced by Equation (2).
_NOISE_MASK = np.array(
    [
        [1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1],
    ],
    dtype=np.float64,
)


def estimate_gaussian_noise_sigma(x: np.ndarray) -> float:
    """Equation (2): sigma_GN = 1/(M*N) * sum(|X * MASK|)."""
    x = x.astype(np.float64)
    m, n = x.shape[:2]
    response = convolve(x, _NOISE_MASK, mode="mirror")
    return float(np.sum(np.abs(response)) / (m * n))


def mask_size_from_sigma(sigma_gn: float) -> int:
    """Equation (3): pick a 3x3 window for low noise, 5x5 otherwise."""
    return 3 if sigma_gn < 20 else 5


def hpwf_denoise(x: np.ndarray) -> np.ndarray:
    """Run the full HPWF pipeline (Table 2) on a single-channel image.

    Parameters
    ----------
    x: 2D array (grayscale image / one color channel), any numeric dtype.

    Returns
    -------
    2D float64 array, the denoised image Y_ij.
    """
    x = x.astype(np.float64)

    # Step 2: estimate noise deviation.
    sigma_gn = estimate_gaussian_noise_sigma(x)

    # Step 3: pick window size w x w from sigma_GN.
    w = mask_size_from_sigma(sigma_gn)

    # Step 4: mean filter (low-level noise removal), Eq. (4).
    x_mean = uniform_filter(x, size=w, mode="mirror")

    # Step 5: absolute difference mask, Eq. (5).
    diff = np.abs(x - x_mean)

    # Step 6: threshold by the global mean of the difference mask and keep
    # either the mean-filtered pixel or the difference pixel accordingly.
    mu = float(np.mean(diff))
    v_stack = np.where(diff < mu, diff, x_mean)

    # Step 7: weighted average -> here realised as a local (window-sized)
    # mean of the surviving pixel set, producing the denoised image Y_ij.
    y = uniform_filter(v_stack, size=w, mode="mirror")
    return y


def hpwf_denoise_color(image: np.ndarray) -> np.ndarray:
    """Apply :func:`hpwf_denoise` independently to every channel of a
    color (H, W, C) image, or directly if the image is already grayscale.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        return hpwf_denoise(image)
    channels = [hpwf_denoise(image[..., c]) for c in range(image.shape[-1])]
    return np.stack(channels, axis=-1)
