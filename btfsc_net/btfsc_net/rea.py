"""Robust Edge Analysis (REA).

Implements Table 4 ("REA algorithm") and Equations (8)-(9) of the BTFSC-Net
paper. REA decomposes a medical image into a slope/edge-analysed
representation used to keep MR/CT/PET/SPECT images registered before
fusion.

Phase mapping (Table 4 -> code):

    Phase 1  TrainTest(X)                     -> input passthrough
    Phase 2  GaussianFiltering                -> scipy gaussian_filter
    Phase 3  CannyEdgeDetection                -> skimage.feature.canny
    Phase 4  EdgeRemovalAverageThresholdWeight -> _threshold_weak_edges
    Phase 5  LayerWiseEnergyCal (Eq. 8)        -> _layer_energy
    Phase 6  PerfectEnergyCal (slope param xi) -> _optimise_energy
    Phase 7  EnergyOptimizedImages / smooth    -> _edge_based_slope /
                                                   _energy_map (Eq. 9)
    Phase 8  Output                            -> the returned array
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from skimage.feature import canny


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _threshold_weak_edges(edges: np.ndarray, decomposed: np.ndarray) -> np.ndarray:
    """Phase 4: remove edges/detail layers below the mean edge-response
    threshold, keeping only the strong (well-supported) boundaries.
    """
    gradient = np.hypot(sobel(decomposed, axis=0), sobel(decomposed, axis=1))
    mu = float(np.mean(gradient))
    strong = edges & (gradient >= mu)
    return strong.astype(np.float64) * gradient


def _layer_energy(fined: np.ndarray, xi: float = 0.5) -> np.ndarray:
    """Phase 5/6/Eq.(8)-(9): E = xi * (|psi*R - (M-C)|^2 + |psi*R - (M+C)|^2).

    Here R is the fined edge/slope layer itself, M is its local mean and C
    its local (registration) offset approximated by the local standard
    deviation; psi is fixed to 1 (no down-sampling) since REA here operates
    on already test-extracted single-resolution images.
    """
    r = fined
    m = float(np.mean(r))
    c = float(np.std(r))
    psi = 1.0
    energy = xi * (np.abs(psi * r - (m - c)) ** 2 + np.abs(psi * r - (m + c)) ** 2)
    return energy


def _is_smooth(image: np.ndarray, threshold: float = 0.02) -> bool:
    """Classify a layer as smooth vs. non-smooth from its normalized
    gradient energy, used to pick between Phase 7's two branches.
    """
    grad = np.hypot(sobel(image, axis=0), sobel(image, axis=1))
    return float(np.mean(grad)) < threshold


def _edge_based_slope(energy: np.ndarray) -> np.ndarray:
    """Smooth branch of Phase 7: sharpen slope/edges via a gradient map
    computed directly on the (already smooth) energy image.
    """
    gx = sobel(energy, axis=0)
    gy = sobel(energy, axis=1)
    return np.hypot(gx, gy)


def _energy_map(energy: np.ndarray) -> np.ndarray:
    """Non-smooth branch of Phase 7: normalize the energy map itself."""
    return _normalize01(energy)


def robust_edge_analysis(image: np.ndarray, xi: float = 0.5) -> np.ndarray:
    """Run the full REA pipeline (Table 4) on a single-channel image.

    Parameters
    ----------
    image: 2D array (grayscale image or one color channel).
    xi: slope/energy-optimisation parameter (paper's "ideal slope
        parameter" in Eq. 9).

    Returns
    -------
    2D float64 array in [0, 1], the edge/slope-analysed image
    (I_OptimizedImage in Table 4).
    """
    x = _normalize01(image)

    # Phase 2: Gaussian filtering -> noise-free image.
    y_noise_free = gaussian_filter(x, sigma=1.0)

    # Phase 3: Canny edge detection -> decomposed image (shapes/edges).
    edges = canny(y_noise_free, sigma=1.0)

    # Phase 4: remove weak/unnecessary edges via average threshold weight.
    fined = _threshold_weak_edges(edges, y_noise_free)

    # Phase 5 & 6: layer-wise + optimised energy (Eq. 8/9).
    energy = _layer_energy(fined, xi=xi)

    # Phase 7: branch on smoothness of the optimised-energy image.
    if _is_smooth(energy):
        optimized = _edge_based_slope(energy)
    else:
        optimized = _energy_map(energy)

    # Phase 8: output.
    return _normalize01(optimized)


def robust_edge_analysis_color(image: np.ndarray, xi: float = 0.5) -> np.ndarray:
    """Apply REA per-channel to a color image, or directly if grayscale."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        return robust_edge_analysis(image, xi=xi)
    channels = [robust_edge_analysis(image[..., c], xi=xi) for c in range(image.shape[-1])]
    return np.stack(channels, axis=-1)
