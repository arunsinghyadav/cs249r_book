"""Hybrid feature extraction: RDWT low-level features + GLCM texture
features + statistical color features.

Implements Section 2.4 ("Proposed Hybrid Feature Extraction"), Figure 5
(two-level RDWT), and Equations (18)-(24) of the BTFSC-Net paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt
from skimage.feature import graycomatrix, graycoprops


# --------------------------------------------------------------------------
# GLCM texture features (Equations 18-22)
# --------------------------------------------------------------------------

@dataclass
class GLCMFeatures:
    contrast: float
    homogeneity: float
    correlation: float
    asm: float
    energy: float
    entropy: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.contrast, self.homogeneity, self.correlation, self.asm, self.energy, self.entropy]
        )


def _quantize(image: np.ndarray, levels: int = 32) -> np.ndarray:
    image = image.astype(np.float64)
    lo, hi = float(np.min(image)), float(np.max(image))
    if hi - lo < 1e-12:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - lo) / (hi - lo) * (levels - 1)
    return scaled.astype(np.uint8)


def glcm_features(image: np.ndarray, levels: int = 32, distance: int = 1) -> GLCMFeatures:
    """Equations (18)-(22): contrast, homogeneity, correlation, ASM, energy,
    computed from the gray-level co-occurrence matrix S_{a,b}.
    """
    q = _quantize(image, levels=levels)
    glcm = graycomatrix(
        q, distances=[distance], angles=[0], levels=levels, symmetric=True, normed=True
    )

    contrast = float(graycoprops(glcm, "contrast")[0, 0])       # Eq. 18
    homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])  # Eq. 19
    correlation = float(graycoprops(glcm, "correlation")[0, 0])  # Eq. 20
    asm = float(graycoprops(glcm, "ASM")[0, 0])                  # Eq. 21
    energy = float(graycoprops(glcm, "energy")[0, 0])            # Eq. 22, sqrt(ASM)

    p = glcm[:, :, 0, 0]
    p_nonzero = p[p > 0]
    entropy = float(-np.sum(p_nonzero * np.log2(p_nonzero)))

    return GLCMFeatures(contrast, homogeneity, correlation, asm, energy, entropy)


# --------------------------------------------------------------------------
# Two-level RDWT low-level features (Figure 5)
# --------------------------------------------------------------------------

@dataclass
class RDWTLevelFeatures:
    entropy: float
    energy: float
    correlation: float

    def to_array(self) -> np.ndarray:
        return np.array([self.entropy, self.energy, self.correlation])


def _band_entropy_energy_correlation(band: np.ndarray, levels: int = 32) -> RDWTLevelFeatures:
    """Entropy/energy/correlation of a wavelet sub-band, via the same
    GLCM-style co-occurrence measures used in Equations (18)-(22)."""
    g = glcm_features(band, levels=levels)
    return RDWTLevelFeatures(entropy=g.entropy, energy=g.energy, correlation=g.correlation)


@dataclass
class RDWTFeatures:
    level1: RDWTLevelFeatures
    level2: RDWTLevelFeatures
    bands: dict  # LL1, LH1, HL1, HH1, LL2, LH2, HL2, HH2

    def to_array(self) -> np.ndarray:
        return np.concatenate([self.level1.to_array(), self.level2.to_array()])


def two_level_rdwt(image: np.ndarray, wavelet: str = "haar") -> RDWTFeatures:
    """Figure 5: two-level Redundant (stationary) Discrete Wavelet
    Transform. RDWT (a.k.a. SWT) keeps every band at the input's
    resolution, unlike the plain DWT, which halves it at each level.
    """
    image = image.astype(np.float64)

    # pywt.swt2 requires each spatial dimension to be a multiple of 2**level.
    h, w = image.shape
    pad_h = (-h) % 4
    pad_w = (-w) % 4
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")

    (ll2, (hl2, lh2, hh2)), (ll1, (hl1, lh1, hh1)) = pywt.swt2(
        padded, wavelet=wavelet, level=2
    )

    bands = {
        "LL1": ll1, "LH1": lh1, "HL1": hl1, "HH1": hh1,
        "LL2": ll2, "LH2": lh2, "HL2": hl2, "HH2": hh2,
    }

    level1 = _band_entropy_energy_correlation(ll1)
    level2 = _band_entropy_energy_correlation(ll2)
    return RDWTFeatures(level1, level2, bands)


# --------------------------------------------------------------------------
# Statistical color features (Equations 23-24)
# --------------------------------------------------------------------------

def statistical_color_features(image: np.ndarray) -> np.ndarray:
    """Equations (23)-(24): mean and standard deviation of the segmented
    image, used as low-level statistical color features."""
    image = image.astype(np.float64)
    n = image.size
    mu = float(np.sum(image) / n)
    sigma = float(np.sqrt(np.sum((image - mu) ** 2) / n))
    return np.array([mu, sigma])


# --------------------------------------------------------------------------
# Hybrid feature concatenation
# --------------------------------------------------------------------------

FEATURE_NAMES = [
    "glcm_contrast", "glcm_homogeneity", "glcm_correlation", "glcm_asm", "glcm_energy", "glcm_entropy",
    "rdwt_l1_entropy", "rdwt_l1_energy", "rdwt_l1_correlation",
    "rdwt_l2_entropy", "rdwt_l2_energy", "rdwt_l2_correlation",
    "mean", "std",
]


def extract_hybrid_features(segmented_image: np.ndarray, wavelet: str = "haar") -> np.ndarray:
    """Section 2.4: concatenate GLCM texture, RDWT low-level, and
    statistical color features into a single hybrid feature vector."""
    if segmented_image.ndim == 3:
        segmented_image = segmented_image.mean(axis=-1)

    glcm = glcm_features(segmented_image)
    rdwt = two_level_rdwt(segmented_image, wavelet=wavelet)
    color = statistical_color_features(segmented_image)

    return np.concatenate([glcm.to_array(), rdwt.to_array(), color])
