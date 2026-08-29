"""DLCNN-based Fusion-Net and the proposed fusion strategy.

Implements Figure 2/3, Table 3 ("Proposed Fusion algorithm"), Table 4
(REA, see :mod:`btfsc_net.rea`) and Equations (10)-(15) of the BTFSC-Net
paper, plus the RGB<->YCbCr conversions used for PET/SPECT color inputs.

Design note
-----------
Figure 3 of the paper depicts two CNN sketches that both terminate in
global-average-pooling + a 2-unit fully-connected layer (i.e., a *change
classification* head), which is inconsistent with the network's stated
job of producing a fused *image* that downstream HFCMIK segmentation and
RDWT/GLCM feature extraction operate on. We follow Equations (10)-(15)
literally for the feature-extraction/combination path (F1, F2 -> F3, F4
-> F_R) and use the resulting fused feature-activity maps as
spatially-varying pixel weights that blend the HPWF-denoised MR and
CT/PET/SPECT images -- exactly what Section 2.2 describes in prose
("The fused output is then produced by combining the results from the
HPWF and REA."). This keeps every named component (F1..F4, F_R, W1..W3,
B1..B3, ReLU) real and load-bearing while yielding an actual fused image
of the same resolution as the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hpwf import hpwf_denoise
from .rea import robust_edge_analysis


# --------------------------------------------------------------------------
# Color conversions (Table 3: RGB2YCbCrColorCon / YCbCr2RGBColorCon)
# --------------------------------------------------------------------------

def rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB2YCbCrColorCon: split a color (PET/SPECT) image into Y, Cb, Cr."""
    rgb = rgb.astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """YCbCr2RGBColorCon: recombine Y, Cb, Cr back into an RGB image."""
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255)


# --------------------------------------------------------------------------
# DLCNN-based Fusion-Net (Equations 10-15)
# --------------------------------------------------------------------------

class DLCNNFusionNet(nn.Module):
    """Two-branch fusion network following Equations (10)-(15).

    F1 = ReLU(W1 * X_MR + B1)             (Eq. 10)
    F2 = ReLU(W1 * X_CT + B1)             (Eq. 11)   -- shared weights W1,B1
    F3 = ReLU(W2 * (F1 + F2) + B2)        (Eq. 13)
    F4 = ReLU(W2 * (F2 + F1) + B2)        (Eq. 14)
    F_R = ReLU(W3 * concat(F3, F4) + B3)  (Eq. 15)
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        # W1, B1: shared 3x3 conv applied to both MR and CT branches.
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        # W2, B2: 3x3 conv combining the two branches (MaxPooling-stage conv).
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # W3, B3: fuses the concatenated F3/F4 activations into F_R.
        self.conv3 = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)

    def forward(self, x_mr: torch.Tensor, x_ct: torch.Tensor):
        f1 = F.relu(self.conv1(x_mr))
        f2 = F.relu(self.conv1(x_ct))
        f3 = F.relu(self.conv2(f1 + f2))
        f4 = F.relu(self.conv2(f2 + f1))
        f_r = F.relu(self.conv3(torch.cat([f3, f4], dim=1)))
        return f1, f2, f_r


def _activity_weight_maps(f1: torch.Tensor, f2: torch.Tensor, f_r: torch.Tensor):
    """Turn the branch feature maps into per-pixel fusion weights.

    Each branch's "activity level" is its L1-norm across feature channels
    (a standard deep-fusion activity measure); F_R re-weights how much the
    combined evidence should trust MR vs. CT/PET/SPECT at that pixel.
    """
    act_mr = f1.abs().mean(dim=1, keepdim=True)
    act_ct = f2.abs().mean(dim=1, keepdim=True)
    act_r = f_r.abs().mean(dim=1, keepdim=True) + 1e-8

    stacked = torch.cat([act_mr, act_ct], dim=1)
    weights = torch.softmax(stacked * act_r, dim=1)
    return weights[:, 0:1], weights[:, 1:2]


@dataclass
class FusionResult:
    fused: np.ndarray            # fused grayscale/Y-channel image, float64 in [0, 255]
    weight_mr: np.ndarray        # per-pixel trust weight given to the MR image
    weight_secondary: np.ndarray  # per-pixel trust weight given to CT/PET/SPECT
    mr_edge: np.ndarray          # REA output for the MR image
    secondary_edge: np.ndarray   # REA output for the CT/PET/SPECT image


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img.astype(np.float32))
    return t.unsqueeze(0).unsqueeze(0)


def fuse_grayscale(
    mr: np.ndarray,
    secondary: np.ndarray,
    net: DLCNNFusionNet | None = None,
) -> FusionResult:
    """Table 3, Steps 1-4 for a single-channel secondary image (CT).

    Both inputs must be 2D arrays of the same shape.
    """
    if mr.shape != secondary.shape:
        raise ValueError("MR and secondary image must share the same shape")
    net = net or DLCNNFusionNet()
    net.eval()

    # Step 1: HPWF pre-processing (denoise).
    mr_d = hpwf_denoise(mr)
    sec_d = hpwf_denoise(secondary)

    # Step 2: REA image decomposition -> edge/slope-analysed images.
    mr_edge = robust_edge_analysis(mr_d)
    sec_edge = robust_edge_analysis(sec_d)

    # Step 3: F_features <- HPWF(E_edgeSlopeImg) -- smooth the edge maps.
    mr_feat = hpwf_denoise(mr_edge * 255.0)
    sec_feat = hpwf_denoise(sec_edge * 255.0)

    with torch.no_grad():
        f1, f2, f_r = net(_to_tensor(mr_feat), _to_tensor(sec_feat))
        w_mr, w_sec = _activity_weight_maps(f1, f2, f_r)
        w_mr = w_mr.squeeze().numpy()
        w_sec = w_sec.squeeze().numpy()

    # Step 4: fused output = spatially-weighted blend of the denoised inputs.
    fused = w_mr * mr_d + w_sec * sec_d
    return FusionResult(fused, w_mr, w_sec, mr_edge, sec_edge)


def fuse_images(
    mr: np.ndarray,
    secondary: np.ndarray,
    image_type: str = "CT",
    net: DLCNNFusionNet | None = None,
) -> np.ndarray:
    """Full Table 3 algorithm: MRI + (CT | PET | SPECT) -> fused image.

    Parameters
    ----------
    mr: grayscale MRI image, 2D array.
    secondary: CT image (2D grayscale) or PET/SPECT image (2D grayscale or
        3-channel color array).
    image_type: one of "CT", "PET", "SPECT".
    net: optional pre-built :class:`DLCNNFusionNet` (a fresh one is created
        otherwise).

    Returns
    -------
    Fused image: 2D grayscale array for CT, or a 3-channel RGB array for
    PET/SPECT (colour is restored via YCbCr2RGBColorCon, Table 3 Step 4).
    """
    image_type = image_type.upper()
    if image_type == "CT":
        result = fuse_grayscale(mr, secondary, net=net)
        return result.fused

    if image_type in ("PET", "SPECT"):
        secondary = np.asarray(secondary, dtype=np.float64)
        if secondary.ndim == 2:
            secondary = np.stack([secondary] * 3, axis=-1)
        y, cb, cr = rgb_to_ycbcr(secondary)
        result = fuse_grayscale(mr, y, net=net)
        return ycbcr_to_rgb(result.fused, cb, cr)

    raise ValueError(f"Unsupported image_type: {image_type!r}")
