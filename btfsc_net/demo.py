#!/usr/bin/env python3
"""End-to-end demo of the BTFSC-Net re-implementation.

Since the BraTS2020 dataset used by the paper isn't bundled with this
repository, this script synthesizes a pair of MRI/CT-like brain phantoms
with a known tumor region, then runs the *entire* published pipeline on
them:

    HPWF denoise -> REA + DLCNN Fusion-Net -> HFCMIK segmentation
    -> RDWT/GLCM/color hybrid features -> DLPNN classification
    -> objective evaluation (Sets 1 & 2 from Table 1)

To run on your own MRI/CT (or MRI/PET, MRI/SPECT) pair instead, replace
``make_synthetic_pair`` below with e.g. ``skimage.io.imread`` calls on
your own files (grayscale, same shape).

Usage
-----
    python demo.py [--outdir outputs] [--image-type CT]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from skimage.io import imsave

from btfsc_net.metrics import evaluate_fusion, evaluate_segmentation
from btfsc_net.pipeline import BTFSCNet


def make_synthetic_pair(size: int = 128, seed: int = 0):
    """Build a synthetic MRI-like and CT-like brain phantom sharing a
    tumor lesion, plus the ground-truth tumor mask, purely for
    demonstration (no real patient data is used or required)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2

    # Skull + brain tissue (two concentric ellipses).
    brain_r = size * 0.42
    skull_r = size * 0.46
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    brain_mask = dist < brain_r
    skull_mask = (dist >= brain_r) & (dist < skull_r)

    # Tumor lesion: a bright blob off-center inside the brain.
    tcy, tcx = cy - size * 0.1, cx + size * 0.12
    tumor_r = size * 0.09
    tumor_mask = np.sqrt((yy - tcy) ** 2 + (xx - tcx) ** 2) < tumor_r

    mri = np.zeros((size, size), dtype=np.float64)
    mri[brain_mask] = 120 + 10 * rng.standard_normal(int(brain_mask.sum()))
    mri[skull_mask] = 220 + 5 * rng.standard_normal(int(skull_mask.sum()))
    mri[tumor_mask] = 200 + 8 * rng.standard_normal(int(tumor_mask.sum()))
    mri = np.clip(mri, 0, 255)

    ct = np.zeros((size, size), dtype=np.float64)
    ct[brain_mask] = 60 + 8 * rng.standard_normal(int(brain_mask.sum()))
    ct[skull_mask] = 250 + 4 * rng.standard_normal(int(skull_mask.sum()))
    ct[tumor_mask] = 150 + 10 * rng.standard_normal(int(tumor_mask.sum()))
    ct = np.clip(ct, 0, 255)

    return mri, ct, tumor_mask, (int(tcy), int(tcx))


def _to_png(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--image-type", default="CT", choices=["CT", "PET", "SPECT"])
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)  # reproducible Fusion-Net / DLPNN weight init

    print(f"[1/4] Synthesizing an MRI/{args.image_type} phantom pair ({args.size}x{args.size})...")
    mri, secondary, tumor_gt, seed_point = make_synthetic_pair(size=args.size)
    imsave(outdir / "input_mri.png", _to_png(mri), check_contrast=False)
    imsave(outdir / f"input_{args.image_type.lower()}.png", _to_png(secondary), check_contrast=False)

    print("[2/4] Running the full BTFSC-Net pipeline "
          "(HPWF -> REA+Fusion-Net -> HFCMIK -> RDWT/GLCM -> DLPNN)...")
    print(f"      (HFCMIK is unsupervised; we pass it the known lesion seed point {seed_point}"
          " to pick the tumor cluster, same as a radiologist-click prior would)")
    net = BTFSCNet(n_clusters=3)
    result = net.run(mri, secondary, image_type=args.image_type, tumor_seed_point=seed_point)

    imsave(outdir / "denoised_mri.png", _to_png(result.denoised_mr), check_contrast=False)
    imsave(outdir / "denoised_secondary.png", _to_png(result.denoised_secondary), check_contrast=False)
    imsave(outdir / "fused_image.png", _to_png(result.fused_image), check_contrast=False)
    imsave(outdir / "segmentation_mask.png", _to_png(result.segmentation.tumor_mask), check_contrast=False)

    print(f"[3/4] Wrote denoised/fused/segmented images to {outdir}/")

    fused_gray = result.fused_image.mean(axis=-1) if result.fused_image.ndim == 3 else result.fused_image
    fusion_metrics = evaluate_fusion(mri, secondary, fused_gray)
    seg_metrics = evaluate_segmentation(tumor_gt, result.segmentation.tumor_mask)

    print("\n[4/4] Results")
    print("=" * 60)
    print("Fusion quality (Table 1, Objective Evaluation-Set 1):")
    for k, v in fusion_metrics.as_dict().items():
        print(f"  {k:8s}: {v:.4f}")

    print("\nSegmentation quality vs. synthetic ground truth "
          "(Table 1, Objective Evaluation-Set 2; illustrative only -- HFCMIK here "
          "is fully unsupervised with randomly-initialized Fusion-Net weights, "
          "so this is not a benchmark reproduction of the paper's Table 8):")
    for k, v in seg_metrics.items():
        print(f"  {k:8s}: {v:.2f}%")

    print("\nHybrid feature vector (GLCM + RDWT + color stats,"
          f" {len(result.features)} dims):")
    for name, value in zip(result.feature_names, result.features):
        print(f"  {name:20s}: {value:.4f}")

    print("\nDLPNN classification (untrained weights -- see README to train "
          "on labeled data for meaningful predictions):")
    print(f"  Prediction   : {result.classification['prediction']}")
    for name, p in result.classification["probabilities"].items():
        print(f"  P({name:9s}) : {p:.4f}")

    print("\nDone. See the outputs/ directory for saved images.")


if __name__ == "__main__":
    main()
