"""End-to-end BTFSC-Net pipeline, wiring together every stage in Table 1.

    Step 1  HPWF preprocessing              -> btfsc_net.hpwf
    Step 2  DLCNN-based Fusion-Net + REA    -> btfsc_net.rea, btfsc_net.fusion
    Step 3  HFCMIK segmentation             -> btfsc_net.segmentation
    Step 4  Hybrid features + DLPNN         -> btfsc_net.features, btfsc_net.classifier
    Step 5  Objective evaluation            -> btfsc_net.metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .classifier import DLPNN, classify
from .features import extract_hybrid_features, FEATURE_NAMES
from .fusion import DLCNNFusionNet, fuse_images
from .hpwf import hpwf_denoise_color
from .segmentation import hfcmik_segment, SegmentationResult


@dataclass
class PipelineResult:
    denoised_mr: np.ndarray
    denoised_secondary: np.ndarray
    fused_image: np.ndarray
    segmentation: SegmentationResult
    features: np.ndarray
    feature_names: list = field(default_factory=lambda: list(FEATURE_NAMES))
    classification: dict | None = None


class BTFSCNet:
    """The full BTFSC-Net pipeline (Table 1) as a single callable object.

    Parameters
    ----------
    n_clusters: number of HFCMIK clusters (default 3: background, tissue,
        tumor).
    classifier: an optional pre-trained :class:`~btfsc_net.classifier.DLPNN`
        instance. If omitted, a freshly (randomly) initialised network is
        used -- see ``btfsc_net.classifier.train_dlpnn`` to train one on
        your own labeled data first.
    """

    def __init__(self, n_clusters: int = 3, classifier: DLPNN | None = None):
        self.n_clusters = n_clusters
        self.fusion_net = DLCNNFusionNet()
        self.classifier = classifier or DLPNN(num_classes=2)

    def run(
        self,
        mr_image: np.ndarray,
        secondary_image: np.ndarray,
        image_type: str = "CT",
        class_names=("benign", "malignant"),
        tumor_seed_point: tuple[int, int] | None = None,
    ) -> PipelineResult:
        """Run MRI + (CT|PET|SPECT) through the full BTFSC-Net pipeline.

        Parameters
        ----------
        mr_image: 2D grayscale MRI array.
        secondary_image: 2D grayscale CT array, or a grayscale/RGB
            PET/SPECT array, matching ``image_type``.
        image_type: "CT", "PET", or "SPECT".
        tumor_seed_point: optional (row, col) known to lie inside the
            lesion (e.g., from a radiologist click or another detector).
            When given, HFCMIK picks the cluster containing that point as
            the tumor region instead of guessing from cluster size/
            intensity alone -- see :func:`btfsc_net.segmentation.hfcmik_segment`.
        """
        # Step 1: HPWF preprocessing (denoise both inputs).
        denoised_mr = hpwf_denoise_color(mr_image)
        denoised_secondary = hpwf_denoise_color(secondary_image)

        # Step 2: REA + DLCNN-based Fusion-Net.
        fused = fuse_images(denoised_mr, denoised_secondary, image_type=image_type, net=self.fusion_net)

        fused_gray = fused.mean(axis=-1) if fused.ndim == 3 else fused

        # Step 3: HFCMIK segmentation -> locate the tumor region.
        seg = hfcmik_segment(
            fused_gray, n_clusters=self.n_clusters, tumor_seed_point=tumor_seed_point
        )

        # Step 4: hybrid feature extraction (RDWT + GLCM + color stats)
        # from the segmented tumor region, then DLPNN classification.
        tumor_only = np.where(seg.tumor_mask, fused_gray, 0.0)
        feats = extract_hybrid_features(tumor_only)
        classification = classify(self.classifier, tumor_only, class_names=class_names)

        return PipelineResult(
            denoised_mr=denoised_mr,
            denoised_secondary=denoised_secondary,
            fused_image=fused,
            segmentation=seg,
            features=feats,
            classification=classification,
        )
