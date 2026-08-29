"""Objective evaluation metrics, matching Table 1's three evaluation sets
and Tables 6-9 of the BTFSC-Net paper.

    Set 1 (Fusion-Net):        Entropy, MI, UQI/SSIM, STD, PSNR
    Set 2 (HFCMIK):            SACC, SSEN, SPEC, SPR, SNPV, SFPR, SFDR,
                                SFNR, SF1, SMCC
    Set 3 (DLPNN):             CACC, CSEN, CPEC, CPR, CNPV, CFPR, CFDR,
                                CFNR, CF1, CMCC

Sets 2 and 3 are the same confusion-matrix-derived metrics (accuracy,
sensitivity/recall, specificity, precision, NPV, FPR, FDR, FNR, F1, MCC),
just reported under an "S" (segmentation) or "C" (classification) prefix,
so both are computed by one shared implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity as _ssim


# --------------------------------------------------------------------------
# Set 1: Fusion quality metrics
# --------------------------------------------------------------------------

def entropy(image: np.ndarray, bins: int = 256) -> float:
    """Shannon entropy of an image's intensity histogram."""
    hist, _ = np.histogram(image, bins=bins, range=(image.min(), image.max() + 1e-12))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def standard_deviation(image: np.ndarray) -> float:
    return float(np.std(image))


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 256) -> float:
    """Mutual information between two (same-shape) images, from their
    joint histogram."""
    a = a.ravel()
    b = b.ravel()
    joint_hist, _, _ = np.histogram2d(a, b, bins=bins)
    p_joint = joint_hist / joint_hist.sum()
    p_a = p_joint.sum(axis=1, keepdims=True)
    p_b = p_joint.sum(axis=0, keepdims=True)

    nonzero = p_joint > 0
    denom = p_a @ p_b
    return float(np.sum(p_joint[nonzero] * np.log2(p_joint[nonzero] / denom[nonzero])))


def psnr(reference: np.ndarray, test: np.ndarray, data_range: float | None = None) -> float:
    reference = reference.astype(np.float64)
    test = test.astype(np.float64)
    mse = float(np.mean((reference - test) ** 2))
    if mse < 1e-12:
        return float("inf")
    data_range = data_range or float(reference.max() - reference.min()) or 255.0
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


def uqi(reference: np.ndarray, test: np.ndarray) -> float:
    """Universal Quality Index (Wang & Bovik, 2002)."""
    x = reference.astype(np.float64).ravel()
    y = test.astype(np.float64).ravel()
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    numerator = 4 * cov * mx * my
    denominator = (vx + vy) * (mx ** 2 + my ** 2)
    if denominator < 1e-12:
        return 1.0 if numerator < 1e-12 else 0.0
    return float(numerator / denominator)


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    data_range = float(reference.max() - reference.min()) or 1.0
    return float(_ssim(reference, test, data_range=data_range))


@dataclass
class FusionMetrics:
    entropy: float
    mi: float
    psnr: float
    ssim: float
    uqi: float
    std: float

    def as_dict(self) -> dict:
        return {
            "Entropy": self.entropy,
            "MI": self.mi,
            "PSNR": self.psnr,
            "SSIM": self.ssim,
            "UQI": self.uqi,
            "STD": self.std,
        }


def evaluate_fusion(source1: np.ndarray, source2: np.ndarray, fused: np.ndarray) -> FusionMetrics:
    """Table 1 "Objective Evaluation-Set 1": compare the fused image
    against both source images, averaging the reference-based metrics.
    """
    source1 = np.resize(source1, fused.shape).astype(np.float64)
    source2 = np.resize(source2, fused.shape).astype(np.float64)
    fused = fused.astype(np.float64)

    mi_val = 0.5 * (mutual_information(source1, fused) + mutual_information(source2, fused))
    psnr_val = 0.5 * (psnr(source1, fused) + psnr(source2, fused))
    ssim_val = 0.5 * (ssim(source1, fused) + ssim(source2, fused))
    uqi_val = 0.5 * (uqi(source1, fused) + uqi(source2, fused))

    return FusionMetrics(
        entropy=entropy(fused),
        mi=mi_val,
        psnr=psnr_val,
        ssim=ssim_val,
        uqi=uqi_val,
        std=standard_deviation(fused),
    )


# --------------------------------------------------------------------------
# Sets 2 & 3: confusion-matrix (segmentation / classification) metrics
# --------------------------------------------------------------------------

@dataclass
class ConfusionMetrics:
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    npv: float
    fpr: float
    fdr: float
    fnr: float
    f1: float
    mcc: float

    def as_dict(self, prefix: str) -> dict:
        keys = {
            "accuracy": f"{prefix}ACC",
            "sensitivity": f"{prefix}SEN",
            # Table 8 spells this "SPEC" and Table 9 "CPEC" -- both are
            # just prefix + "PEC" ("S"+"PEC"="SPEC", "C"+"PEC"="CPEC").
            "specificity": f"{prefix}PEC",
            "precision": f"{prefix}PR",
            "npv": f"{prefix}NPV",
            "fpr": f"{prefix}FPR",
            "fdr": f"{prefix}FDR",
            "fnr": f"{prefix}FNR",
            "f1": f"{prefix}F1",
            "mcc": f"{prefix}MCC",
        }
        return {label: getattr(self, field) for field, label in keys.items()}


def _binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true).astype(bool).ravel()
    y_pred = np.asarray(y_pred).astype(bool).ravel()
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    return tp, tn, fp, fn


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMetrics:
    """Every metric in Table 8/9 derived from one binary confusion matrix.

    Works identically for segmentation (pixel masks) and classification
    (sample labels) -- the paper reports the same ten quantities for both,
    just under an "S" or "C" prefix (Table 1, evaluation sets 2 and 3).
    """
    tp, tn, fp, fn = _binary_confusion(y_true, y_pred)
    eps = 1e-12

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)          # a.k.a. recall
    specificity = tn / (tn + fp + eps)
    precision = tp / (tp + fp + eps)
    npv = tn / (tn + fn + eps)
    fpr = fp / (fp + tn + eps)
    fdr = fp / (fp + tp + eps)
    fnr = fn / (fn + tp + eps)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + eps)
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
    mcc = (tp * tn - fp * fn) / mcc_denom

    return ConfusionMetrics(
        accuracy=float(accuracy * 100),
        sensitivity=float(sensitivity * 100),
        specificity=float(specificity * 100),
        precision=float(precision * 100),
        npv=float(npv * 100),
        fpr=float(fpr * 100),
        fdr=float(fdr * 100),
        fnr=float(fnr * 100),
        f1=float(f1 * 100),
        mcc=float(mcc * 100),
    )


def evaluate_segmentation(mask_true: np.ndarray, mask_pred: np.ndarray) -> dict:
    return confusion_metrics(mask_true, mask_pred).as_dict(prefix="S")


def evaluate_classification(labels_true, labels_pred, positive_label=1) -> dict:
    y_true = np.asarray(labels_true) == positive_label
    y_pred = np.asarray(labels_pred) == positive_label
    return confusion_metrics(y_true, y_pred).as_dict(prefix="C")
