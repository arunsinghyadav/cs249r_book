"""Hybrid Fuzzy C-Means Integrated K-Means (HFCMIK) segmentation.

Implements Figure 4, Table 5 ("Proposed hybrid fuzzy segmentation model")
and Equations (16)-(17) of the BTFSC-Net paper.

Pipeline (Table 5):

    Step 1-2   I -> X[I]           image -> per-pixel feature vectors
    Step 3-4   U[i] = W[i]*X[i]    weighted attribute vectors (Eq. 16/17)
    Step 5     C_centroid = AKMC(U, K)   adaptive k-means++ initial centroids
    Step 6     S_sort = WeightedSorting(U)
    Step 7     D = FKMC(U, C_centroid)   fuzzy-kernel-c-means membership/distance
    Step 8     CS = FindOptimalCentroid(MIN(D))
    Step 9-10  repeat until stable, then combine all cluster segments -> S
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter


def _pixel_feature_vectors(image: np.ndarray, window: int = 5) -> np.ndarray:
    """Step 2 (ImgToVectorConvert): turn each pixel into a small feature
    vector [intensity, local mean, local std], each normalized to [0, 1].
    """
    image = image.astype(np.float64)
    local_mean = uniform_filter(image, size=window, mode="mirror")
    local_sq_mean = uniform_filter(image ** 2, size=window, mode="mirror")
    local_std = np.sqrt(np.clip(local_sq_mean - local_mean ** 2, 0, None))

    feats = np.stack([image, local_mean, local_std], axis=-1)
    flat = feats.reshape(-1, feats.shape[-1])
    lo = flat.min(axis=0)
    hi = flat.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return (flat - lo) / span


def _attribute_weights(x: np.ndarray) -> np.ndarray:
    """Weight each attribute (feature column) by its normalized variance,
    i.e., more discriminative attributes count more toward the weighted
    ranking distance (Eq. 16/17, "weightage of input W_i").
    """
    var = x.var(axis=0)
    total = var.sum()
    if total < 1e-12:
        return np.ones_like(var) / len(var)
    return var / total


def _weighted_points(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Equation (16)/(17): U_i = W ⊙ X_i, the weighted data points."""
    return x * w[None, :]


def _akmc_init(u: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Adaptive K-Means Clustering (AKMC) initial centroid selection.

    Implements a k-means++-style farthest-point-biased sampling: the first
    centroid is picked at random and each subsequent one is drawn with
    probability proportional to its squared distance from the nearest
    already-chosen centroid. This is the "distinct initial centroid
    selection" the paper credits with reaching a global (rather than
    arbitrary local) optimum and lowering the number of refinement
    iterations needed.
    """
    n = u.shape[0]
    centroids = [u[rng.integers(n)]]
    for _ in range(1, k):
        dist_sq = np.min(
            [np.sum((u - c) ** 2, axis=1) for c in centroids], axis=0
        )
        total = dist_sq.sum()
        probs = dist_sq / total if total > 1e-12 else np.full(n, 1.0 / n)
        centroids.append(u[rng.choice(n, p=probs)])
    return np.stack(centroids, axis=0)


def _kernel_distance(u: np.ndarray, centroids: np.ndarray, sigma: float) -> np.ndarray:
    """RBF-kernel-induced distance used by the Fuzzy Kernel C-Means (FKCM)
    step: d(x, c) = 2 * (1 - exp(-||x-c||^2 / (2 sigma^2))).
    """
    diff = u[:, None, :] - centroids[None, :, :]
    sq_dist = np.sum(diff ** 2, axis=-1)
    return 2.0 * (1.0 - np.exp(-sq_dist / (2.0 * sigma ** 2)))


def fuzzy_kernel_c_means(
    u: np.ndarray,
    centroids: np.ndarray,
    m: float = 2.0,
    sigma: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Table 5 Steps 6-9: FKCM refinement given AKMC initial centroids.

    Returns
    -------
    membership: (n_samples, k) fuzzy membership matrix.
    centroids: refined (k, n_features) centroid array.
    """
    n, k = u.shape[0], centroids.shape[0]
    if sigma is None:
        sigma = np.sqrt(u.var(axis=0).sum()) + 1e-8

    centroids = centroids.copy()
    membership = np.full((n, k), 1.0 / k)

    for _ in range(max_iter):
        dist = _kernel_distance(u, centroids, sigma)
        dist = np.maximum(dist, 1e-12)

        inv = dist ** (-1.0 / (m - 1.0))
        new_membership = inv / inv.sum(axis=1, keepdims=True)

        weights = (new_membership ** m)[:, :, None] * np.exp(
            -dist[:, :, None] / (2 * sigma ** 2)
        )
        numer = np.sum(weights * u[:, None, :], axis=0)
        denom = np.sum(weights, axis=0) + 1e-12  # shape (k, 1), broadcasts against numer's (k, n_features)
        new_centroids = numer / denom

        shift = np.max(np.abs(new_centroids - centroids))
        centroids, membership = new_centroids, new_membership
        if shift < tol:
            break

    return membership, centroids


@dataclass
class SegmentationResult:
    labels: np.ndarray          # (H, W) integer cluster id per pixel
    membership: np.ndarray      # (H, W, k) fuzzy membership
    centroids: np.ndarray       # (k, n_features) cluster centroids
    tumor_mask: np.ndarray      # (H, W) boolean mask of the tumor cluster
    tumor_cluster: int


def hfcmik_segment(
    image: np.ndarray,
    n_clusters: int = 3,
    m: float = 2.0,
    window: int = 5,
    random_state: int = 0,
    tumor_seed_point: tuple[int, int] | None = None,
) -> SegmentationResult:
    """Run the full HFCMIK pipeline (Table 5) on a fused grayscale image.

    Clustering itself (AKMC init + FKCM refinement) is fully unsupervised,
    exactly as in Table 5. Deciding *which* resulting cluster is "the
    tumor", however, is not specified by the paper's algorithm -- real
    systems either fall back on a heuristic or use a small piece of prior
    knowledge (e.g., a radiologist-marked seed point inside the lesion).
    Both options are supported here:

    - ``tumor_seed_point=(row, col)``: pick whichever cluster contains
      that pixel. This is the recommended, reliable choice whenever a
      seed point (from a click, a bounding box center, or another
      detector) is available.
    - Otherwise, fall back to a naive heuristic: exclude the background
      (the cluster with the lowest mean intensity -- the dark region
      outside the skull/brain tissue) and take the smallest remaining
      cluster, on the assumption that a lesion is more compact than the
      large-area tissue/bone clusters it is competing with.
    """
    if image.ndim == 3:
        image = image.mean(axis=-1)
    h, w = image.shape

    x = _pixel_feature_vectors(image, window=window)
    weights = _attribute_weights(x)
    u = _weighted_points(x, weights)  # Eq. 16/17

    rng = np.random.default_rng(random_state)
    centroids = _akmc_init(u, n_clusters, rng)  # Step 5 (AKMC)
    membership, centroids = fuzzy_kernel_c_means(u, centroids, m=m)  # Steps 6-9 (FKCM)

    labels_flat = np.argmax(membership, axis=1)  # Step 8 (FindOptimalCentroid)
    labels = labels_flat.reshape(h, w)
    membership_img = membership.reshape(h, w, n_clusters)

    background_cluster = int(np.argmin(centroids[:, 0]))

    if tumor_seed_point is not None:
        tumor_cluster = int(labels[tumor_seed_point])
    else:
        remaining = [c for c in range(n_clusters) if c != background_cluster]
        sizes = {c: int(np.sum(labels_flat == c)) for c in remaining}
        tumor_cluster = min(remaining, key=lambda c: sizes[c])
    tumor_mask = labels == tumor_cluster

    return SegmentationResult(labels, membership_img, centroids, tumor_mask, tumor_cluster)
