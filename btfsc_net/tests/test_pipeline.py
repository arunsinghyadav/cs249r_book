"""Smoke tests for every BTFSC-Net stage, run on small synthetic images
so the suite is fast and needs no external dataset."""

import numpy as np
import pytest

from btfsc_net.classifier import DLPNN, classify, preprocess_for_classifier
from btfsc_net.features import extract_hybrid_features, glcm_features, two_level_rdwt
from btfsc_net.fusion import DLCNNFusionNet, fuse_images, rgb_to_ycbcr, ycbcr_to_rgb
from btfsc_net.hpwf import hpwf_denoise, hpwf_denoise_color
from btfsc_net.metrics import (
    confusion_metrics,
    evaluate_classification,
    evaluate_fusion,
    evaluate_segmentation,
)
from btfsc_net.pipeline import BTFSCNet
from btfsc_net.rea import robust_edge_analysis
from btfsc_net.segmentation import hfcmik_segment


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def gray_image(rng):
    return rng.uniform(0, 255, size=(48, 48))


def test_hpwf_denoise_preserves_shape_and_range(gray_image):
    denoised = hpwf_denoise(gray_image)
    assert denoised.shape == gray_image.shape
    assert np.isfinite(denoised).all()


def test_hpwf_denoise_color(rng):
    color = rng.uniform(0, 255, size=(32, 32, 3))
    out = hpwf_denoise_color(color)
    assert out.shape == color.shape


def test_rea_output_is_normalized(gray_image):
    edges = robust_edge_analysis(gray_image)
    assert edges.shape == gray_image.shape
    assert edges.min() >= 0.0
    assert edges.max() <= 1.0 + 1e-9


def test_fusion_grayscale_ct(gray_image, rng):
    ct = rng.uniform(0, 255, size=gray_image.shape)
    net = DLCNNFusionNet()
    fused = fuse_images(gray_image, ct, image_type="CT", net=net)
    assert fused.shape == gray_image.shape
    assert np.isfinite(fused).all()


def test_fusion_pet_roundtrips_color(gray_image, rng):
    pet = rng.uniform(0, 255, size=(*gray_image.shape, 3))
    net = DLCNNFusionNet()
    fused = fuse_images(gray_image, pet, image_type="PET", net=net)
    assert fused.shape == (*gray_image.shape, 3)


def test_ycbcr_roundtrip(rng):
    rgb = rng.uniform(0, 255, size=(16, 16, 3))
    y, cb, cr = rgb_to_ycbcr(rgb)
    back = ycbcr_to_rgb(y, cb, cr)
    assert np.allclose(back, rgb, atol=1.0)


def test_hfcmik_segmentation_shapes(gray_image):
    seg = hfcmik_segment(gray_image, n_clusters=3)
    assert seg.labels.shape == gray_image.shape
    assert seg.tumor_mask.shape == gray_image.shape
    assert seg.tumor_mask.dtype == bool
    assert set(np.unique(seg.labels)).issubset({0, 1, 2})


def test_hfcmik_seed_point_selects_containing_cluster(gray_image):
    seg = hfcmik_segment(gray_image, n_clusters=3, tumor_seed_point=(5, 5))
    assert seg.tumor_mask[5, 5]


def test_glcm_features_are_finite(gray_image):
    feats = glcm_features(gray_image)
    assert np.isfinite(feats.to_array()).all()


def test_two_level_rdwt_bands_shapes(gray_image):
    rdwt = two_level_rdwt(gray_image)
    for band in rdwt.bands.values():
        assert band.shape[0] >= gray_image.shape[0]
        assert band.shape[1] >= gray_image.shape[1]


def test_hybrid_feature_vector_length(gray_image):
    feats = extract_hybrid_features(gray_image)
    assert feats.shape == (14,)
    assert np.isfinite(feats).all()


def test_dlpnn_forward_pass_shape():
    model = DLPNN(num_classes=2)
    x = preprocess_for_classifier(np.random.rand(50, 50))
    import torch

    tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    logits = model(tensor)
    assert logits.shape == (1, 2)


def test_classify_returns_valid_probabilities(gray_image):
    model = DLPNN(num_classes=2)
    result = classify(model, gray_image)
    probs = list(result["probabilities"].values())
    assert result["prediction"] in ("benign", "malignant")
    assert abs(sum(probs) - 1.0) < 1e-5


def test_confusion_metrics_perfect_prediction():
    y = np.array([1, 0, 1, 1, 0])
    m = confusion_metrics(y, y)
    assert m.accuracy == pytest.approx(100.0)
    assert m.mcc == pytest.approx(100.0)


def test_evaluate_segmentation_and_classification_prefixes():
    mask_true = np.array([1, 0, 1, 1])
    mask_pred = np.array([1, 0, 0, 1])
    seg = evaluate_segmentation(mask_true, mask_pred)
    assert "SACC" in seg and "SPEC" in seg and "SSEN" in seg

    labels_true = [0, 1, 1, 0]
    labels_pred = [0, 1, 0, 0]
    cls = evaluate_classification(labels_true, labels_pred, positive_label=1)
    assert "CACC" in cls and "CPEC" in cls


def test_evaluate_fusion_metrics(gray_image, rng):
    ct = rng.uniform(0, 255, size=gray_image.shape)
    fused = 0.5 * gray_image + 0.5 * ct
    metrics = evaluate_fusion(gray_image, ct, fused)
    d = metrics.as_dict()
    assert set(d) == {"Entropy", "MI", "PSNR", "SSIM", "UQI", "STD"}
    assert np.isfinite(list(d.values())).all()


def test_full_pipeline_runs_end_to_end(gray_image, rng):
    ct = rng.uniform(0, 255, size=gray_image.shape)
    net = BTFSCNet(n_clusters=3)
    result = net.run(gray_image, ct, image_type="CT")

    assert result.fused_image.shape == gray_image.shape
    assert result.segmentation.tumor_mask.shape == gray_image.shape
    assert result.features.shape == (14,)
    assert result.classification["prediction"] in ("benign", "malignant")
