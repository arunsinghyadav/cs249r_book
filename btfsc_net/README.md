# BTFSC-Net — Python Implementation

A from-scratch, runnable Python re-implementation of every named component in:

> Yadav, A.S.; Kumar, S.; Karetla, G.R.; Cotrina-Aliaga, J.C.; Arias-Gonáles, J.L.; Kumar, V.; Srivastava, S.; Gupta, R.; Ibrahim, S.; Paul, R.; Naik, N.; Singla, B.; Tatkar, N.S.
> **"A Feature Extraction Using Probabilistic Neural Network and BTFSC-Net Model with Deep Learning for Brain Tumor Classification."**
> *J. Imaging* **2023**, *9*, 10. https://doi.org/10.3390/jimaging9010010

BTFSC-Net (**B**rain **T**umor **F**usion-based **S**egments and
**C**lassification) is a five-stage pipeline: denoise → fuse → segment →
extract features → classify. This package implements each stage as its
own module, following the paper's tables and equations directly.

## Module map

| Paper section | Module | What it does |
|---|---|---|
| §2.1, Table 2, Eq. (1)–(7) | `btfsc_net/hpwf.py` | Hybrid Probabilistic Wiener Filter — adaptive denoising |
| §2.2.1, Table 4, Eq. (8)–(9) | `btfsc_net/rea.py` | Robust Edge Analysis — edge/slope decomposition |
| §2.2, §2.2.2, Table 3, Figs. 2–3, Eq. (10)–(15) | `btfsc_net/fusion.py` | DLCNN Fusion-Net + RGB↔YCbCr — MRI+CT/PET/SPECT fusion |
| §2.3, Table 5, Fig. 4, Eq. (16)–(17) | `btfsc_net/segmentation.py` | HFCMIK — AKMC init + Fuzzy Kernel C-Means tumor segmentation |
| §2.4, Fig. 5, Eq. (18)–(24) | `btfsc_net/features.py` | GLCM texture + 2-level RDWT + color statistics → hybrid feature vector |
| §2.5, Fig. 6 | `btfsc_net/classifier.py` | DLPNN — CNN classifier (benign vs. malignant) |
| Table 1, §3 | `btfsc_net/metrics.py` | All three objective evaluation sets (fusion / segmentation / classification metrics) |
| Table 1 (end to end) | `btfsc_net/pipeline.py` | `BTFSCNet` — wires all five steps together |

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python demo.py
```

This synthesizes an MRI/CT brain-phantom pair (no real patient data
needed), runs it through the full pipeline, saves the denoised, fused,
and segmented images to `outputs/`, and prints the hybrid feature vector,
the DLPNN prediction, and every metric from Table 1.

To fuse PET or SPECT instead of CT: `python demo.py --image-type PET`.

### Using your own images

```python
from skimage.io import imread
from btfsc_net.pipeline import BTFSCNet

mri = imread("patient_mri.png", as_gray=True) * 255
ct  = imread("patient_ct.png",  as_gray=True) * 255

net = BTFSCNet(n_clusters=3)
result = net.run(mri, ct, image_type="CT")

result.fused_image           # fused MRI/CT image
result.segmentation.tumor_mask  # boolean tumor mask (HFCMIK)
result.features              # 14-dim hybrid feature vector (GLCM+RDWT+color)
result.classification        # {"prediction": ..., "probabilities": {...}}
```

### Training the DLPNN classifier

The paper reports these exact hyperparameters (Section 2.5), which are
the defaults of `DLPNNHyperparameters`: 369 training samples, batch size
90, 300 epochs, 25,000 iterations, learning rate 0.02.

```python
from btfsc_net.classifier import DLPNN, DLPNNHyperparameters, train_dlpnn

model = DLPNN(num_classes=2)  # 0 = benign, 1 = malignant
train_dlpnn(model, images, labels, hp=DLPNNHyperparameters())
```

`images` should be a list of 2D arrays (e.g., the fused/segmented tumor
regions produced by `BTFSCNet.run`) and `labels` the matching 0/1 ground
truth. The paper trains/evaluates on the [BraTS2020 dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation).

### Evaluation metrics (Table 1)

```python
from btfsc_net.metrics import evaluate_fusion, evaluate_segmentation, evaluate_classification

evaluate_fusion(mr_image, ct_image, fused_image)             # Entropy, MI, PSNR, SSIM, UQI, STD
evaluate_segmentation(mask_ground_truth, mask_predicted)     # SACC, SSEN, SPEC, SPR, SNPV, SFPR, SFDR, SFNR, SF1, SMCC
evaluate_classification(labels_true, labels_pred)            # CACC, CSEN, CPEC, CPR, CNPV, CFPR, CFDR, CFNR, CF1, CMCC
```

## Design notes and where the paper required interpretation

This is a faithful *engineering* reproduction, not a copy-paste of
released code (the authors did not publish any). A few places in the
paper are underspecified or internally inconsistent; here is exactly
what this implementation does and why, so results can be understood and
adjusted:

- **HPWF (Eq. 6)**: implemented literally. Pixels whose local difference
  from the mean-filtered image is *below* the global mean threshold are
  replaced by that (small) difference value rather than by their
  intensity; pixels above threshold keep the mean-filtered value. This
  is unusual (it suppresses contrast in homogeneous regions) but is
  exactly what Eq. (6) specifies, so it is kept as-is.
- **Fusion-Net (Fig. 3)**: the figure's two sketches both end in global
  average pooling + a 2-unit fully-connected layer — a *classification*
  head — which cannot itself produce a fused *image* for the downstream
  HFCMIK/RDWT/GLCM stages. This implementation follows Equations
  (10)–(15) literally for the F1/F2 → F3/F4 → F_R feature path, then uses
  the resulting per-branch activation strength as spatially-varying
  pixel weights to blend the HPWF-denoised MR and CT/PET/SPECT images —
  exactly what §2.2's prose describes ("the fused output is ... produced
  by combining the results from the HPWF and REA").
- **HFCMIK tumor-cluster selection (Table 5)**: the clustering itself
  (AKMC init + weighted Fuzzy Kernel C-Means) is unsupervised and fully
  implemented, but the paper does not say how to pick *which* resulting
  cluster is "the tumor". `hfcmik_segment` supports an optional
  `tumor_seed_point=(row, col)` (pick whichever cluster contains a known
  point inside the lesion — the reliable option whenever any prior is
  available, e.g. from a click or another detector) and otherwise falls
  back to a naive smallest-non-background-cluster heuristic.
- **DLPNN output layer (Fig. 6)**: the paper's own text lists a final
  dense layer of size "1 × 21", which is inconsistent with its stated
  binary benign/malignant task; `DLPNN(num_classes=2)` defaults to 2 and
  the value is a constructor argument if you need to match "21" exactly.
- **RDWT (Fig. 5)**: implemented as a genuine *redundant* (a.k.a.
  stationary) wavelet transform via `pywt.swt2`, which — unlike a plain
  DWT — keeps every sub-band at full input resolution across both
  decomposition levels, matching the figure.

None of this changes the *shape* of the pipeline or skips any named
component (HPWF, REA, DLCNN Fusion-Net, HFCMIK/AKMC/FKCM, RDWT, GLCM,
DLPNN) — every equation and table in the paper maps to real, executable
code, listed in the table above.

### On "reproducing" the paper's reported numbers

Table 6–9's 99%+ figures come from training/tuning the Fusion-Net and
DLPNN weights on the authors' full BraTS2020 split, which was not
released. This package gives you the exact architectures and
algorithms needed to reproduce that process on your own copy of the
dataset (see "Training the DLPNN classifier" above); `demo.py`'s numbers
on a synthetic phantom are for illustrating that the pipeline runs
correctly end-to-end, not a claim of matching those tables.

## Repository layout

```
btfsc_net/
├── btfsc_net/
│   ├── hpwf.py            # Hybrid Probabilistic Wiener Filter
│   ├── rea.py              # Robust Edge Analysis
│   ├── fusion.py           # DLCNN Fusion-Net + YCbCr color handling
│   ├── segmentation.py     # HFCMIK (AKMC + Fuzzy Kernel C-Means)
│   ├── features.py         # RDWT + GLCM + color statistical features
│   ├── classifier.py       # DLPNN CNN classifier + training loop
│   ├── metrics.py          # Fusion / segmentation / classification metrics
│   └── pipeline.py         # BTFSCNet: wires all stages together
├── tests/test_pipeline.py  # Smoke tests for every module (pytest)
├── demo.py                 # Runnable end-to-end example
└── requirements.txt
```

## Running the tests

```bash
pip install pytest
pytest tests/
```
