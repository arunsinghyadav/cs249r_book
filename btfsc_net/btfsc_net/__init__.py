"""BTFSC-Net: Brain Tumor Fusion-based Segmentation and Classification.

A from-scratch Python re-implementation of every named stage of:

    Yadav, A.S. et al. "A Feature Extraction Using Probabilistic Neural
    Network and BTFSC-Net Model with Deep Learning for Brain Tumor
    Classification." J. Imaging 2023, 9, 10.
    https://doi.org/10.3390/jimaging9010010

See ``README.md`` for the mapping between this package's modules and the
paper's tables/equations, and ``demo.py`` for a runnable end-to-end example.
"""

from .pipeline import BTFSCNet, PipelineResult

__all__ = ["BTFSCNet", "PipelineResult"]
