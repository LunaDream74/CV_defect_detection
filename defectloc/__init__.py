"""Reconstruction-based defect localisation for industrial inspection.

A convolutional autoencoder trained only on normal bottles produces a spatial
anomaly map, so a defect's location can be traced back to the upstream
manufacturing station that caused it.

This package is the module refactor of `patchcore-base.ipynb` (the v3 Kaggle
prototype). The numerics are a faithful port: same architecture, same loss,
same anomaly map, same image score, so the notebook's results reproduce.
"""

__version__ = "3.1.0"

from defectloc.config import Config  # noqa: F401
