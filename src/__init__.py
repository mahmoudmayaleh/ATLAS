"""
ATLAS: Adaptive Task-aware Federated Learning with Heterogeneous Splitting
"""

__version__ = "0.1.0"
# Phase 1 – clustering
from .phase1_clustering import GradientExtractor, TaskClusterer, CFLClusterer, visualize_clusters  # noqa: F401
from .phase1_fingerprint import FingerprintExtractor, build_cosine_fingerprints                      # noqa: F401