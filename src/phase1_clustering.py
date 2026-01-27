"""
Phase 1: Task Clustering Components
- GradientExtractor: Extract 64-D fingerprints from gradients with layer normalization
- TaskClusterer: Cluster clients into task groups with temporal stability
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional, Union
import warnings


class GradientExtractor:
    """
    Extract task-relevant features from model gradients.
    
    Improvements for heterogeneous FL literature:
    - Parameter whitelisting (focus on adapted layers only)
    - Layer-wise normalization (avoid domination by large layers)
    - Incremental PCA for temporal updates
    """
    
    def __init__(self, dim: int = 64, device: str = 'cpu', 
                 param_whitelist: Optional[List[str]] = None,
                 layer_normalize: bool = True,
                 use_incremental_pca: bool = False):
        """        
        Args:
            dim: Dimensionality of fingerprint (default: 64)
            device: 'cpu' or 'cuda' for computation device
            param_whitelist: List of parameter name prefixes to include (e.g., ['transformer.h.10', 'transformer.h.11'])
                           If None, uses all parameters
            layer_normalize: If True, normalize each layer's gradient by its norm before concatenation
            use_incremental_pca: If True, uses IncrementalPCA for online fitting
        """
        self.dim = dim
        self.device = device
        self.param_whitelist = param_whitelist
        self.layer_normalize = layer_normalize
        self.use_incremental_pca = use_incremental_pca
        self.pca = None
        self.is_fitted = False
        self.gradient_buffer = []  # For incremental fitting
        
    def _filter_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Filter gradients by parameter whitelist.
        
        Args:
            gradients: Dict of {layer_name: gradient_tensor}
            
        Returns:
            Filtered gradient dict
        """
        if self.param_whitelist is None:
            return gradients
        
        filtered = {}
        for name, grad in gradients.items():
            # Check if name matches any whitelist prefix
            if any(name.startswith(prefix) for prefix in self.param_whitelist):
                filtered[name] = grad
        
        if not filtered:
            warnings.warn(
                f"No parameters matched whitelist {self.param_whitelist}. "
                f"Using all {len(gradients)} parameters."
            )
            return gradients
        
        return filtered
    
    def _flatten(self, gradients: Union[Dict[str, torch.Tensor], torch.Tensor]) -> np.ndarray:
        """
        Convert gradient tensors to 1D flattened vector with optional layer normalization.
        
        Args:
            gradients: Dict of {layer_name: gradient_tensor} or single tensor
            
        Returns:
            Flattened numpy array
        """
        if isinstance(gradients, dict):
            # Filter by whitelist if specified
            gradients = self._filter_gradients(gradients)
            
            # Flatten each layer separately with optional normalization
            flat_grads = []
            for name, grad in gradients.items():
                if grad is not None:
                    # Move to CPU if on GPU
                    if grad.is_cuda:
                        grad = grad.cpu()
                    
                    layer_flat = grad.flatten().numpy()
                    
                    # Layer-wise normalization (avoid domination by large layers)
                    if self.layer_normalize:
                        norm = np.linalg.norm(layer_flat)
                        if norm > 1e-8:
                            layer_flat = layer_flat / norm
                    
                    flat_grads.append(layer_flat)
            
            if not flat_grads:
                raise ValueError("No valid gradients found in dictionary")
            
            return np.concatenate(flat_grads)
        
        elif isinstance(gradients, torch.Tensor):
            # Single tensor case
            if gradients.is_cuda:
                gradients = gradients.cpu()
            flat = gradients.flatten().numpy()
            
            if self.layer_normalize:
                norm = np.linalg.norm(flat)
                if norm > 1e-8:
                    flat = flat / norm
            
            return flat
        
        else:
            raise TypeError(f"Expected dict or torch.Tensor, got {type(gradients)}")
    
    def fit(self, gradient_list: List[Union[Dict[str, torch.Tensor], torch.Tensor]],
            update_buffer: bool = True):
        """
        Fit PCA on a collection of gradients.
        
        Args:
            gradient_list: List of gradient dicts/tensors from multiple clients
            update_buffer: If True, adds gradients to buffer for incremental fitting
        """
        # Flatten all gradients
        flat_grads = []
        for grads in gradient_list:
            try:
                flat = self._flatten(grads)
                flat_grads.append(flat)
            except Exception as e:
                warnings.warn(f"Skipping gradient due to error: {e}")
                continue
        
        if not flat_grads:
            raise ValueError("No valid gradients to fit PCA")
        
        # Update buffer for incremental PCA
        if update_buffer and self.use_incremental_pca:
            self.gradient_buffer.extend(flat_grads)
            # Keep buffer size manageable (last 1000 samples)
            if len(self.gradient_buffer) > 1000:
                self.gradient_buffer = self.gradient_buffer[-1000:]
        
        # Stack into matrix (n_samples, n_features)
        grad_matrix = np.vstack(flat_grads)
        
        # Adjust n_components if necessary
        max_components = min(grad_matrix.shape[0], grad_matrix.shape[1])
        n_components = min(self.dim, max_components)
        
        if n_components < self.dim:
            warnings.warn(
                f"Reducing n_components from {self.dim} to {n_components} "
                f"(max possible with {grad_matrix.shape[0]} samples and {grad_matrix.shape[1]} features)"
            )
        
        # Choose PCA type
        if self.use_incremental_pca:
            from sklearn.decomposition import IncrementalPCA
            
            if self.pca is None or not self.is_fitted:
                # Initial fit
                self.pca = IncrementalPCA(n_components=n_components, batch_size=max(10, grad_matrix.shape[0] // 5))
                self.pca.fit(grad_matrix)
            else:
                # Partial fit (update existing PCA)
                self.pca.partial_fit(grad_matrix)
        else:
            # Standard PCA
            self.pca = PCA(n_components=n_components, random_state=42)
            self.pca.fit(grad_matrix)
        
        self.is_fitted = True
        
        print(f"PCA fitted: {grad_matrix.shape[0]} samples, {grad_matrix.shape[1]} features")
        print(f"Using {n_components} components (target was {self.dim})")
        if hasattr(self.pca, 'explained_variance_ratio_'):
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
    
    def extract(self, gradients: Union[Dict[str, torch.Tensor], torch.Tensor]) -> np.ndarray:
        """
        Extract d-dimensional fingerprint from gradients.
        
        Args:
            gradients: Gradient dict or tensor
            
        Returns:
            Fingerprint vector of shape (dim,)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before extract()")
        
        # Flatten gradients
        flat_grads = self._flatten(gradients)
        
        # Transform using PCA
        fingerprint = self.pca.transform(flat_grads.reshape(1, -1))[0]
        
        # Pad to target dimension if PCA used fewer components
        if len(fingerprint) < self.dim:
            fingerprint = np.pad(fingerprint, (0, self.dim - len(fingerprint)), mode='constant')
        
        # Normalize to unit L2 norm
        norm = np.linalg.norm(fingerprint)
        if norm > 1e-8:
            fingerprint = fingerprint / norm
        
        return fingerprint
    
    def extract_batch(self, gradient_list: List[Union[Dict[str, torch.Tensor], torch.Tensor]]) -> np.ndarray:
        """
        Extract fingerprints for multiple clients.
        
        Args:
            gradient_list: List of gradient dicts/tensors
            
        Returns:
            Fingerprint matrix of shape (n_clients, dim)
        """
        fingerprints = []
        for grads in gradient_list:
            try:
                fp = self.extract(grads)
                fingerprints.append(fp)
            except Exception as e:
                warnings.warn(f"Failed to extract fingerprint: {e}")
                # Add zero vector as fallback
                fingerprints.append(np.zeros(self.dim))
        
        return np.vstack(fingerprints)
    
    def save(self, path: str):
        """Save fitted PCA model."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'pca': self.pca,
                'dim': self.dim,
                'is_fitted': self.is_fitted,
                'param_whitelist': self.param_whitelist,
                'layer_normalize': self.layer_normalize,
                'use_incremental_pca': self.use_incremental_pca
            }, f)
        print(f"GradientExtractor saved to {path}")
    
    def load(self, path: str):
        """Load fitted PCA model."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.pca = data['pca']
        self.dim = data['dim']
        self.is_fitted = data['is_fitted']
        self.param_whitelist = data.get('param_whitelist')
        self.layer_normalize = data.get('layer_normalize', True)
        self.use_incremental_pca = data.get('use_incremental_pca', False)
        print(f"GradientExtractor loaded from {path}")


class TaskClusterer:
    """
    Cluster clients into task groups using k-Means on gradient fingerprints.
    
    Improvements for heterogeneous FL:
    - Multi-metric k selection (Silhouette + Davies-Bouldin + Calinski-Harabasz)
    - Temporal stability (initialize with previous centroids)
    - Minimum cluster size enforcement
    - Cluster-level summaries for Phase 2 integration
    """
    
    def __init__(self, n_clusters_range: Tuple[int, int] = (2, 5),
                 min_cluster_size: int = 2,
                 temporal_consistency_weight: float = 0.0,
                 fixed_k: Optional[int] = None):
        """
        Initialize TaskClusterer.
        
        Args:
            n_clusters_range: (min, max) number of clusters to try (ignored if fixed_k is set)
            min_cluster_size: Minimum number of clients per cluster
            temporal_consistency_weight: Weight for temporal stability (0 = no consistency)
            fixed_k: If set, uses this fixed number of clusters
        """
        self.n_clusters_range = n_clusters_range
        self.min_cluster_size = min_cluster_size
        self.temporal_consistency_weight = temporal_consistency_weight
        self.fixed_k = fixed_k
        self.best_kmeans = None
        self.best_score = -1
        self.best_n_clusters = None
        self.labels_ = None
        self.prev_labels_ = None  # For temporal consistency
        self.cluster_centers_ = None
        self.metrics_ = {}
        self.cluster_summaries_ = {}  # For Phase 2 integrationtractor loaded from {path}")


class TaskClusterer:
    """
    Cluster clients into task groups using k-Means on gradient fingerprints.
    
    Automatically selects optimal number of clusters based on Silhouette score.
    """
    
    def __init__(self, n_clusters_range: tuple = (2, 5), fixed_k: int = None, min_cluster_size: int = 2):
        """
        Initialize TaskClusterer.
        
        Args:
            n_clusters_range: Tuple of (min, max) clusters to try
            fixed_k: If provided, use fixed number of clusters (no auto-selection)
            min_cluster_size: Minimum samples per cluster
        """
        self.n_clusters_range = n_clusters_range
        self.fixed_k = fixed_k
        self.min_cluster_size = min_cluster_size
        self.best_kmeans = None
        self.best_n_clusters = None
        self.best_score = -1.0
        
        # Clustering state
        self.labels_ = None
        self.cluster_centers_ = None
        self.n_clusters_ = None
        self.silhouette_score_ = None
        self.inertia_ = None
        
        # History for temporal stability
        self.history = []
    
    def _compute_combined_score(self, fingerprints: np.ndarray, labels: np.ndarray,
                                kmeans: KMeans) -> float:
        """
        Compute combined clustering quality score using multiple metrics.
        
        Combines:
        - Silhouette (maximize, range [-1, 1])
        - Davies-Bouldin (minimize, range [0, ∞))
        - Calinski-Harabasz (maximize, range [0, ∞))
        
        Returns normalized score in [0, 1]
        """
        sil = silhouette_score(fingerprints, labels)
        db = davies_bouldin_score(fingerprints, labels)
        ch = calinski_harabasz_score(fingerprints, labels)
        
        # Normalize scores to [0, 1]
        sil_norm = (sil + 1) / 2  # [-1, 1] → [0, 1]
        db_norm = 1 / (1 + db)     # Lower is better, asymptotic to 1
        ch_norm = ch / (ch + 1000) # Normalize by typical range
        
        # Weighted combination (can be tuned)
        combined = 0.5 * sil_norm + 0.3 * db_norm + 0.2 * ch_norm
        
        return combined
    
    def _check_min_cluster_size(self, labels: np.ndarray) -> bool:
        """Check if all clusters meet minimum size requirement."""
        unique, counts = np.unique(labels, return_counts=True)
        return all(count >= self.min_cluster_size for count in counts)
    
    def _compute_temporal_consistency(self, new_labels: np.ndarray) -> float:
        """
        Compute consistency with previous clustering.
        
        Returns fraction of clients that kept the same label.
        """
        if self.prev_labels_ is None:
            return 1.0  # First clustering, fully consistent
        
        if len(new_labels) != len(self.prev_labels_):
            warnings.warn("Label length mismatch, cannot compute temporal consistency")
            return 0.5
        
        # Compute assignment similarity (ignoring label permutations)
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics import confusion_matrix
        
        # Build confusion matrix
        cm = confusion_matrix(self.prev_labels_, new_labels)
        
        # Find optimal label matching (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(-cm)
        
        # Count matched assignments
        matched = cm[row_ind, col_ind].sum()
        consistency = matched / len(new_labels)
        
        return consistency
    
    def cluster(self, fingerprints: np.ndarray, verbose: bool = True) -> Dict:
        """
        Cluster fingerprints with multi-metric selection and temporal stability.
        
        Args:
            fingerprints: Fingerprint matrix of shape (n_clients, dim)
            verbose: Print progress information
            
        Returns:
            Dict with clustering results
        """
        # Determine k range
        if self.fixed_k is not None:
            k_range = [self.fixed_k]
        else:
            min_k = max(self.n_clusters_range[0], 2)
            max_k = min(self.n_clusters_range[1], fingerprints.shape[0])
            k_range = range(min_k, max_k + 1)
        
        if fingerprints.shape[0] < min(k_range):
            raise ValueError(
                f"Need at least {min(k_range)} samples, got {fingerprints.shape[0]}"
            )
        
        best_n = None
        best_score = -1
        best_kmeans = None
        best_temporal_score = 0
        
        # Try different numbers of clusters
        for n_clusters in k_range:
            if n_clusters > fingerprints.shape[0]:
                break
            
            # Initialize with previous centroids if available (temporal stability)
            if self.cluster_centers_ is not None and self.cluster_centers_.shape[0] == n_clusters:
                init_centroids = self.cluster_centers_
                n_init = 1  # Already have good initialization
            else:
                init_centroids = 'k-means++'
                n_init = 10
            
            # Run k-Means
            kmeans = KMeans(
                n_clusters=n_clusters,
                init=init_centroids,
                n_init=n_init,
                max_iter=300,
                tol=1e-4,
                random_state=42
            )
            labels = kmeans.fit_predict(fingerprints)
            
            # Check minimum cluster size
            if not self._check_min_cluster_size(labels):
                if verbose:
                    print(f"k={n_clusters}: Rejected (cluster too small, min={self.min_cluster_size})")
                continue
            
            # Compute combined quality score
            quality_score = self._compute_combined_score(fingerprints, labels, kmeans)
            
            # Compute temporal consistency
            temporal_consistency = self._compute_temporal_consistency(labels)
            
            # Combined score with temporal consistency
            total_score = (1 - self.temporal_consistency_weight) * quality_score + \
                         self.temporal_consistency_weight * temporal_consistency
            
            if verbose:
                sil = silhouette_score(fingerprints, labels)
                db = davies_bouldin_score(fingerprints, labels)
                print(f"k={n_clusters}: Combined={total_score:.4f} "
                      f"(Sil={sil:.3f}, DB={db:.3f}, Temporal={temporal_consistency:.3f})")
            
            # Track best clustering
            if total_score > best_score:
                best_score = total_score
                best_kmeans = kmeans
                best_n = n_clusters
                best_temporal_score = temporal_consistency
        
        if best_kmeans is None:
            raise ValueError("No valid clustering found. Try reducing min_cluster_size.")
        
        # Store previous labels for next round
        self.prev_labels_ = self.labels_.copy() if self.labels_ is not None else None
        
        # Store best results
        self.best_kmeans = best_kmeans
        self.best_score = best_score
        self.best_n_clusters = best_n
        self.labels_ = best_kmeans.labels_
        self.cluster_centers_ = best_kmeans.cluster_centers_
        
        # Compute additional metrics
        self.metrics_ = {
            'combined_score': best_score,
            'silhouette_score': silhouette_score(fingerprints, self.labels_),
            'davies_bouldin_index': davies_bouldin_score(fingerprints, self.labels_),
            'calinski_harabasz_score': calinski_harabasz_score(fingerprints, self.labels_),
            'temporal_consistency': best_temporal_score,
            'inertia': best_kmeans.inertia_
        }
        
        # Compute cluster-level summaries for Phase 2
        self._compute_cluster_summaries(fingerprints)
        
        if verbose:
            print(f"\n✓ Best clustering: k={best_n}")
            print(f"  Combined Score: {self.metrics_['combined_score']:.4f}")
            print(f"  Silhouette: {self.metrics_['silhouette_score']:.4f}")
            print(f"  Davies-Bouldin: {self.metrics_['davies_bouldin_index']:.4f}")
            print(f"  Calinski-Harabasz: {self.metrics_['calinski_harabasz_score']:.2f}")
            print(f"  Temporal Consistency: {self.metrics_['temporal_consistency']:.4f}")
        
        return {
            'n_clusters': best_n,
            'labels': self.labels_,
            'centroids': self.cluster_centers_,
            'metrics': self.metrics_,
            'cluster_summaries': self.cluster_summaries_
        }
    
    def _compute_cluster_summaries(self, fingerprints: np.ndarray):
        """
        Compute cluster-level summaries for downstream phases.
        
        Computes:
        - Mean fingerprint per cluster
        - Within-cluster variance
        - Cluster size
        """
        self.cluster_summaries_ = {}
        
        for cluster_id in range(self.best_n_clusters):
            mask = self.labels_ == cluster_id
            cluster_fps = fingerprints[mask]
            
            self.cluster_summaries_[cluster_id] = {
                'mean_fingerprint': cluster_fps.mean(axis=0),
                'variance': cluster_fps.var(axis=0).mean(),  # Average variance across dimensions
                'std': cluster_fps.std(axis=0).mean(),
                'size': int(mask.sum()),
                'centroid': self.cluster_centers_[cluster_id]
            }
    
    def get_cluster_summary(self, cluster_id: int) -> Dict:
        """
        Get summary statistics for a specific cluster.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            Dict with mean fingerprint, variance, size, etc.
        """
        if cluster_id not in self.cluster_summaries_:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        return self.cluster_summaries_[cluster_id]
    
    def get_client_cluster_mapping(self, client_ids: List[int]) -> Dict[int, int]:
        """
        Map client IDs to their cluster assignments.
        
        Args:
            client_ids: List of client IDs
            
        Returns:
            Dict mapping {client_id: cluster_id}
        """
        if self.labels_ is None:
            raise RuntimeError("Must call cluster() first")
        
        if len(client_ids) != len(self.labels_):
            raise ValueError(
                f"Number of client_ids ({len(client_ids)}) != "
                f"number of labels ({len(self.labels_)})"
            )
        
        return {client_id: int(label) for client_id, label in zip(client_ids, self.labels_)}
    
    def get_task_groups(self, client_ids: List[int]) -> Dict[int, List[int]]:
        """
        Return task group assignments.
        
        Args:
            client_ids: List of client IDs
            
        Returns:
            Dict mapping {task_group_id: [client_ids]}
        """
        if self.labels_ is None:
            raise RuntimeError("Must call cluster() first")
        
        if len(client_ids) != len(self.labels_):
            raise ValueError(
                f"Number of client_ids ({len(client_ids)}) != "
                f"number of labels ({len(self.labels_)})"
            )
        
        groups = {}
        for client_id, label in zip(client_ids, self.labels_):
            if label not in groups:
                groups[label] = []
            groups[label].append(client_id)
        
        return groups
    
    def predict(self, fingerprints: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new fingerprints.
        
        Args:
            fingerprints: Fingerprint matrix
            
        Returns:
            Cluster labels
        """
        if self.best_kmeans is None:
            raise RuntimeError("Must call cluster() first")
        
        return self.best_kmeans.predict(fingerprints)
    
    def save(self, path: str):
        """Save clustering model."""
        if self.best_kmeans is None:
            raise RuntimeError("Cannot save unfitted model")
        
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'kmeans': self.best_kmeans,
                'n_clusters_range': self.n_clusters_range,
                'best_n_clusters': self.best_n_clusters,
                'best_score': self.best_score,
                'metrics': self.metrics_
            }, f)
        print(f"TaskClusterer saved to {path}")
    
    def load(self, path: str):
        """Load clustering model."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.best_kmeans = data['kmeans']
        self.n_clusters_range = data['n_clusters_range']
        self.best_n_clusters = data['best_n_clusters']
        self.best_score = data['best_score']
        self.metrics_ = data['metrics']
        self.labels_ = self.best_kmeans.labels_
        self.cluster_centers_ = self.best_kmeans.cluster_centers_
        print(f"TaskClusterer loaded from {path}")


def visualize_clusters(fingerprints: np.ndarray, labels: np.ndarray, 
                       method: str = 'tsne', save_path: Optional[str] = None):
    """
    Visualize clusters in 2D using t-SNE or PCA.
    
    Args:
        fingerprints: Fingerprint matrix (n_clients, dim)
        labels: Cluster labels
        method: 'tsne' or 'pca'
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")
        return
    
    # Reduce to 2D
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(fingerprints)
        title = "Task Clustering (t-SNE)"
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(fingerprints)
        title = "Task Clustering (PCA)"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Plot
    plt.figure(figsize=(10, 8))
    n_clusters = len(np.unique(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        plt.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[colors[cluster_id]],
            label=f'Task Group {cluster_id}',
            s=100,
            alpha=0.7,
            edgecolors='black'
        )
    
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Quick test
    print("Phase 1 Clustering Module Loaded Successfully!")
    print("=" * 60)
    print("Available classes:")
    print("  - GradientExtractor")
    print("  - TaskClusterer")
    print("Available functions:")
    print("  - visualize_clusters()")
    print("=" * 60)
