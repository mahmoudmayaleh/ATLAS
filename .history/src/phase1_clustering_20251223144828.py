"""
Phase 1: Task Clustering Components
- GradientExtractor: Extract 64-D fingerprints from gradients
- TaskClusterer: Cluster clients into task groups using k-Means
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, List, Tuple, Optional, Union
import warnings


class GradientExtractor:
    """
    Extract task-relevant features from model gradients.
    
    Supports both CPU and GPU tensors. Automatically handles device placement.
    """
    
    def __init__(self, dim: int = 64, device: str = 'cpu'):
        """
        Initialize GradientExtractor.
        
        Args:
            dim: Dimensionality of fingerprint (default: 64)
            device: 'cpu' or 'cuda' for computation device
        """
        self.dim = dim
        self.device = device
        self.pca = None
        self.is_fitted = False
        
    def _flatten(self, gradients: Union[Dict[str, torch.Tensor], torch.Tensor]) -> np.ndarray:
        """
        Convert gradient tensors to 1D flattened vector.
        
        Args:
            gradients: Dict of {layer_name: gradient_tensor} or single tensor
            
        Returns:
            Flattened numpy array
        """
        if isinstance(gradients, dict):
            # Flatten all gradient tensors and concatenate
            flat_grads = []
            for name, grad in gradients.items():
                if grad is not None:
                    # Move to CPU if on GPU
                    if grad.is_cuda:
                        grad = grad.cpu()
                    flat_grads.append(grad.flatten().numpy())
            
            if not flat_grads:
                raise ValueError("No valid gradients found in dictionary")
            
            return np.concatenate(flat_grads)
        
        elif isinstance(gradients, torch.Tensor):
            # Single tensor case
            if gradients.is_cuda:
                gradients = gradients.cpu()
            return gradients.flatten().numpy()
        
        else:
            raise TypeError(f"Expected dict or torch.Tensor, got {type(gradients)}")
    
    def fit(self, gradient_list: List[Union[Dict[str, torch.Tensor], torch.Tensor]]):
        """
        Fit PCA on a collection of gradients.
        
        Args:
            gradient_list: List of gradient dicts/tensors from multiple clients
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
        
        # Fit PCA
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca.fit(grad_matrix)
        self.is_fitted = True
        
        print(f"PCA fitted: {grad_matrix.shape[0]} samples, {grad_matrix.shape[1]} features")
        print(f"Using {n_components} components (target was {self.dim})")
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
                'device': self.device
            }, f)
        print(f"GradientExtractor saved to {path}")
    
    def load(self, path: str):
        """Load fitted PCA model."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.pca = data['pca']
        self.dim = data['dim']
        self.device = data['device']
        self.is_fitted = True
        print(f"GradientExtractor loaded from {path}")


class TaskClusterer:
    """
    Cluster clients into task groups using k-Means on gradient fingerprints.
    
    Automatically selects optimal number of clusters based on Silhouette score.
    """
    
    def __init__(self, n_clusters_range: Tuple[int, int] = (2, 5)):
        """
        Initialize TaskClusterer.
        
        Args:
            n_clusters_range: (min, max) number of clusters to try
        """
        self.n_clusters_range = n_clusters_range
        self.best_kmeans = None
        self.best_score = -1
        self.best_n_clusters = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.metrics_ = {}
    
    def cluster(self, fingerprints: np.ndarray, verbose: bool = True) -> Dict:
        """
        Cluster fingerprints and return task groups.
        
        Args:
            fingerprints: Fingerprint matrix of shape (n_clients, dim)
            verbose: Print progress information
            
        Returns:
            Dict with clustering results
        """
        if fingerprints.shape[0] < self.n_clusters_range[0]:
            raise ValueError(
                f"Need at least {self.n_clusters_range[0]} samples, got {fingerprints.shape[0]}"
            )
        
        best_n = None
        best_score = -1
        best_kmeans = None
        
        # Try different numbers of clusters
        for n_clusters in range(self.n_clusters_range[0], self.n_clusters_range[1] + 1):
            if n_clusters > fingerprints.shape[0]:
                break
            
            # Run k-Means with multiple initializations
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=10,
                max_iter=300,
                tol=1e-4,
                random_state=42
            )
            labels = kmeans.fit_predict(fingerprints)
            
            # Compute Silhouette score
            score = silhouette_score(fingerprints, labels)
            
            if verbose:
                print(f"k={n_clusters}: Silhouette={score:.4f}, Inertia={kmeans.inertia_:.2f}")
            
            # Track best clustering
            if score > best_score:
                best_score = score
                best_kmeans = kmeans
                best_n = n_clusters
        
        # Store best results
        self.best_kmeans = best_kmeans
        self.best_score = best_score
        self.best_n_clusters = best_n
        self.labels_ = best_kmeans.labels_
        self.cluster_centers_ = best_kmeans.cluster_centers_
        
        # Compute additional metrics
        self.metrics_ = {
            'silhouette_score': best_score,
            'davies_bouldin_index': davies_bouldin_score(fingerprints, self.labels_),
            'calinski_harabasz_score': calinski_harabasz_score(fingerprints, self.labels_),
            'inertia': best_kmeans.inertia_
        }
        
        if verbose:
            print(f"\n✓ Best clustering: k={best_n}")
            print(f"  Silhouette Score: {self.metrics_['silhouette_score']:.4f}")
            print(f"  Davies-Bouldin Index: {self.metrics_['davies_bouldin_index']:.4f}")
            print(f"  Calinski-Harabasz Score: {self.metrics_['calinski_harabasz_score']:.2f}")
        
        return {
            'n_clusters': best_n,
            'labels': self.labels_,
            'centroids': self.cluster_centers_,
            'silhouette_score': best_score,
            'metrics': self.metrics_
        }
    
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
