"""
Unit tests for Phase 1: Task Clustering
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from phase1_clustering import GradientExtractor, TaskClusterer


class TestGradientExtractor(unittest.TestCase):
    """Test cases for GradientExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = GradientExtractor(dim=64, device='cpu')
        
        # Create dummy gradients
        self.dummy_grads_dict = {
            'layer1': torch.randn(10, 20),
            'layer2': torch.randn(30, 40),
            'layer3': torch.randn(50, 60)
        }
        
        self.dummy_grads_tensor = torch.randn(100, 200)
    
    def test_initialization(self):
        """Test GradientExtractor initialization."""
        self.assertEqual(self.extractor.dim, 64)
        self.assertEqual(self.extractor.device, 'cpu')
        self.assertIsNone(self.extractor.pca)
        self.assertFalse(self.extractor.is_fitted)
    
    def test_flatten_dict(self):
        """Test flattening gradient dictionary."""
        flat = self.extractor._flatten(self.dummy_grads_dict)
        
        # Check type and shape
        self.assertIsInstance(flat, np.ndarray)
        expected_size = 10*20 + 30*40 + 50*60
        self.assertEqual(flat.shape, (expected_size,))
    
    def test_flatten_tensor(self):
        """Test flattening single tensor."""
        flat = self.extractor._flatten(self.dummy_grads_tensor)
        
        # Check type and shape
        self.assertIsInstance(flat, np.ndarray)
        self.assertEqual(flat.shape, (100*200,))
    
    def test_fit(self):
        """Test PCA fitting."""
        # Create multiple gradient samples (need enough for 64 components)
        gradient_list = [
            {f'layer{i}': torch.randn(10, 20) for i in range(3)}
            for _ in range(70)  # More samples to support 64 PCA components
        ]
        
        # Fit extractor
        self.extractor.fit(gradient_list)
        
        # Check fitted state
        self.assertTrue(self.extractor.is_fitted)
        self.assertIsNotNone(self.extractor.pca)
        self.assertEqual(self.extractor.pca.n_components, 64)
    
    def test_extract_single(self):
        """Test extracting single fingerprint."""
        # First fit the extractor
        gradient_list = [
            {f'layer{i}': torch.randn(10, 20) for i in range(3)}
            for _ in range(70)  # More samples to support 64 PCA components
        ]
        self.extractor.fit(gradient_list)
        
        # Extract fingerprint
        fingerprint = self.extractor.extract(self.dummy_grads_dict)
        
        # Check shape and normalization
        self.assertEqual(fingerprint.shape, (64,))
        norm = np.linalg.norm(fingerprint)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_extract_batch(self):
        """Test extracting multiple fingerprints."""
        # First fit the extractor
        gradient_list = [
            {f'layer{i}': torch.randn(10, 20) for i in range(3)}
            for _ in range(70)  # More samples to support 64 PCA components
        ]
        self.extractor.fit(gradient_list)
        
        # Extract batch
        fingerprints = self.extractor.extract_batch(gradient_list[:10])
        
        # Check shape
        self.assertEqual(fingerprints.shape, (10, 64))
        
        # Check normalization for each fingerprint
        for i in range(10):
            norm = np.linalg.norm(fingerprints[i])
            self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_extract_before_fit_raises_error(self):
        """Test that extracting before fitting raises error."""
        with self.assertRaises(RuntimeError):
            self.extractor.extract(self.dummy_grads_dict)
    
    def test_reproducibility(self):
        """Test that extraction is reproducible."""
        # Fit extractor
        gradient_list = [
            {f'layer{i}': torch.randn(10, 20) for i in range(3)}
            for _ in range(70)  # More samples to support 64 PCA components
        ]
        self.extractor.fit(gradient_list)
        
        # Extract twice
        fp1 = self.extractor.extract(self.dummy_grads_dict)
        fp2 = self.extractor.extract(self.dummy_grads_dict)
        
        # Should be identical
        np.testing.assert_array_almost_equal(fp1, fp2)


class TestTaskClusterer(unittest.TestCase):
    """Test cases for TaskClusterer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.clusterer = TaskClusterer(n_clusters_range=(2, 5))
    
    def test_initialization(self):
        """Test TaskClusterer initialization."""
        self.assertEqual(self.clusterer.n_clusters_range, (2, 5))
        self.assertIsNone(self.clusterer.best_kmeans)
        self.assertEqual(self.clusterer.best_score, -1)
    
    def test_cluster_synthetic_data(self):
        """Test clustering on synthetic data with known clusters."""
        # Create synthetic data with 3 clear clusters
        np.random.seed(42)
        cluster1 = np.random.randn(20, 64) + np.array([5, 0] + [0]*62)
        cluster2 = np.random.randn(20, 64) + np.array([0, 5] + [0]*62)
        cluster3 = np.random.randn(20, 64) + np.array([-5, -5] + [0]*62)
        
        fingerprints = np.vstack([cluster1, cluster2, cluster3])
        
        # Cluster
        result = self.clusterer.cluster(fingerprints, verbose=False)
        
        # Check results
        self.assertIn('n_clusters', result)
        self.assertIn('labels', result)
        self.assertIn('centroids', result)
        self.assertIn('silhouette_score', result)
        
        # Should find 2 or 3 clusters (silhouette optimization may prefer 2)
        self.assertIn(result['n_clusters'], [2, 3])
        
        # Silhouette score should be reasonable
        self.assertGreater(result['silhouette_score'], 0.3)
        
        # Check labels shape
        self.assertEqual(len(result['labels']), 60)
    
    def test_get_task_groups(self):
        """Test getting task group assignments."""
        # Create synthetic data
        np.random.seed(42)
        fingerprints = np.vstack([
            np.random.randn(10, 64) + 5,
            np.random.randn(10, 64) - 5
        ])
        
        # Cluster
        self.clusterer.cluster(fingerprints, verbose=False)
        
        # Get task groups
        client_ids = list(range(20))
        groups = self.clusterer.get_task_groups(client_ids)
        
        # Should have 2 groups
        self.assertEqual(len(groups), 2)
        
        # Each group should have 10 clients
        for group_id, clients in groups.items():
            self.assertEqual(len(clients), 10)
        
        # All clients should be assigned
        all_clients = []
        for clients in groups.values():
            all_clients.extend(clients)
        self.assertEqual(sorted(all_clients), client_ids)
    
    def test_predict_new_samples(self):
        """Test predicting clusters for new samples."""
        # Create and fit on training data
        np.random.seed(42)
        train_fingerprints = np.vstack([
            np.random.randn(10, 64) + 5,
            np.random.randn(10, 64) - 5
        ])
        
        self.clusterer.cluster(train_fingerprints, verbose=False)
        
        # Predict on new samples
        new_fingerprints = np.vstack([
            np.random.randn(5, 64) + 5,
            np.random.randn(5, 64) - 5
        ])
        
        predictions = self.clusterer.predict(new_fingerprints)
        
        # Check shape
        self.assertEqual(len(predictions), 10)
        
        # Check all predictions are valid cluster IDs
        unique_clusters = np.unique(predictions)
        self.assertTrue(all(c in [0, 1] for c in unique_clusters))
    
    def test_metrics_computation(self):
        """Test that clustering metrics are computed."""
        # Create synthetic data
        np.random.seed(42)
        fingerprints = np.random.randn(30, 64)
        
        # Cluster
        result = self.clusterer.cluster(fingerprints, verbose=False)
        
        # Check metrics
        self.assertIn('metrics', result)
        metrics = result['metrics']
        
        self.assertIn('silhouette_score', metrics)
        self.assertIn('davies_bouldin_index', metrics)
        self.assertIn('calinski_harabasz_score', metrics)
        self.assertIn('inertia', metrics)
        
        # Silhouette should be between -1 and 1
        self.assertGreaterEqual(metrics['silhouette_score'], -1)
        self.assertLessEqual(metrics['silhouette_score'], 1)
    
    def test_cluster_before_get_task_groups_raises_error(self):
        """Test that getting task groups before clustering raises error."""
        with self.assertRaises(RuntimeError):
            self.clusterer.get_task_groups([1, 2, 3])
    
    def test_too_few_samples_raises_error(self):
        """Test that clustering with too few samples raises error."""
        # Only 1 sample but need at least 2 clusters
        fingerprints = np.random.randn(1, 64)
        
        with self.assertRaises(ValueError):
            self.clusterer.cluster(fingerprints, verbose=False)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow: fit -> extract -> cluster."""
        # Step 1: Create dummy gradients for multiple clients
        np.random.seed(42)
        torch.manual_seed(42)
        
        n_clients = 30
        gradient_list = []
        for i in range(n_clients):
            # Create gradients with some task structure
            # Clients 0-9: Task A (larger gradients in layer1)
            # Clients 10-19: Task B (larger gradients in layer2)
            # Clients 20-29: Task C (larger gradients in layer3)
            if i < 10:
                grads = {
                    'layer1': torch.randn(10, 20) * 5,
                    'layer2': torch.randn(10, 20),
                    'layer3': torch.randn(10, 20)
                }
            elif i < 20:
                grads = {
                    'layer1': torch.randn(10, 20),
                    'layer2': torch.randn(10, 20) * 5,
                    'layer3': torch.randn(10, 20)
                }
            else:
                grads = {
                    'layer1': torch.randn(10, 20),
                    'layer2': torch.randn(10, 20),
                    'layer3': torch.randn(10, 20) * 5
                }
            gradient_list.append(grads)
        
        # Step 2: Fit GradientExtractor
        extractor = GradientExtractor(dim=64, device='cpu')
        extractor.fit(gradient_list)
        
        # Step 3: Extract fingerprints
        fingerprints = extractor.extract_batch(gradient_list)
        
        # Check fingerprints shape
        self.assertEqual(fingerprints.shape, (n_clients, 64))
        
        # Step 4: Cluster clients
        clusterer = TaskClusterer(n_clusters_range=(2, 5))
        result = clusterer.cluster(fingerprints, verbose=False)
        
        # Should ideally find 3 clusters
        self.assertIn(result['n_clusters'], [2, 3, 4])
        
        # Step 5: Get task groups
        client_ids = list(range(n_clients))
        groups = clusterer.get_task_groups(client_ids)
        
        # Check all clients are assigned
        all_assigned = sum(len(clients) for clients in groups.values())
        self.assertEqual(all_assigned, n_clients)
        
        print(f"\n✓ End-to-end test passed!")
        print(f"  Found {result['n_clusters']} task groups")
        print(f"  Silhouette score: {result['silhouette_score']:.4f}")


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGradientExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestTaskClusterer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
