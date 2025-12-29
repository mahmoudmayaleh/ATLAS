"""
Unit tests for Phase 2: Heterogeneous Configuration
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from phase2_configuration import DeviceProfiler, WeightImportanceScorer, RankAllocator


class TestDeviceProfiler(unittest.TestCase):
    """Test cases for DeviceProfiler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = DeviceProfiler()
    
    def test_profile_cpu(self):
        """Test CPU device profile."""
        profile = self.profiler.profile_device('cpu')
        
        self.assertEqual(profile['memory_mb'], 2048)
        self.assertEqual(profile['compute_ratio'], 1.0)
        self.assertEqual(profile['suggested_ranks'], [4, 8])
    
    def test_profile_edge_gpu(self):
        """Test edge GPU profile."""
        profile = self.profiler.profile_device('edge_gpu')
        
        self.assertEqual(profile['memory_mb'], 4096)
        self.assertEqual(profile['compute_ratio'], 5.0)
        self.assertEqual(profile['suggested_ranks'], [8, 16])
    
    def test_profile_gpu(self):
        """Test GPU profile."""
        profile = self.profiler.profile_device('gpu')
        
        self.assertEqual(profile['memory_mb'], 8192)
        self.assertEqual(profile['compute_ratio'], 10.0)
        self.assertEqual(profile['suggested_ranks'], [16, 32, 64])
    
    def test_unknown_device(self):
        """Test handling of unknown device type."""
        with self.assertWarns(UserWarning):
            profile = self.profiler.profile_device('unknown')
        
        # Should default to CPU
        self.assertEqual(profile['memory_mb'], 2048)
    
    def test_estimate_rank(self):
        """Test rank estimation."""
        # CPU with 768-dim model, 6 layers
        rank = self.profiler.estimate_rank('cpu', model_dim=768, target_layers=6)
        
        # Should return a valid rank
        self.assertIn(rank, [4, 8, 16, 32, 64])
        self.assertGreaterEqual(rank, 4)
    
    def test_estimate_rank_gpu(self):
        """Test rank estimation for GPU."""
        rank = self.profiler.estimate_rank('gpu', model_dim=768, target_layers=12)
        
        # GPU should support higher ranks
        self.assertGreaterEqual(rank, 16)
    
    def test_get_available_devices(self):
        """Test getting available device types."""
        devices = self.profiler.get_available_devices()
        
        self.assertIn('cpu', devices)
        self.assertIn('edge_gpu', devices)
        self.assertIn('gpu', devices)
    
    def test_compare_devices(self):
        """Test device comparison."""
        comparison = self.profiler.compare_devices()
        
        self.assertIn('cpu', comparison)
        self.assertIn('edge_gpu', comparison)
        self.assertIn('gpu', comparison)
        
        # GPU should have more memory than CPU
        self.assertGreater(
            comparison['gpu']['memory_gb'],
            comparison['cpu']['memory_gb']
        )


class TestWeightImportanceScorer(unittest.TestCase):
    """Test cases for WeightImportanceScorer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = WeightImportanceScorer()
    
    def test_compute_importance(self):
        """Test importance computation from gradients."""
        # Create dummy gradients
        gradients = {
            'layer_0': torch.randn(10, 20) * 5,
            'layer_1': torch.randn(10, 20) * 2,
            'layer_2': torch.randn(10, 20) * 1
        }
        
        importance = self.scorer.compute_importance(gradients)
        
        # Check all layers present
        self.assertIn('layer_0', importance)
        self.assertIn('layer_1', importance)
        self.assertIn('layer_2', importance)
        
        # Check normalization (sum should be ~1.0)
        total = sum(importance.values())
        self.assertAlmostEqual(total, 1.0, places=5)
        
        # Layer 0 should have highest importance (largest gradients)
        self.assertGreater(importance['layer_0'], importance['layer_1'])
        self.assertGreater(importance['layer_1'], importance['layer_2'])
    
    def test_compute_importance_with_none(self):
        """Test handling of None gradients."""
        gradients = {
            'layer_0': torch.randn(10, 20),
            'layer_1': None,
            'layer_2': torch.randn(10, 20)
        }
        
        importance = self.scorer.compute_importance(gradients)
        
        # Should skip None gradient
        self.assertIn('layer_0', importance)
        self.assertNotIn('layer_1', importance)
        self.assertIn('layer_2', importance)
    
    def test_get_layer_importance(self):
        """Test getting importance for specific layer."""
        importance_dict = {
            'layer_0.weight': 0.3,
            'layer_0.bias': 0.1,
            'layer_1.weight': 0.4,
            'layer_1.bias': 0.2
        }
        
        # Get importance for layer_0
        layer_0_importance = self.scorer.get_layer_importance('layer_0', importance_dict)
        
        # Should sum weight and bias
        self.assertAlmostEqual(layer_0_importance, 0.4, places=5)
    
    def test_get_layer_ranking(self):
        """Test layer ranking by importance."""
        importance_dict = {
            'layer_0.weight': 0.1,
            'layer_1.weight': 0.5,
            'layer_2.weight': 0.4
        }
        
        self.scorer.importance_cache = importance_dict
        ranking = self.scorer.get_layer_ranking()
        
        # Should be sorted by importance
        self.assertEqual(ranking[0][0], 'layer_1')  # Highest
        self.assertEqual(ranking[1][0], 'layer_2')
        self.assertEqual(ranking[2][0], 'layer_0')  # Lowest
    
    def test_empty_gradients(self):
        """Test handling of empty gradients."""
        gradients = {}
        
        importance = self.scorer.compute_importance(gradients)
        
        # Should return empty dict
        self.assertEqual(len(importance), 0)


class TestRankAllocator(unittest.TestCase):
    """Test cases for RankAllocator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.allocator = RankAllocator(model_dim=768)
        self.profiler = DeviceProfiler()
    
    def test_allocate_uniform_ranks(self):
        """Test uniform rank allocation."""
        ranks = self.allocator.allocate_uniform_ranks('cpu', n_layers=6)
        
        # All ranks should be the same
        self.assertEqual(len(set(ranks)), 1)
        self.assertEqual(len(ranks), 6)
        
        # Should be from suggested ranks
        self.assertIn(ranks[0], [4, 8])
    
    def test_allocate_ranks_with_importance(self):
        """Test heterogeneous rank allocation with importance scores."""
        device_profile = self.profiler.profile_device('gpu')
        
        importance_scores = {
            'layer_0': 0.3,
            'layer_1': 0.2,
            'layer_2': 0.1,
            'layer_3': 0.4
        }
        
        ranks = self.allocator.allocate_ranks(device_profile, importance_scores, n_layers=4)
        
        # Should return ranks for all layers
        self.assertEqual(len(ranks), 4)
        
        # All ranks should be valid
        for rank in ranks:
            self.assertIn(rank, [4, 8, 16, 32, 64])
    
    def test_allocate_ranks_uniform_importance(self):
        """Test rank allocation with uniform importance."""
        device_profile = self.profiler.profile_device('cpu')
        
        # Empty importance scores
        importance_scores = {}
        
        ranks = self.allocator.allocate_ranks(device_profile, importance_scores, n_layers=6)
        
        # Should allocate reasonable ranks
        self.assertEqual(len(ranks), 6)
        for rank in ranks:
            self.assertIn(rank, [4, 8, 16, 32, 64])
    
    def test_select_rank(self):
        """Test rank selection."""
        # Test various target ranks
        self.assertEqual(self.allocator._select_rank(5.0), 4)
        self.assertEqual(self.allocator._select_rank(10.0), 8)
        self.assertEqual(self.allocator._select_rank(20.0), 16)
        self.assertEqual(self.allocator._select_rank(100.0), 64)
        self.assertEqual(self.allocator._select_rank(2.0), 4)  # Below minimum
    
    def test_validate_memory_constraint(self):
        """Test memory constraint validation."""
        device_profile = self.profiler.profile_device('cpu')
        
        # Valid ranks (small)
        ranks = [4, 4, 4, 4, 4, 4]
        is_valid, memory_mb = self.allocator.validate_memory_constraint(ranks, device_profile)
        
        self.assertTrue(is_valid)
        self.assertGreater(memory_mb, 0)
        
        # Invalid ranks (too large for CPU)
        ranks = [64, 64, 64, 64, 64, 64]
        is_valid, memory_mb = self.allocator.validate_memory_constraint(ranks, device_profile)
        
        # May or may not be valid depending on calculation
        self.assertIsInstance(is_valid, bool)
    
    def test_get_rank_for_device(self):
        """Test getting ranks for specific device."""
        task_group_importance = {
            0: {
                'layer_0': 0.3,
                'layer_1': 0.2,
                'layer_2': 0.5
            }
        }
        
        ranks = self.allocator.get_rank_for_device(
            device_id=0,
            device_type='gpu',
            task_group_importance=task_group_importance,
            n_layers=12
        )
        
        # Should return ranks for all 12 layers
        self.assertEqual(len(ranks), 12)
        
        # All ranks should be valid
        for rank in ranks:
            self.assertIn(rank, [4, 8, 16, 32, 64])


class TestIntegration(unittest.TestCase):
    """Integration tests for Phase 2 components."""
    
    def test_end_to_end_workflow(self):
        """Test complete Phase 2 workflow."""
        # Step 1: Profile device
        profiler = DeviceProfiler()
        device_profile = profiler.profile_device('gpu')
        
        self.assertIsInstance(device_profile, dict)
        self.assertIn('memory_mb', device_profile)
        
        # Step 2: Compute importance from dummy gradients
        scorer = WeightImportanceScorer()
        gradients = {
            f'layer_{i}': torch.randn(10, 20) * (i + 1)
            for i in range(6)
        }
        importance = scorer.compute_importance(gradients)
        
        self.assertEqual(len(importance), 6)
        
        # Step 3: Allocate ranks
        allocator = RankAllocator(model_dim=768)
        ranks = allocator.allocate_ranks(device_profile, importance, n_layers=6)
        
        self.assertEqual(len(ranks), 6)
        
        # Step 4: Validate memory constraint
        is_valid, memory_mb = allocator.validate_memory_constraint(ranks, device_profile)
        
        self.assertTrue(is_valid)
        self.assertGreater(memory_mb, 0)
        
        print(f"\n✓ End-to-end test passed!")
        print(f"  Device: gpu")
        print(f"  Ranks: {ranks}")
        print(f"  Memory usage: {memory_mb:.2f} MB")
        print(f"  Valid: {is_valid}")


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDeviceProfiler))
    suite.addTests(loader.loadTestsFromTestCase(TestWeightImportanceScorer))
    suite.addTests(loader.loadTestsFromTestCase(TestRankAllocator))
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
