"""
Phase 4: Privacy-Aware Aggregation - Unit Tests

Tests for:
- AggregationEngine
- PrivacyVerifier
- Task-aware weighting
- SVD-based merging
- Privacy metrics

Author: ATLAS Team
Date: December 2025
"""

import unittest
import torch
import numpy as np
from src.phase4_aggregation import (
    AggregationEngine,
    PrivacyVerifier,
    compute_task_aware_weights,
    visualize_aggregation_quality
)


class TestAggregationEngine(unittest.TestCase):
    """Test cases for AggregationEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.hidden_dim = 768
        self.target_rank = 16
        self.engine = AggregationEngine(target_rank=self.target_rank)
        
    def test_initialization(self):
        """Test engine initialization"""
        self.assertEqual(self.engine.target_rank, self.target_rank)
        self.assertEqual(self.engine.aggregation_method, 'svd')
        self.assertEqual(len(self.engine.aggregation_history), 0)
        
    def test_normalize_ranks_padding(self):
        """Test rank normalization with padding"""
        weights = [
            torch.randn(self.hidden_dim, 8),
            torch.randn(self.hidden_dim, 16),
            torch.randn(self.hidden_dim, 4)
        ]
        
        normalized = self.engine._normalize_ranks(weights, target_rank=16)
        
        self.assertEqual(len(normalized), 3)
        for w in normalized:
            self.assertEqual(w.shape, (self.hidden_dim, 16))
            
    def test_normalize_ranks_truncation(self):
        """Test rank normalization with truncation"""
        weights = [
            torch.randn(self.hidden_dim, 32),
            torch.randn(self.hidden_dim, 64)
        ]
        
        normalized = self.engine._normalize_ranks(weights, target_rank=16)
        
        for w in normalized:
            self.assertEqual(w.shape, (self.hidden_dim, 16))
            
    def test_average_aggregate(self):
        """Test simple averaging aggregation"""
        A_list = [torch.randn(self.hidden_dim, 8) for _ in range(3)]
        B_list = [torch.randn(self.hidden_dim, 8) for _ in range(3)]
        
        A_merged, B_merged = self.engine._average_aggregate(A_list, B_list)
        
        self.assertEqual(A_merged.shape, (self.hidden_dim, self.target_rank))
        self.assertEqual(B_merged.shape, (self.hidden_dim, self.target_rank))
        
    def test_svd_aggregate(self):
        """Test SVD-based aggregation"""
        A_list = [torch.randn(self.hidden_dim, 8) for _ in range(2)]
        B_list = [torch.randn(self.hidden_dim, 8) for _ in range(2)]
        
        A_merged, B_merged = self.engine._svd_aggregate(A_list, B_list)
        
        self.assertEqual(A_merged.shape[0], self.hidden_dim)
        self.assertEqual(B_merged.shape[0], self.hidden_dim)
        self.assertLessEqual(A_merged.shape[1], self.target_rank)
        
    def test_aggregate_task_group(self):
        """Test single task group aggregation"""
        client_updates = {
            0: {'layer_0': {'A': torch.randn(self.hidden_dim, 8), 
                           'B': torch.randn(self.hidden_dim, 8)}},
            1: {'layer_0': {'A': torch.randn(self.hidden_dim, 8), 
                           'B': torch.randn(self.hidden_dim, 8)}}
        }
        group_clients = [0, 1]
        
        aggregated = self.engine.aggregate_task_group(
            client_updates, group_clients, group_id=0
        )
        
        self.assertIn('layer_0', aggregated)
        self.assertIn('A', aggregated['layer_0'])
        self.assertIn('B', aggregated['layer_0'])
        
    def test_aggregate_all_groups(self):
        """Test aggregation across multiple task groups"""
        client_updates = {
            i: {'layer_0': {'A': torch.randn(self.hidden_dim, 8),
                           'B': torch.randn(self.hidden_dim, 8)}}
            for i in range(4)
        }
        task_groups = {0: [0, 1], 1: [2, 3]}
        
        aggregated = self.engine.aggregate_all_groups(client_updates, task_groups)
        
        self.assertEqual(len(aggregated), 2)
        self.assertIn(0, aggregated)
        self.assertIn(1, aggregated)
        
    def test_weighted_merge(self):
        """Test weighted merging of group aggregates"""
        aggregated_groups = {
            0: {'layer_0': {'A': torch.randn(self.hidden_dim, 16),
                           'B': torch.randn(self.hidden_dim, 16)}},
            1: {'layer_0': {'A': torch.randn(self.hidden_dim, 16),
                           'B': torch.randn(self.hidden_dim, 16)}}
        }
        
        global_weights = self.engine.weighted_merge(aggregated_groups)
        
        self.assertIn('layer_0', global_weights)
        self.assertEqual(global_weights['layer_0']['A'].shape, (self.hidden_dim, 16))
        
    def test_weighted_merge_custom_weights(self):
        """Test weighted merge with custom weights"""
        aggregated_groups = {
            0: {'layer_0': {'A': torch.ones(self.hidden_dim, 16),
                           'B': torch.ones(self.hidden_dim, 16)}},
            1: {'layer_0': {'A': torch.zeros(self.hidden_dim, 16),
                           'B': torch.zeros(self.hidden_dim, 16)}}
        }
        custom_weights = {0: 0.7, 1: 0.3}
        
        global_weights = self.engine.weighted_merge(aggregated_groups, custom_weights)
        
        # Result should be 0.7 * ones + 0.3 * zeros = 0.7 * ones
        expected = torch.ones(self.hidden_dim, 16) * 0.7
        self.assertTrue(torch.allclose(global_weights['layer_0']['A'], expected))
        
    def test_compute_aggregation_quality(self):
        """Test aggregation quality metrics"""
        original_weights = [
            {'layer_0': {'A': torch.randn(self.hidden_dim, 8),
                        'B': torch.randn(self.hidden_dim, 8)}}
            for _ in range(3)
        ]
        aggregated_weights = {
            'layer_0': {'A': torch.randn(self.hidden_dim, 16),
                       'B': torch.randn(self.hidden_dim, 16)}
        }
        
        metrics = self.engine.compute_aggregation_quality(
            original_weights, aggregated_weights
        )
        
        self.assertIn('mean_reconstruction_error', metrics)
        self.assertIn('rank_compression_ratio', metrics)
        self.assertGreater(metrics['mean_reconstruction_error'], 0)
        
    def test_heterogeneous_ranks(self):
        """Test aggregation with heterogeneous ranks"""
        client_updates = {
            0: {'layer_0': {'A': torch.randn(self.hidden_dim, 4),
                           'B': torch.randn(self.hidden_dim, 4)}},
            1: {'layer_0': {'A': torch.randn(self.hidden_dim, 8),
                           'B': torch.randn(self.hidden_dim, 8)}},
            2: {'layer_0': {'A': torch.randn(self.hidden_dim, 16),
                           'B': torch.randn(self.hidden_dim, 16)}}
        }
        task_groups = {0: [0, 1, 2]}
        
        aggregated = self.engine.aggregate_all_groups(client_updates, task_groups)
        
        # Should successfully aggregate despite different ranks
        self.assertEqual(len(aggregated), 1)
        self.assertIn('layer_0', aggregated[0])
        
    def test_empty_group_error(self):
        """Test error handling for empty groups"""
        client_updates = {}
        group_clients = [0, 1]
        
        with self.assertRaises(ValueError):
            self.engine.aggregate_task_group(
                client_updates, group_clients, group_id=0
            )
            
    def test_aggregation_method_switching(self):
        """Test switching between aggregation methods"""
        # SVD method
        engine_svd = AggregationEngine(target_rank=16, aggregation_method='svd')
        self.assertEqual(engine_svd.aggregation_method, 'svd')
        
        # Average method
        engine_avg = AggregationEngine(target_rank=16, aggregation_method='average')
        self.assertEqual(engine_avg.aggregation_method, 'average')


class TestPrivacyVerifier(unittest.TestCase):
    """Test cases for PrivacyVerifier"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.verifier = PrivacyVerifier()
        
    def test_initialization(self):
        """Test verifier initialization"""
        self.assertEqual(len(self.verifier.verification_history), 0)
        
    def test_check_gradient_leakage_pass(self):
        """Test gradient leakage check - passing case"""
        original = torch.randn(100)
        aggregated = original * 0.5  # Reduced magnitude
        
        result = self.verifier.check_gradient_leakage(original, aggregated)
        
        self.assertIn('norm_ratio', result)
        self.assertTrue(result['passes_check'])
        self.assertLess(result['norm_ratio'], 1.0)
        
    def test_check_gradient_leakage_fail(self):
        """Test gradient leakage check - failing case"""
        original = torch.randn(100)
        aggregated = original * 2.0  # Increased magnitude
        
        result = self.verifier.check_gradient_leakage(original, aggregated)
        
        self.assertFalse(result['passes_check'])
        self.assertGreater(result['norm_ratio'], 1.0)
        
    def test_compute_privacy_score(self):
        """Test privacy score computation"""
        score = self.verifier.compute_privacy_score(
            task_group_size=10,
            n_total_clients=50
        )
        
        self.assertGreater(score, 0)
        self.assertIsInstance(score, float)
        
    def test_privacy_score_increases_with_group_size(self):
        """Test that privacy score increases with larger groups"""
        score_small = self.verifier.compute_privacy_score(5, 50)
        score_large = self.verifier.compute_privacy_score(20, 50)
        
        self.assertGreater(score_large, score_small)
        
    def test_check_update_indistinguishability(self):
        """Test update indistinguishability check"""
        updates_before = [torch.randn(50) for _ in range(5)]
        updates_after = [u + torch.randn(50) * 0.1 for u in updates_before]
        
        result = self.verifier.check_update_indistinguishability(
            updates_before, updates_after
        )
        
        self.assertIn('diversity', result)
        self.assertIn('passes_check', result)
        
    def test_membership_inference_resistance(self):
        """Test membership inference resistance computation"""
        hidden_dim = 768
        aggregated = {
            'layer_0': {'A': torch.randn(hidden_dim, 16),
                       'B': torch.randn(hidden_dim, 16)}
        }
        client_weights = [
            {'layer_0': {'A': torch.randn(hidden_dim, 8),
                        'B': torch.randn(hidden_dim, 8)}}
            for _ in range(5)
        ]
        
        resistance = self.verifier.compute_membership_inference_resistance(
            aggregated, client_weights
        )
        
        self.assertGreaterEqual(resistance, 0.0)
        self.assertLessEqual(resistance, 1.0)
        
    def test_weight_similarity(self):
        """Test weight similarity computation"""
        weights1 = {
            'layer_0': {'A': torch.randn(768, 8), 'B': torch.randn(768, 8)}
        }
        weights2 = {
            'layer_0': {'A': weights1['layer_0']['A'].clone(),
                       'B': weights1['layer_0']['B'].clone()}
        }
        
        similarity = self.verifier._compute_weight_similarity(weights1, weights2)
        
        # Identical weights should have similarity close to 1
        self.assertGreater(similarity, 0.99)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_compute_task_aware_weights_size_based(self):
        """Test size-based task weighting"""
        task_groups = {0: [0, 1, 2], 1: [3, 4]}
        
        weights = compute_task_aware_weights(task_groups)
        
        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(weights[0], 0.6, places=5)
        self.assertAlmostEqual(weights[1], 0.4, places=5)
        
    def test_compute_task_aware_weights_importance_based(self):
        """Test importance-based task weighting"""
        task_groups = {0: [0, 1], 1: [2, 3]}
        importance = {0: 0.7, 1: 0.3}
        
        weights = compute_task_aware_weights(task_groups, importance)
        
        self.assertAlmostEqual(weights[0], 0.7, places=5)
        self.assertAlmostEqual(weights[1], 0.3, places=5)
        
    def test_compute_task_aware_weights_normalization(self):
        """Test that weights sum to 1"""
        task_groups = {0: [0], 1: [1], 2: [2]}
        
        weights = compute_task_aware_weights(task_groups)
        
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow"""
    
    def test_end_to_end_aggregation(self):
        """Test complete aggregation pipeline"""
        print("\n" + "="*60)
        print("INTEGRATION TEST: End-to-End Aggregation")
        print("="*60)
        
        # Setup
        hidden_dim = 768
        n_clients = 6
        n_groups = 2
        
        print(f"\n[1] Creating {n_clients} client updates...")
        client_updates = {}
        for i in range(n_clients):
            rank = 8 if i % 2 == 0 else 16  # Heterogeneous ranks
            client_updates[i] = {
                'layer_0': {
                    'A': torch.randn(hidden_dim, rank),
                    'B': torch.randn(hidden_dim, rank)
                },
                'layer_1': {
                    'A': torch.randn(hidden_dim, rank),
                    'B': torch.randn(hidden_dim, rank)
                }
            }
        print(f"   Ranks: {[client_updates[i]['layer_0']['A'].shape[1] for i in range(n_clients)]}")
        
        # Task groups
        task_groups = {0: [0, 1, 2], 1: [3, 4, 5]}
        
        print(f"\n[2] Aggregating {n_groups} task groups...")
        engine = AggregationEngine(target_rank=16, aggregation_method='svd')
        aggregated_groups = engine.aggregate_all_groups(client_updates, task_groups)
        print(f"   Group 0: {len(task_groups[0])} clients")
        print(f"   Group 1: {len(task_groups[1])} clients")
        
        print(f"\n[3] Computing task-aware weights...")
        weights = compute_task_aware_weights(task_groups)
        print(f"   Weights: {weights}")
        
        print(f"\n[4] Merging groups globally...")
        global_weights = engine.weighted_merge(aggregated_groups, weights)
        print(f"   Global model: {len(global_weights)} layers")
        print(f"   Rank: {global_weights['layer_0']['A'].shape[1]}")
        
        print(f"\n[5] Computing quality metrics...")
        original_weights = [client_updates[i] for i in range(n_clients)]
        quality = engine.compute_aggregation_quality(original_weights, global_weights)
        print(f"   Reconstruction error: {quality['mean_reconstruction_error']:.4f}")
        print(f"   Rank compression: {quality['rank_compression_ratio']:.2f}x")
        
        print(f"\n[6] Verifying privacy...")
        verifier = PrivacyVerifier()
        
        # Privacy score
        privacy_score = verifier.compute_privacy_score(
            task_group_size=len(task_groups[0]),
            n_total_clients=n_clients
        )
        print(f"   Privacy score: {privacy_score:.3f}")
        
        # Membership inference resistance
        resistance = verifier.compute_membership_inference_resistance(
            global_weights, original_weights
        )
        print(f"   MI resistance: {resistance:.3f}")
        
        print("\n" + "="*60)
        print("[SUCCESS] End-to-end aggregation complete!")
        print("="*60)
        
        # Assertions
        self.assertEqual(len(aggregated_groups), n_groups)
        self.assertIn('layer_0', global_weights)
        self.assertGreater(privacy_score, 0)
        
    def test_privacy_verification_pipeline(self):
        """Test complete privacy verification"""
        hidden_dim = 768
        
        # Create updates
        original_grads = [torch.randn(hidden_dim) for _ in range(5)]
        
        # Simulate aggregation (averaging)
        aggregated_grad = torch.stack(original_grads).mean(dim=0)
        
        # Verify privacy
        verifier = PrivacyVerifier()
        
        # Check leakage
        leakage_results = []
        for orig in original_grads:
            result = verifier.check_gradient_leakage(orig, aggregated_grad)
            leakage_results.append(result)
        
        # Most should pass (aggregation reduces magnitude)
        passes = sum(r['passes_check'] for r in leakage_results)
        self.assertGreater(passes, 0)
        
    def test_aggregation_with_real_workflow(self):
        """Test aggregation in realistic federated learning scenario"""
        # Simulate Phase 3 integration
        hidden_dim = 768
        
        # Client updates from Phase 3
        client_updates = {
            i: {
                f'layer_{j}': {
                    'A': torch.randn(hidden_dim, 8),
                    'B': torch.randn(hidden_dim, 8)
                }
                for j in range(3)  # 3 layers
            }
            for i in range(10)  # 10 clients
        }
        
        # Task groups from Phase 1
        task_groups = {0: [0, 1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]}
        
        # Phase 4: Aggregate
        engine = AggregationEngine(target_rank=16)
        aggregated = engine.aggregate_all_groups(client_updates, task_groups)
        
        # Merge globally
        weights = compute_task_aware_weights(task_groups)
        global_weights = engine.weighted_merge(aggregated, weights)
        
        # Verify structure
        self.assertEqual(len(aggregated), 3)  # 3 task groups
        self.assertEqual(len(global_weights), 3)  # 3 layers
        
        for layer_name in global_weights:
            self.assertEqual(global_weights[layer_name]['A'].shape[0], hidden_dim)
            self.assertEqual(global_weights[layer_name]['B'].shape[0], hidden_dim)


def run_tests():
    """Run all tests and display results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAggregationEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestPrivacyVerifier))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 4 TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n[SUCCESS] All Phase 4 tests passed!")
    else:
        print("\n[FAILED] Some tests failed. See details above.")
    
    print("="*80)
    
    return result


if __name__ == '__main__':
    run_tests()
