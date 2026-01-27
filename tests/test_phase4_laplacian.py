"""
Tests for Phase 4 Laplacian Aggregation (MIRA-Compliant)

Tests the CORRECTED implementation where:
- Each client maintains its OWN model
- Laplacian regularization nudges similar tasks together
- No global averaging (models remain distinct)
"""

import unittest
import torch
import numpy as np
from src.phase4_laplacian import (
    TaskGraph,
    LaplacianAggregation,
    compute_adjacency_weights
)


class TestTaskGraph(unittest.TestCase):
    """Test TaskGraph construction and operations"""
    
    def test_basic_graph_construction(self):
        """Test creating graph with edges"""
        graph = TaskGraph()
        
        graph.add_edge(0, 1, 0.8)
        graph.add_edge(0, 2, 0.6)
        graph.add_edge(1, 0, 0.8)
        
        self.assertEqual(len(graph.get_neighbors(0)), 2)
        self.assertIn(1, graph.get_neighbors(0))
        self.assertIn(2, graph.get_neighbors(0))
        self.assertEqual(graph.get_edge_weight(0, 1), 0.8)
    
    def test_from_task_clusters(self):
        """Test building graph from clustering results"""
        task_clusters = {
            0: [0, 1, 2],  # Group 0: clients 0,1,2
            1: [3, 4]      # Group 1: clients 3,4
        }
        
        graph = TaskGraph.from_task_clusters(task_clusters)
        
        # Client 0 should have neighbors 1,2 (same group)
        neighbors_0 = graph.get_neighbors(0)
        self.assertEqual(len(neighbors_0), 2)
        self.assertIn(1, neighbors_0)
        self.assertIn(2, neighbors_0)
        
        # Client 3 should have neighbor 4 (same group)
        neighbors_3 = graph.get_neighbors(3)
        self.assertEqual(len(neighbors_3), 1)
        self.assertIn(4, neighbors_3)
        
        # Client 0 should NOT have neighbor 3 (different group)
        self.assertNotIn(3, neighbors_0)
    
    def test_weighted_graph(self):
        """Test graph with custom similarity weights"""
        task_clusters = {0: [0, 1, 2]}
        
        # Custom similarity matrix
        similarity = np.array([
            [1.0, 0.9, 0.5],
            [0.9, 1.0, 0.6],
            [0.5, 0.6, 1.0]
        ])
        
        graph = TaskGraph.from_task_clusters(task_clusters, similarity)
        
        # Check edge weights match similarities
        self.assertAlmostEqual(graph.get_edge_weight(0, 1), 0.9)
        self.assertAlmostEqual(graph.get_edge_weight(0, 2), 0.5)


class TestLaplacianAggregation(unittest.TestCase):
    """Test MIRA's Laplacian regularization"""
    
    def setUp(self):
        self.hidden_dim = 64
        self.rank = 8
        self.aggregator = LaplacianAggregation(eta=0.1)
    
    def _create_model(self):
        """Create dummy LoRA model"""
        return {
            'layer_0': {
                'A': torch.randn(self.hidden_dim, self.rank),
                'B': torch.randn(self.hidden_dim, self.rank)
            }
        }
    
    def test_no_neighbors_no_update(self):
        """Test that clients without neighbors stay unchanged"""
        # Create client models
        client_models = {
            0: self._create_model(),
            1: self._create_model()
        }
        
        # Empty graph (no neighbors)
        graph = TaskGraph()
        
        # Apply Laplacian update
        updated = self.aggregator.laplacian_update(client_models, graph)
        
        # Models should be unchanged
        self.assertTrue(torch.allclose(
            updated[0]['layer_0']['A'],
            client_models[0]['layer_0']['A']
        ))
    
    def test_neighbors_pull_models_together(self):
        """Test that Laplacian regularization pulls neighbors together"""
        # Create two different models
        model_0 = {
            'layer_0': {
                'A': torch.ones(self.hidden_dim, self.rank),
                'B': torch.ones(self.hidden_dim, self.rank)
            }
        }
        model_1 = {
            'layer_0': {
                'A': torch.zeros(self.hidden_dim, self.rank),
                'B': torch.zeros(self.hidden_dim, self.rank)
            }
        }
        
        client_models = {0: model_0, 1: model_1}
        
        # Create graph where 0 and 1 are neighbors
        graph = TaskGraph()
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 0, 1.0)
        
        # Apply Laplacian update
        updated = self.aggregator.laplacian_update(client_models, graph)
        
        # Updated models should be closer to each other
        # Model 0 was pulled toward model 1 (zeros)
        # So updated[0] should be less than original (ones)
        self.assertTrue(
            torch.mean(updated[0]['layer_0']['A']) < 
            torch.mean(model_0['layer_0']['A'])
        )
        
        # Model 1 was pulled toward model 0 (ones)
        # So updated[1] should be greater than original (zeros)
        self.assertTrue(
            torch.mean(updated[1]['layer_0']['A']) > 
            torch.mean(model_1['layer_0']['A'])
        )
    
    def test_models_remain_distinct(self):
        """Test that models DON'T become identical (no averaging)"""
        # Create different models
        client_models = {
            0: {
                'layer_0': {
                    'A': torch.ones(self.hidden_dim, self.rank),
                    'B': torch.ones(self.hidden_dim, self.rank)
                }
            },
            1: {
                'layer_0': {
                    'A': torch.zeros(self.hidden_dim, self.rank),
                    'B': torch.zeros(self.hidden_dim, self.rank)
                }
            }
        }
        
        # Create neighbor relationship
        graph = TaskGraph()
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 0, 1.0)
        
        # Apply multiple rounds of updates
        for _ in range(5):
            client_models = self.aggregator.laplacian_update(client_models, graph)
        
        # Models should be closer but NOT identical
        distance = torch.norm(
            client_models[0]['layer_0']['A'] - 
            client_models[1]['layer_0']['A']
        ).item()
        
        # Distance should be non-zero (models distinct)
        self.assertGreater(distance, 0.1)
        
        # But closer than original distance (1.0 apart initially)
        original_distance = torch.norm(
            torch.ones(self.hidden_dim, self.rank) - 
            torch.zeros(self.hidden_dim, self.rank)
        ).item()
        self.assertLess(distance, original_distance)
    
    def test_heterogeneous_ranks(self):
        """Test handling clients with different LoRA ranks"""
        # Create models with different ranks
        client_models = {
            0: {
                'layer_0': {
                    'A': torch.ones(self.hidden_dim, 8),
                    'B': torch.ones(self.hidden_dim, 8)
                }
            },
            1: {
                'layer_0': {
                    'A': torch.zeros(self.hidden_dim, 16),
                    'B': torch.zeros(self.hidden_dim, 16)
                }
            }
        }
        
        graph = TaskGraph()
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 0, 1.0)
        
        # Should not raise error
        updated = self.aggregator.laplacian_update(client_models, graph)
        
        # Output ranks should match input ranks
        self.assertEqual(updated[0]['layer_0']['A'].shape[1], 8)
        self.assertEqual(updated[1]['layer_0']['A'].shape[1], 16)
    
    def test_multiple_neighbors(self):
        """Test client with multiple neighbors"""
        # Create 4 clients
        client_models = {
            i: {
                'layer_0': {
                    'A': torch.full((self.hidden_dim, self.rank), float(i)),
                    'B': torch.full((self.hidden_dim, self.rank), float(i))
                }
            }
            for i in range(4)
        }
        
        # Client 0 has neighbors 1, 2, 3
        graph = TaskGraph()
        for i in [1, 2, 3]:
            graph.add_edge(0, i, 1.0 / 3.0)  # Equal weights
        
        # Apply update
        updated = self.aggregator.laplacian_update(client_models, graph)
        
        # Client 0 should be pulled toward average of neighbors
        # Original: 0, Neighbors: 1,2,3 (avg=2)
        # Updated should be between 0 and 2
        updated_mean = torch.mean(updated[0]['layer_0']['A']).item()
        self.assertGreater(updated_mean, 0.0)
        self.assertLess(updated_mean, 2.0)
    
    def test_diversity_computation(self):
        """Test that model diversity can be measured"""
        # Create diverse models
        client_models = {
            0: {
                'layer_0': {
                    'A': torch.ones(self.hidden_dim, self.rank),
                    'B': torch.ones(self.hidden_dim, self.rank)
                }
            },
            1: {
                'layer_0': {
                    'A': torch.zeros(self.hidden_dim, self.rank),
                    'B': torch.zeros(self.hidden_dim, self.rank)
                }
            }
        }
        
        diversity = self.aggregator.compute_model_diversity(client_models)
        
        self.assertIn('mean_diversity', diversity)
        self.assertGreater(diversity['mean_diversity'], 0.0)


class TestAdjacencyWeights(unittest.TestCase):
    """Test adjacency weight computation"""
    
    def test_uniform_weights(self):
        """Test uniform weighting (default)"""
        task_clusters = {0: [0, 1, 2]}
        
        weights = compute_adjacency_weights(task_clusters, method='uniform')
        
        # All edges should have weight 1/(n-1) = 1/2
        self.assertAlmostEqual(weights[(0, 1)], 0.5)
        self.assertAlmostEqual(weights[(0, 2)], 0.5)
        self.assertAlmostEqual(weights[(1, 0)], 0.5)
    
    def test_similarity_weights(self):
        """Test similarity-based weighting"""
        task_clusters = {0: [0, 1, 2]}
        
        # Custom similarities
        similarities = np.array([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.3],
            [0.2, 0.3, 1.0]
        ])
        
        weights = compute_adjacency_weights(
            task_clusters,
            gradient_similarities=similarities,
            method='similarity'
        )
        
        # Weight from 0->1 should be higher than 0->2 (higher similarity)
        self.assertGreater(weights[(0, 1)], weights[(0, 2)])


class TestIntegration(unittest.TestCase):
    """Integration tests for full MIRA workflow"""
    
    def test_full_workflow(self):
        """Test complete workflow from clustering to Laplacian update"""
        # Setup
        hidden_dim = 64
        rank = 8
        
        # 1. Create client models (simulating Phase 3 output)
        client_models = {
            i: {
                'layer_0': {
                    'A': torch.randn(hidden_dim, rank),
                    'B': torch.randn(hidden_dim, rank)
                }
            }
            for i in range(6)
        }
        
        # 2. Task clustering results (simulating Phase 1 output)
        task_clusters = {
            0: [0, 1, 2],  # Task group 0: sentiment analysis
            1: [3, 4, 5]   # Task group 1: question answering
        }
        
        # 3. Build task graph
        graph = TaskGraph.from_task_clusters(task_clusters)
        
        # 4. Apply Laplacian regularization
        aggregator = LaplacianAggregation(eta=0.1)
        updated_models = aggregator.laplacian_update(client_models, graph)
        
        # 5. Verify results
        # - All clients should have updated models
        self.assertEqual(len(updated_models), 6)
        
        # - Models within same group should be more similar after update
        # (but not identical!)
        diversity_before = aggregator.compute_model_diversity({
            0: client_models[0],
            1: client_models[1]
        })['mean_diversity']
        
        diversity_after = aggregator.compute_model_diversity({
            0: updated_models[0],
            1: updated_models[1]
        })['mean_diversity']
        
        self.assertLess(diversity_after, diversity_before)
        self.assertGreater(diversity_after, 0.0)  # Still distinct
        
        # - Models from different groups should remain very different
        cross_group_distance = aggregator._model_distance(
            updated_models[0],  # From group 0
            updated_models[3]   # From group 1
        )
        same_group_distance = aggregator._model_distance(
            updated_models[0],  # From group 0
            updated_models[1]   # Also from group 0
        )
        self.assertGreater(cross_group_distance, same_group_distance)


if __name__ == '__main__':
    unittest.main()
