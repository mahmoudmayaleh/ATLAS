"""
Phase 3: Split Federated Learning - Unit Tests

Tests for:
- LoRAAdapter
- SplitClient
- SplitServer
- Communication protocol
- Integration tests

Author: ATLAS Team
Date: December 2025
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from src.phase3_split_fl import (
    LoRAAdapter,
    SplitClient,
    SplitServer,
    get_split_point,
    federated_training_round,
    create_message
)


class TestLoRAAdapter(unittest.TestCase):
    """Test cases for LoRA adapter module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.in_dim = 768
        self.rank = 16
        self.batch_size = 4
        self.seq_len = 10
        
    def test_initialization(self):
        """Test LoRA adapter initialization"""
        adapter = LoRAAdapter(in_dim=self.in_dim, rank=self.rank)
        
        self.assertEqual(adapter.in_dim, self.in_dim)
        self.assertEqual(adapter.rank, self.rank)
        self.assertEqual(adapter.A.shape, (self.in_dim, self.rank))
        self.assertEqual(adapter.B.shape, (self.in_dim, self.rank))  # B has same shape as A
        
        # B should be initialized to zeros
        self.assertTrue(torch.allclose(adapter.B, torch.zeros_like(adapter.B)))
        
    def test_forward_2d(self):
        """Test forward pass with 2D input"""
        adapter = LoRAAdapter(in_dim=self.in_dim, rank=self.rank)
        x = torch.randn(self.batch_size, self.in_dim)  # Correct dimension
        
        output = adapter(x)
        
        self.assertEqual(output.shape, x.shape)
        
    def test_forward_3d(self):
        """Test forward pass with 3D input (sequence)"""
        adapter = LoRAAdapter(in_dim=self.in_dim, rank=self.rank)
        x = torch.randn(self.batch_size, self.seq_len, self.in_dim)  # Correct dimension
        
        output = adapter(x)
        
        self.assertEqual(output.shape, x.shape)
        
    def test_parameter_count(self):
        """Test parameter count"""
        adapter = LoRAAdapter(in_dim=self.in_dim, rank=self.rank)
        
        total_params = sum(p.numel() for p in adapter.parameters())
        expected_params = self.in_dim * self.rank * 2  # A + B
        
        self.assertEqual(total_params, expected_params)
        
    def test_reset_parameters(self):
        """Test parameter reset"""
        adapter = LoRAAdapter(in_dim=self.in_dim, rank=self.rank)
        
        # Modify parameters
        adapter.A.data.fill_(1.0)
        adapter.B.data.fill_(2.0)
        
        # Reset
        adapter.reset_parameters()
        
        # B should be zeros again
        self.assertTrue(torch.allclose(adapter.B, torch.zeros_like(adapter.B)))
        # A should be different from before
        self.assertFalse(torch.allclose(adapter.A, torch.ones_like(adapter.A)))
        
    def test_scaling(self):
        """Test LoRA scaling factor"""
        rank = 8
        alpha = 16.0
        adapter = LoRAAdapter(in_dim=self.in_dim, rank=rank, alpha=alpha)
        
        self.assertEqual(adapter.scaling, alpha / rank)


class TestSplitClient(unittest.TestCase):
    """Test cases for SplitClient"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.client_id = 0
        self.model_name = 'gpt2'
        self.rank_config = {0: 8, 1: 16, 2: 8}
        self.split_layer = 6
        
    def test_initialization(self):
        """Test client initialization"""
        client = SplitClient(
            client_id=self.client_id,
            model_name=self.model_name,
            rank_config=self.rank_config,
            split_layer=self.split_layer,
            device='cpu'
        )
        
        self.assertEqual(client.client_id, self.client_id)
        self.assertEqual(len(client.lora_adapters), len(self.rank_config))
        self.assertIsNotNone(client.optimizer)
        
    def test_lora_creation(self):
        """Test LoRA adapters are created correctly"""
        client = SplitClient(
            client_id=self.client_id,
            model_name=self.model_name,
            rank_config=self.rank_config,
            split_layer=self.split_layer
        )
        
        for layer_idx, rank in self.rank_config.items():
            layer_name = f'layer_{layer_idx}'
            self.assertIn(layer_name, client.lora_adapters)
            self.assertEqual(client.lora_adapters[layer_name].rank, rank)
            
    def test_compute_activations(self):
        """Test activation computation"""
        client = SplitClient(
            client_id=self.client_id,
            model_name=self.model_name,
            rank_config=self.rank_config,
            split_layer=self.split_layer
        )
        
        # Use actual hidden_dim from client (may be different due to dummy model)
        batch = {'inputs': torch.randn(2, client.hidden_dim)}
        activations = client.compute_activations(batch)
        
        self.assertEqual(len(activations.shape), 2)
        self.assertEqual(activations.shape[0], 2)  # batch size
        
    def test_get_lora_weights(self):
        """Test LoRA weight extraction"""
        client = SplitClient(
            client_id=self.client_id,
            model_name=self.model_name,
            rank_config=self.rank_config,
            split_layer=self.split_layer
        )
        
        weights = client.get_lora_weights()
        
        self.assertEqual(len(weights), len(self.rank_config))
        for layer_name in weights:
            self.assertIn('A', weights[layer_name])
            self.assertIn('B', weights[layer_name])
            
    def test_set_lora_weights(self):
        """Test LoRA weight setting"""
        client = SplitClient(
            client_id=self.client_id,
            model_name=self.model_name,
            rank_config=self.rank_config,
            split_layer=self.split_layer
        )
        
        # Get initial weights
        initial_weights = client.get_lora_weights()
        
        # Modify weights
        new_weights = {}
        for layer_name, weight_dict in initial_weights.items():
            new_weights[layer_name] = {
                'A': weight_dict['A'] * 2,
                'B': weight_dict['B'] + 1
            }
        
        # Set new weights
        client.set_lora_weights(new_weights)
        
        # Verify weights changed
        updated_weights = client.get_lora_weights()
        for layer_name in new_weights:
            self.assertTrue(
                torch.allclose(
                    updated_weights[layer_name]['A'],
                    new_weights[layer_name]['A']
                )
            )
            
    def test_memory_usage(self):
        """Test memory calculation"""
        client = SplitClient(
            client_id=self.client_id,
            model_name=self.model_name,
            rank_config=self.rank_config,
            split_layer=self.split_layer
        )
        
        memory_mb = client.get_memory_usage()
        
        self.assertGreater(memory_mb, 0)
        self.assertIsInstance(memory_mb, float)
        
    def test_train_step(self):
        """Test training step with gradients"""
        client = SplitClient(
            client_id=self.client_id,
            model_name=self.model_name,
            rank_config=self.rank_config,
            split_layer=self.split_layer
        )
        
        # Use actual hidden_dim from client
        batch = {'inputs': torch.randn(2, client.hidden_dim)}
        
        # Compute activations first
        activations = client.compute_activations(batch)
        
        # Create fake gradients from server
        activation_gradients = torch.randn_like(activations)
        
        # Train step
        updated_weights = client.train_step(batch, activation_gradients)
        
        self.assertIsInstance(updated_weights, dict)
        self.assertEqual(len(updated_weights), len(self.rank_config))


class TestSplitServer(unittest.TestCase):
    """Test cases for SplitServer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_name = 'gpt2'
        self.n_tasks = 3
        self.split_layer = 6
        self.num_classes = 10
        
    def test_initialization(self):
        """Test server initialization"""
        server = SplitServer(
            model_name=self.model_name,
            n_tasks=self.n_tasks,
            split_layer=self.split_layer,
            num_classes=self.num_classes,
            device='cpu'
        )
        
        self.assertEqual(server.n_tasks, self.n_tasks)
        self.assertEqual(len(server.task_heads), self.n_tasks)
        self.assertIsNotNone(server.optimizer)
        
    def test_task_heads_creation(self):
        """Test task heads are created correctly"""
        server = SplitServer(
            model_name=self.model_name,
            n_tasks=self.n_tasks,
            split_layer=self.split_layer,
            num_classes=self.num_classes
        )
        
        for i in range(self.n_tasks):
            task_name = f'task_{i}'
            self.assertIn(task_name, server.task_heads)
            
    def test_forward(self):
        """Test forward pass"""
        server = SplitServer(
            model_name=self.model_name,
            n_tasks=self.n_tasks,
            split_layer=self.split_layer,
            num_classes=self.num_classes
        )
        
        batch_size = 4
        activations = torch.randn(batch_size, server.hidden_dim)
        
        logits = server.forward(activations, task_id=0)
        
        self.assertEqual(logits.shape, (batch_size, self.num_classes))
        
    def test_compute_loss(self):
        """Test loss computation and gradient extraction"""
        server = SplitServer(
            model_name=self.model_name,
            n_tasks=self.n_tasks,
            split_layer=self.split_layer,
            num_classes=self.num_classes
        )
        
        batch_size = 4
        activations = torch.randn(batch_size, server.hidden_dim)
        labels = torch.randint(0, self.num_classes, (batch_size,))
        
        loss, activation_gradients = server.compute_loss(activations, labels, task_id=0)
        
        self.assertIsInstance(loss.item(), float)
        self.assertEqual(activation_gradients.shape, activations.shape)
        
    def test_evaluate(self):
        """Test evaluation mode"""
        server = SplitServer(
            model_name=self.model_name,
            n_tasks=self.n_tasks,
            split_layer=self.split_layer,
            num_classes=self.num_classes
        )
        
        batch_size = 4
        activations = torch.randn(batch_size, server.hidden_dim)
        labels = torch.randint(0, self.num_classes, (batch_size,))
        
        metrics = server.evaluate(activations, labels, task_id=0)
        
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        
    def test_aggregate_lora_weights(self):
        """Test LoRA weight aggregation"""
        server = SplitServer(
            model_name=self.model_name,
            n_tasks=self.n_tasks,
            split_layer=self.split_layer,
            num_classes=self.num_classes
        )
        
        # Create dummy client weights
        client_weights = {}
        for client_id in range(3):
            client_weights[client_id] = {
                'layer_0': {
                    'A': torch.randn(768, 16),
                    'B': torch.randn(16, 768)
                },
                'layer_1': {
                    'A': torch.randn(768, 16),
                    'B': torch.randn(16, 768)
                }
            }
        
        task_groups = {0: [0, 1], 1: [2]}
        
        aggregated = server.aggregate_lora_weights(client_weights, task_groups)
        
        self.assertEqual(len(aggregated), 2)  # 2 layers
        self.assertIn('layer_0', aggregated)
        self.assertIn('layer_1', aggregated)
        
    def test_multiple_tasks(self):
        """Test server handles multiple tasks correctly"""
        server = SplitServer(
            model_name=self.model_name,
            n_tasks=self.n_tasks,
            split_layer=self.split_layer,
            num_classes=self.num_classes
        )
        
        activations = torch.randn(2, server.hidden_dim)
        
        # Test each task
        for task_id in range(self.n_tasks):
            logits = server.forward(activations, task_id)
            self.assertEqual(logits.shape[0], 2)
            self.assertEqual(logits.shape[1], self.num_classes)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_get_split_point_gpt2(self):
        """Test split point for GPT-2"""
        split = get_split_point('gpt2')
        self.assertEqual(split, 6)
        
    def test_get_split_point_llama(self):
        """Test split point for LLaMA"""
        split = get_split_point('llama-7b')
        self.assertEqual(split, 16)
        
    def test_get_split_point_bert(self):
        """Test split point for BERT"""
        split = get_split_point('bert-base-uncased')
        self.assertEqual(split, 6)
        
    def test_get_split_point_unknown(self):
        """Test split point for unknown model"""
        split = get_split_point('unknown-model')
        self.assertEqual(split, 6)  # Default
        
    def test_create_message(self):
        """Test message creation"""
        msg = create_message(
            message_type='activations',
            sender_id=0,
            round_num=1,
            data={'activations': torch.randn(2, 768)}
        )
        
        self.assertEqual(msg['type'], 'activations')
        self.assertEqual(msg['sender_id'], 0)
        self.assertEqual(msg['round'], 1)
        self.assertIn('data', msg)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow"""
    
    def test_client_server_communication(self):
        """Test client-server communication flow"""
        # Create client
        client = SplitClient(
            client_id=0,
            model_name='gpt2',
            rank_config={0: 8, 1: 8},
            split_layer=6
        )
        
        # Create server
        server = SplitServer(
            model_name='gpt2',
            n_tasks=2,
            split_layer=6,
            num_classes=5
        )
        
        # Client computes activations
        batch = {'inputs': torch.randn(2, client.hidden_dim)}
        activations = client.compute_activations(batch)
        
        # Server computes loss and gradients
        labels = torch.randint(0, 5, (2,))
        loss, activation_gradients = server.compute_loss(activations, labels, task_id=0)
        
        # Client trains with gradients
        updated_weights = client.train_step(batch, activation_gradients)
        
        # Verify shapes
        self.assertEqual(activations.shape[0], 2)
        self.assertEqual(activation_gradients.shape, activations.shape)
        self.assertIsInstance(updated_weights, dict)
        
    def test_federated_training_round(self):
        """Test complete federated training round"""
        # Create clients
        clients = []
        for i in range(3):
            client = SplitClient(
                client_id=i,
                model_name='gpt2',
                rank_config={0: 8, 1: 8},
                split_layer=6
            )
            clients.append(client)
        
        # Create server
        server = SplitServer(
            model_name='gpt2',
            n_tasks=2,
            split_layer=6,
            num_classes=5
        )
        
        # Create training data
        train_data = {}
        for i in range(3):
            train_data[i] = {
                'inputs': torch.randn(2, clients[i].hidden_dim),
                'labels': torch.randint(0, 5, (2,))
            }
        
        # Task groups
        task_groups = {0: [0, 1], 1: [2]}
        
        # Run one round
        metrics = federated_training_round(clients, server, train_data, task_groups)
        
        # Verify metrics
        self.assertIn('avg_loss', metrics)
        self.assertIsInstance(metrics['avg_loss'], float)
        
    def test_weight_aggregation_and_broadcast(self):
        """Test weight aggregation and broadcast"""
        # Create clients
        clients = []
        for i in range(2):
            client = SplitClient(
                client_id=i,
                model_name='gpt2',
                rank_config={0: 8},
                split_layer=6
            )
            clients.append(client)
        
        # Create server
        server = SplitServer(
            model_name='gpt2',
            n_tasks=1,
            split_layer=6,
            num_classes=5
        )
        
        # Get initial weights from each client
        client_weights = {i: clients[i].get_lora_weights() for i in range(2)}
        
        # Aggregate
        task_groups = {0: [0, 1]}
        aggregated = server.aggregate_lora_weights(client_weights, task_groups)
        
        # Broadcast to clients
        for client in clients:
            client.set_lora_weights(aggregated)
        
        # Verify all clients have same weights
        weights_0 = clients[0].get_lora_weights()
        weights_1 = clients[1].get_lora_weights()
        
        for layer_name in weights_0:
            self.assertTrue(
                torch.allclose(weights_0[layer_name]['A'], weights_1[layer_name]['A'])
            )
            self.assertTrue(
                torch.allclose(weights_0[layer_name]['B'], weights_1[layer_name]['B'])
            )
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        print("\n" + "="*60)
        print("INTEGRATION TEST: End-to-End Workflow")
        print("="*60)
        
        # Setup
        n_clients = 3
        n_tasks = 2
        n_rounds = 2
        
        print(f"\n[1] Creating {n_clients} clients...")
        clients = []
        for i in range(n_clients):
            client = SplitClient(
                client_id=i,
                model_name='gpt2',
                rank_config={0: 8, 1: 8},
                split_layer=6
            )
            clients.append(client)
            print(f"   Client {i}: Memory = {client.get_memory_usage():.2f} MB")
        
        print(f"\n[2] Creating server with {n_tasks} tasks...")
        server = SplitServer(
            model_name='gpt2',
            n_tasks=n_tasks,
            split_layer=6,
            num_classes=10
        )
        print(f"   Server: {n_tasks} task heads, {server.num_classes} classes")
        
        print(f"\n[3] Running {n_rounds} federated training rounds...")
        task_groups = {0: [0, 1], 1: [2]}
        
        for round_num in range(n_rounds):
            # Create training data
            train_data = {}
            for i in range(n_clients):
                train_data[i] = {
                    'inputs': torch.randn(4, clients[i].hidden_dim),
                    'labels': torch.randint(0, 10, (4,))
                }
            
            # Run round
            metrics = federated_training_round(clients, server, train_data, task_groups)
            print(f"   Round {round_num + 1}: Loss = {metrics['avg_loss']:.4f}")
        
        print("\n" + "="*60)
        print("✓ End-to-end workflow successful!")
        print("="*60)
        
        # Assertions
        self.assertEqual(len(clients), n_clients)
        self.assertIsNotNone(server.global_lora_weights)


def run_tests():
    """Run all tests and display results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLoRAAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestSplitClient))
    suite.addTests(loader.loadTestsFromTestCase(TestSplitServer))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 3 TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All Phase 3 tests passed!")
    else:
        print("\n✗ Some tests failed. See details above.")
    
    print("="*80)
    
    return result


if __name__ == '__main__':
    run_tests()
