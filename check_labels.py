import torch
from datasets import load_dataset
from torch.utils.data import Subset
import numpy as np

dataset = load_dataset("glue", "cola", split="train")
labels = np.array(dataset['label'])

# Simulate the partitioning in atlas_integrated.py
# _setup_multi_task_clients
indices = np.arange(len(dataset))
np.random.seed(42)
np.random.shuffle(indices)

# Label-stratified partitioning
class_indices = {}
for idx in indices:
    label = labels[idx]
    if label not in class_indices:
        class_indices[label] = []
    class_indices[label].append(idx)

clients_per_task = 3
max_samples = 3000

client_indices = [[] for _ in range(clients_per_task)]
for label, idxs in class_indices.items():
    splits = np.array_split(idxs, clients_per_task)
    for i in range(clients_per_task):
        client_indices[i].extend(splits[i])

for i in range(clients_per_task):
    np.random.shuffle(client_indices[i])
    client_indices[i] = client_indices[i][:max_samples]
    
    client_labels = labels[client_indices[i]]
    print(f"Client {i} (CoLA): {len(client_labels)} samples, label 0: {np.sum(client_labels == 0)}, label 1: {np.sum(client_labels == 1)}")

