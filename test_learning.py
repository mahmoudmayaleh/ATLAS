import torch
from experiments.atlas_integrated import ATLASExperiment
from experiments.config import ExperimentConfig

config = ExperimentConfig(
    model_name="distilbert-base-uncased",
    num_clients=2,
    num_rounds=1,
    local_epochs=1,
    batch_size=16,
    mode="atlas",
    seed=42,
    fingerprint_samples=16,
    fingerprint_epochs=1,
    fingerprint_batches=2
)

exp = ATLASExperiment(config)
exp.run()
