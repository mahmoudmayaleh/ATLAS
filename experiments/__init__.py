"""
ATLAS Experiments Package

This package contains all experimental infrastructure for validating
and benchmarking the ATLAS federated learning system.
"""

from .config import (
    ExperimentConfig,
    TrainingConfig,
    GLUE_TASKS,
    MODELS,
    DEVICE_PROFILES,
    BASELINES,
    QUICK_EXPERIMENTS,
    FULL_EXPERIMENTS,
    get_dataset_config,
    get_model_config,
    get_device_profile,
    create_heterogeneous_clients
)

from .run_experiments import (
    ExperimentRunner,
    ExperimentResults,
    DatasetLoader,
    run_experiment_suite
)

from .metrics import (
    MemoryMetrics,
    CommunicationMetrics,
    TrainingMetrics,
    RoundMetrics,
    MetricsLogger,
    ComparisonAnalyzer
)

from .visualize import ExperimentVisualizer

__all__ = [
    # Config
    'ExperimentConfig',
    'TrainingConfig',
    'GLUE_TASKS',
    'MODELS',
    'DEVICE_PROFILES',
    'BASELINES',
    'QUICK_EXPERIMENTS',
    'FULL_EXPERIMENTS',
    'get_dataset_config',
    'get_model_config',
    'get_device_profile',
    'create_heterogeneous_clients',
    
    # Runner
    'ExperimentRunner',
    'ExperimentResults',
    'DatasetLoader',
    'run_experiment_suite',
    
    # Metrics
    'MemoryMetrics',
    'CommunicationMetrics',
    'TrainingMetrics',
    'RoundMetrics',
    'MetricsLogger',
    'ComparisonAnalyzer',
    
    # Visualization
    'ExperimentVisualizer',
]

__version__ = '1.0.0'
