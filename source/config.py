"""Configuration module for the machine learning pipeline.

This module defines the `Config` class, which stores all configurable parameters related to data loading,
model selection, training, and optimization. It ensures data integrity through validation checks and provides
convenient methods for loading and saving configurations in YAML format.
"""

import yaml
from dataclasses import dataclass, asdict
from typing import Optional
import source.utilities as utils


@dataclass(frozen=True)
class Config:
    """Configuration parameters for the pipeline.

    --- General Settings ---
    batch_size: Batch size for training.
    device: Compute device ('cpu' or 'cuda').
    num_epochs: Number of training epochs.

    --- Data Loader settings ---
    data_augmentation: Enables data augmentation.
    dataset: The dataset used for training/testing.
    dataset_directory: Directory for dataset storage.
    num_workers: How many subprocesses to use for data loading.
    shuffle_train: Whether to shuffle training data.

    --- Model Loader Settings ---
    model_name: Name of the model to be used (e.g. ResNet50).

    --- Optimizer Settings ---
    learning_rate: Initial learning rate.
    momentum: Momentum value for optimizers (e.g., SGD).
    optimizer: Optimizer type ('sgd', 'adam', etc.).

    --- Training Settings ---
    early_stopping: Enables early stopping.
    early_stopping_patience: Number of epochs before stopping.
    loss_function: Loss function (e.g., 'cross_entropy').
    use_validation: Enables validation during training.
    """

    # General settings
    batch_size: int = 128
    device: str = "cpu"
    num_epochs: int = 10

    # Data Loader settings
    data_augmentation: bool = False
    dataset: str = "cifar10"
    dataset_directory: str = "./data"
    num_workers: int = 2
    shuffle_train: bool = True

    # Model Loader settings
    model_name: str = "resnet18"

    # Optimizer settings
    learning_rate: float = 0.001
    momentum: Optional[float] = 0.9
    optimizer: str = "sgd"

    # Training settings
    early_stopping: bool = False
    early_stopping_patience: int = 5
    loss_function: str = "cross_entropy"
    use_validation: bool = True

    DEVICES = {
        "cpu",
        "cuda"
    }

    def __post_init__(self):
        """Validates configuration settings upon initialization."""

        self._validate_general_settings()
        self._validate_data_loader_settings()
        self._validate_model_loader_settings()
        self._validate_training_settings()
        self._validate_optimizer_settings()

    def _validate_general_settings(self) -> None:
        """Validates general configuration settings."""
        if self.batch_size <= 0:
            raise ValueError("Config 'batch_size' must be greater than 0.")
        if self.device not in self.DEVICES:
            raise ValueError(f"Invalid config 'device': {self.device}. "
                             f"Supported: {utils.list_possible_values(self.DEVICES)}")
        if self.num_epochs <= 0:
            raise ValueError("Config file 'num_epochs' must be greater than 0.")

    def _validate_data_loader_settings(self) -> None:
        """Validates data loader settings."""
        from source.stages.data_loader_stage import DataLoaderStage

        if not isinstance(self.data_augmentation, bool):
            raise TypeError("Config 'data_augmentation' must be a boolean.")
        if self.dataset not in DataLoaderStage.DATASET:
            raise ValueError(f"Unsupported config 'dataset': {self.dataset} "
                             f"Supported : {utils.list_possible_values(DataLoaderStage.DATASET)}")
        if not isinstance(self.shuffle_train, bool):
            raise TypeError("Config 'shuffle_train' must be a boolean.")

    def _validate_model_loader_settings(self) -> None:
        """Validates model-related settings."""
        from source.stages.model_loader_stage import ModelLoaderStage

        if self.model_name not in ModelLoaderStage.MODELS:
            raise ValueError(f"Unsupported config 'model_name': {self.model_name}. "
                             f"Supported: {utils.list_possible_values(ModelLoaderStage.MODELS)}")

    def _validate_training_settings(self) -> None:
        """Validates training-related settings."""
        from source.stages.training_stage import TrainingStage

        if not isinstance(self.early_stopping, bool):
            raise TypeError("Config 'early_stopping' must be a boolean.")
        if self.early_stopping and self.early_stopping_patience <= 0:
            raise ValueError("Config 'early_stopping_patience' must be a positive integer if 'early_stopping' is "
                             "enabled.")
        if self.loss_function not in TrainingStage.LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported config 'loss_function': {self.loss_function}. "
                             f"Supported: {utils.list_possible_values(TrainingStage.LOSS_FUNCTIONS)}")

    def _validate_optimizer_settings(self) -> None:
        """Validates optimizer-related settings."""
        from source.stages.training_stage import TrainingStage

        if self.learning_rate <= 0:
            raise ValueError("Config 'learning_rate' must be positive.")
        if self.momentum is not None and self.momentum <= 0:
            raise ValueError("Config 'momentum' must be a positive integer if provided.")
        if self.optimizer not in TrainingStage.OPTIMIZERS:
            raise ValueError(f"Unsupported config 'optimizer': {self.optimizer}.\n"
                             f"Supported: {utils.list_possible_values(TrainingStage.OPTIMIZERS)}")

    @classmethod
    def from_yaml(cls, filepath: str) -> "Config":
        """Loads configuration from a YAML file.

        Args:
            filepath (str): The path to the YAML configuration file.

        Returns:
            Config: A `Config` object initialized with the data from the YAML file.
        """
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
        return cls(**data)

    # TODO : @classmethod ?
    def to_yaml(self, filepath: str) -> None:
        """Saves the current configuration to a YAML file.

        Args:
            filepath (str): The path to the YAML file where the configuration will be saved.
        """
        with open(filepath, 'w') as file:
            yaml.safe_dump(asdict(self), file)
