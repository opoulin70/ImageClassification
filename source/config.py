"""TODO"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict
import yaml


# TODO : Add relevant fields
@dataclass(frozen=True)
class Config:  # TODO : Make frozen=True as to make it immutable?
    """Configuration used in the pipeline.

    --- General Settings ---
    batch_size : Batch size for data loaders.
    device : Compute device (cuda, cpu).
    num_epochs : Number of training epochs.

    --- Data Loader settings ---
    data_augmentation: Enables data augmentation.
    dataset : The dataset used for training and testing data.
    dataset_directory : Directory to store the downloaded dataset.
    num_workers : How many subprocesses to use for data loading.
    shuffle_train: Whether to shuffle data for the training.

    --- Model Loader Settings ---
    model_name: Name of the model to be used (e.g. ResNet50).
    pretrained: Indicates whether to use a pre-trained model.

    --- Optimizer Settings ---
    learning_rate : Initial learning rate.
    momentum: Momentum parameter for SGD optimizer.
    optimizer: Optimizer type (sgd, adam, etc.).

    --- Training Settings ---
    early_stopping: Enables early stopping.
    early_stopping_patience: Number of epochs to wait before stopping training.
    loss_function: Loss function type (e.g. cross_entropy_loss).
    use_validation : Enables validation during training stage.
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
    pretrained: bool = True

    # Optimizer settings
    learning_rate: float = 0.001
    momentum: Optional[float] = 0.9
    optimizer: str = "sgd"

    # Training settings
    early_stopping: bool = False
    early_stopping_patience: int = 5
    loss_function: str = "cross_entropy"
    use_validation: bool = True

    # TODO : Move somewhere else ?
    DEVICES = {
        "cpu",
        "cuda"
    }

    # TODO
    def __post_init__(self):
        # TODO : Move dictionaries somewhere else ?
        # TODO : Create functions for each validation to make it more maintainable
        # Import the dictionaries
        from source.stages.model_loader_stage import ModelLoaderStage
        from source.stages.data_loader_stage import DataLoaderStage
        from source.stages.training_stage import TrainingStage

        # General settings
        if self.batch_size <= 0:
            raise ValueError("Config file 'batch_size' must be greater than 0.")
        if self.device not in self.DEVICES:
            raise ValueError("Config file 'device' must be either 'cpu' or 'cuda'.")
        if self.num_epochs <= 0:
            raise ValueError("Config file 'num_epochs' must be greater than 0.")

        # Data Loader settings
        if not isinstance(self.data_augmentation, bool):
            raise TypeError("Config file 'data_augmentation' must be a boolean.")
        if self.dataset not in DataLoaderStage.DATASET:
            raise ValueError(f"Unsupported dataset name: {self.dataset} used in Config file.\n"
                             f"Supported dataset are : {list(DataLoaderStage.DATASET.keys())}")
        if not isinstance(self.shuffle_train, bool):
            raise TypeError("Config file 'shuffle' must be a boolean.")

        # Model Loader settings
        if self.model_name not in ModelLoaderStage.MODELS:
            raise ValueError(f"Unsupported model name: {self.model_name} used in Config file.\n"
                             f"Supported dataset are : {list(ModelLoaderStage.MODELS.keys())}")
        if not isinstance(self.pretrained, bool):
            raise TypeError("Config file 'pretrained' must be a boolean.")

        # Training stage settings
        if not isinstance(self.early_stopping, bool):
            raise TypeError("Config file 'early_stopping' must be a boolean.")
        if self.early_stopping and self.early_stopping_patience <= 0:
            raise ValueError("Config file 'early_stopping_patience' must be a positive integer if 'early_stopping' is "
                             "enabled.")
        if self.loss_function not in TrainingStage.LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss function: {self.loss_function}. "
                             f"Available options are: {list(TrainingStage.LOSS_FUNCTIONS.keys())}")

        # Optimizer settings
        if self.learning_rate <= 0:
            raise ValueError("Config file 'learning_rate' must be positive.")
        if self.momentum is not None and self.momentum <= 0:
            raise ValueError("Config file 'momentum' must be positive if provided.")
        if self.optimizer not in TrainingStage.OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer}.\n"
                             f"Supported dataset are : {TrainingStage.OPTIMIZERS}")

    @classmethod
    def from_yaml(cls, filepath: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
        return cls(**data)

    # TODO : @classmethod ?
    def to_yaml(self, filepath: str) -> None:
        """Save the configuration to a YAML file."""
        with open(filepath, 'w') as file:
            yaml.safe_dump(asdict(self), file)
