"""Pipeline stage for loading and preprocessing data.

TODO : This module should probably be split when we start adding more supported datasets for clarity.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import sklearn
from abc import ABC, abstractmethod
from source.config import Config
from source.pipeline_context import PipelineContext, PipelineData
from source.stages.base_pipeline_stage import BasePipelineStage
import source.utilities as utils


def create_split_stratified_samplers(dataset, validation_split=0.15, random_seed=None):
    """ Create stratified samplers for training and validation datasets.

    Args:
        dataset (Dataset): Dataset object with `targets` attribute (e.g., CIFAR-10).
        validation_split (float): Fraction of data to use for validation.
        random_seed (int): Seed for reproducibility.

    Returns:
        tuple: Two `SubsetRandomSampler` objects for training and validation.
    """
    # Validate parameters.
    if not (0 < validation_split < 1):
        raise ValueError("Dataset validation split should be between 0 and 1.")
    if not hasattr(dataset, "targets"):
        raise ValueError("Dataset must have a 'targets' attribute for stratified sampling.")

    targets = dataset.targets  # Targets are the class labels
    generator = torch.Generator().manual_seed(random_seed) if random_seed is not None else None

    # Create stratified split
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        range(len(targets)),
        test_size=validation_split,
        stratify=targets,
        random_state=random_seed
    )

    # Return samplers
    return (torch.utils.data.SubsetRandomSampler(train_idx, generator=generator),
            torch.utils.data.SubsetRandomSampler(val_idx, generator=generator))


class BaseDataLoader(ABC):
    """Strategy pattern used for loading datasets."""

    @abstractmethod
    def load_data(self, config: Config):
        """Load and prepare the dataset."""
        pass


class CIFAR10DataLoader(BaseDataLoader):
    """Loads and preprocesses data using CIFAR-10 dataset.

    CIFAR-10 dataset is a standard benchmark in deep learning. It involves 60,000 32x32 RGB images across 10 classes.
    """

    def load_data(self, config: Config):
        """
        Args:
            config (Config): Configuration object containing data-loading parameters.

        Returns:
            tuple :
                - train_loader : DataLoader for the training dataset.
                - test_loader : DataLoader for the test dataset.
                - validation_loader : DataLoader for the validation dataset.
                    If `config.use_validation` is True,
                    this loader contains a randomly split subset of the training dataset,
                    representing 15% of the total training data.
                    Returns `None` if validation is not enabled in the configuration.
        """
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = torchvision.datasets.CIFAR10(
            root=config.dataset_directory, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=config.dataset_directory, train=False, download=True, transform=transform
        )

        # TODO : Add new config parameters for random seed ?
        # TODO : add "generator=torch.Generator().manual_seed(config.random_seed)" to split to ensure reproducibility?
        # If using validation, split dataset into training and validation sets.
        # Must be done before getting the train_loader.
        validation_loader = None
        train_sampler = None
        if config.use_validation:
            train_sampler, val_sampler = create_split_stratified_samplers(train_dataset, validation_split=0.15)

            validation_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=config.batch_size,
                sampler=val_sampler,
                num_workers=config.num_workers
            )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),  # Shuffle is mutually exclusive with sampler option.
            sampler=train_sampler,
            num_workers=config.num_workers,
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        return PipelineData.DataLoader(
            class_names=train_dataset.classes,
            train_loader=train_loader,
            test_loader=test_loader,
            validation_loader=validation_loader
        )


class DataLoaderStage(BasePipelineStage):
    """TODO"""

    DATASET = {
        "cifar10": CIFAR10DataLoader()
    }

    def execute(self, config: Config, context: PipelineContext) -> PipelineContext:
        self._verify_data(config, context)

        data_loader = self.DATASET[config.dataset]
        context.data.data_loader = data_loader.load_data(config)

        # TODO : Replace print statements with logging
        print(f"Loaded dataset: {config.dataset}")
        print("Class Names:", context.data.data_loader.class_names)
        print("Number of Classes:", len(context.data.data_loader.class_names))

        return context

    def _verify_data(self, config: Config, context: PipelineContext) -> None:
        self._verify_config(config)
        self._verify_context(context)

    def _verify_config(self, config: Config) -> None:
        if config is None:
            raise ValueError("The Config is not initialized.")
        if config.dataset not in self.DATASET:
            raise ValueError(f"Unsupported config 'dataset': {config.dataset} "
                             f"Supported : {utils.list_possible_values(self.DATASET)}")
        if config.batch_size <= 0:
            raise ValueError("Config 'batch_size' must be greater than 0.")
        if not isinstance(config.shuffle_train, bool):
            raise TypeError("Config 'shuffle_train' must be a boolean.")

    def _verify_context(self, context: PipelineContext) -> None:
        if context is None:
            raise ValueError("The PipelineContext is not initialized.")
