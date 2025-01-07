"""Pipeline stage for loading and preprocessing data."""
import torch
import torchvision
import torchvision.transforms as transforms
import sklearn
from source.stages.base_pipeline_stage import BasePipelineStage
from abc import ABC, abstractmethod
from source.config import Config
from source.pipeline_context import PipelineContext, PipelineData


def create_split_stratified_samplers(dataset, validation_split=0.15, random_seed=42):
    """ Create stratified samplers for training and validation datasets.

    Args:
        dataset (Dataset): Dataset object with `targets` attribute (e.g., CIFAR-10).
        validation_split (float): Fraction of data to use for validation.
        random_seed (int): Seed for reproducibility.

    Returns:
        tuple: Two `SubsetRandomSampler` objects for training and validation.
    """

    # Extract targets (class labels)
    targets = dataset.targets

    # Create stratified split
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        range(len(targets)),
        test_size=validation_split,
        stratify=targets,
        random_state=random_seed
    )

    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    return train_sampler, val_sampler


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
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

        class_names = train_dataset.classes
        shuffle = config.shuffle_train

        # TODO : Add new config parameters
        # TODO : add "generator=torch.Generator().manual_seed(config.random_seed)" to split to ensure reproducibility?
        # If using validation, split dataset into training and validation sets
        validation_loader = None
        train_sampler = None
        if config.use_validation:
            shuffle = False  # Sampler option is mutually exclusive with shuffle
            train_sampler, val_sampler = create_split_stratified_samplers(train_dataset, validation_split=0.15)

            validation_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                sampler=val_sampler,
                num_workers=config.num_workers
            )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
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
            class_names=class_names,
            train_loader=train_loader,
            test_loader=test_loader,
            validation_loader=validation_loader
        )


class DataLoaderStage(BasePipelineStage):
    # TODO : Move this somewhere else ?
    DATASET = {
        "cifar10": CIFAR10DataLoader()
    }

    # TODO : Separate into multiple functions (verify_config, verify_context?)
    def _verify_data(self, config: Config, context: PipelineContext):
        if config is None:
            raise ValueError("The Config is not initialized.")
        if PipelineContext is None:
            raise ValueError("The PipelineContext is not initialized.")
        if config.dataset not in self.DATASET:
            raise ValueError(f"Unsupported model name: {config.dataset}.")
        if config.batch_size <= 0:
            raise ValueError("Config 'batch_size' must be greater than 0.")

    def execute(self, config: Config, context: PipelineContext):
        self._verify_data(config, context)

        data_loader = self.DATASET[config.dataset]
        context.data.data_loader = data_loader.load_data(config)

        # TODO : Remove and use logger
        print("Class Names:", context.data.data_loader.class_names)
        print("Number of Classes:", len(context.data.data_loader.class_names))

        # TODO : Show a few sample images from dataset ? Violates SOLID principles ?
        # _show_sample_images(train_loader)

        return context
