"""TODO
The code supports training on CUDA-enabled GPUs.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from source.config import Config
from source.pipeline_context import PipelineContext, PipelineData
from source.stages.base_pipeline_stage import BasePipelineStage
from tqdm import tqdm


class TrainingStage(BasePipelineStage):
    """Stage for training a model."""

    # TODO : Move these dict ?
    OPTIMIZERS = {
        "sgd",
        "adam"
    }

    # Check the PyTorch documentation for more in-depth information on each loss function :
    # https://pytorch.org/docs/stable/nn.html#loss-functions
    LOSS_FUNCTIONS = {
        # Binary Cross-Entropy Loss: For binary classification tasks with probabilities.
        "bce": nn.BCELoss(),
        # Binary Cross-Entropy Loss with logits: Combines sigmoid activation and binary cross-entropy.
        "bce_with_logits": nn.BCEWithLogitsLoss(),
        # Cross-Entropy Loss: For multi-class classification tasks.
        "cross_entropy": nn.CrossEntropyLoss(),
        # Negative Log Likelihood Loss: For classification with log probabilities.
        "nll": nn.NLLLoss()
    }

    # TODO : Check device
    # TODO : Split this into different functions or make some sort of ConfigValidator utility class
    def _verify_data(self, config: Config, context: PipelineContext):
        if config is None:
            raise ValueError("The Config is not initialized.")
        if PipelineContext is None:
            raise ValueError("The PipelineContext is not initialized.")
        if context.data.model is None:
            raise ValueError("The model is missing from the PipelineContext.")
        if context.data.data_loader is None:
            raise ValueError("The data_loader is missing from the PipelineContext.")
        if config.use_validation and context.data.data_loader.validation_loader is None:
            raise ValueError("The validation_loader is missing from the PipelineContext data_loader.")
        if not isinstance(context.data.model, nn.Module):
            raise TypeError("The provided model must be an instance of nn.Module.")

        # Optimizer
        if config.optimizer not in TrainingStage.OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer type: {config.optimizer}.\n"
                             f"Supported dataset are : {self.OPTIMIZERS}")
        if config.learning_rate <= 0:
            raise ValueError("Config file 'learning_rate' must be positive.")
        if config.momentum <= 0:
            raise ValueError("Config file 'momentum' must be positive if provided.")

        # Criterion
        if config.loss_function not in self.LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss function: {config.loss_function}. "
                             f"Available options are: {list(self.LOSS_FUNCTIONS.keys())}")
        pass

    def _initialize_optimizer(self, config: Config, model: nn.Module):
        """Initializes the optimizer based on the configuration.

        Returns:
            The PyTorch optimizer instance.
        """
        # TODO : Add more optimizers
        if config.optimizer == "sgd":
            return optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
        elif config.optimizer == "adam":
            return optim.Adam(model.parameters(), lr=config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {config.optimizer}.\n"
                             f"Supported dataset are : {self.OPTIMIZERS}")

    # TODO : Use logging.
    # TODO : Use checkpoints to save model
    def _train(self, config: Config, model: nn.Module, train_loader, optimizer, criterion):
        """Trains the model.

        Returns:
            The training metrics used during the evaluation stage.
        """
        training_metrics = PipelineData.TrainingMetrics()
        start_time = time.time()
        model.train()  # Set the model to training mode

        for epoch in range(config.num_epochs):
            running_loss = 0.0

            # Visual progress bar for batches
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}", unit="batch", leave=True)

            # Enumerate instead of iter to track batch index.
            for batch_index, (inputs, labels) in enumerate(progress_bar):
                # Move data to GPU if available
                inputs = inputs.to(config.device)
                labels = labels.to(config.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                progress_bar.set_postfix(batch_loss=loss.item(), avg_loss=running_loss / (batch_index + 1))

            progress_bar.close()

            # Add average epoch training loss to the training metrics.
            training_metrics.epoch_losses.append(running_loss / len(train_loader))

            # Validation phase

        # Finished training
        training_metrics.training_time = time.time() - start_time
        training_metrics.final_loss = training_metrics.epoch_losses[-1]

        return training_metrics

    # TODO : Use logging.
    def execute(self, config: Config, context: PipelineContext):
        self._verify_data(config, context)

        model = context.data.model
        train_loader = context.data.data_loader.train_loader
        criterion = self.LOSS_FUNCTIONS[config.loss_function]

        model.to(config.device)

        optimizer = self._initialize_optimizer(config, model)

        # TODO : Remove or make cleaner ?
        print(f"Starting training for {config.num_epochs} epochs on device: {config.device}")

        try:
            context.data.training_metrics = self._train(
                config,
                model,
                train_loader,
                optimizer,
                criterion)
        except RuntimeError as e:
            print(f"Runtime error during training: {e}")
            if "out of memory" in str(e):
                print("Try reducing the batch size.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise
        finally:
            if config.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # TODO : Log ?
        print(f"Finished training in {context.data.training_metrics.training_time:.2f}s")

        return context
