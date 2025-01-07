"""TODO
The code supports training on CUDA-enabled GPUs.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from source.config import Config
from source.pipeline_context import PipelineContext
from source.stages.base_pipeline_stage import BasePipelineStage
from abc import ABC, abstractmethod


class BaseModelLoader(ABC):
    """Strategy pattern used for loading models."""

    @abstractmethod
    def load_model(self, num_classes):
        pass


class ResNet18ModelLoader(BaseModelLoader):

    # TODO: Cache model ?
    def load_model(self, num_classes) -> nn.Module:
        """Loads a pre-trained ResNet18 model."""
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model


class ModelLoaderStage(BasePipelineStage):

    MODELS = {
        "resnet18": ResNet18ModelLoader()
    }

    # TODO : Split this into different functions or make some sort of ConfigValidator utility class
    def _verify_data(self, config: Config, context: PipelineContext):
        if config is None:
            raise ValueError("The Config is not initialized.")
        if PipelineContext is None:
            raise ValueError('The PipelineContext is not initialized.')
        if config.model_name not in self.MODELS:
            raise ValueError(f"Unsupported model name: {config.model_name}.")
        if config.device not in {"cpu", "cuda"}:
            raise ValueError("Config 'device' must be either 'cpu' or 'cuda'.")
        if config.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device specified but no GPU is available.")

    def execute(self, config: Config, context: PipelineContext):
        """ Loads the specified model, configures it for the given device, and updates the context.

        Args:
            config (Config): Configuration object with model and device specifications.
            context (PipelineContext): Context containing pipeline state.

        Returns:
            PipelineContext: Updated pipeline context with the loaded model.
        """
        self._verify_data(config, context)

        model_loader = self.MODELS[config.model_name]
        model = model_loader.load_model(len(context.data.data_loader.class_names))

        model = model.to(config.device)

        context.data.model = model
        # TODO : Use logger instead
        print(f"Model {config.model_name} successfully loaded.")

        return context
