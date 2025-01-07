"""TODO

1 - DataLoaderStage
        Load the dataset and split into train/test loaders.
2 - AugmentationStage
        Apply augmentations to training data only.
3 - ModelLoaderStage
        Load pre-trained ResNet18 and modify for CIFAR-10.
4 - TrainingStage
        Train the model with W&B logging for metrics, loss, and images.
5 - EvaluationStage
        Compute validation/test metrics.
6 - VisualizationStage
        Plot loss/accuracy curves or use W&B visualizations.
7 - ModelStorageStage
        Save the trained model to disk or W&B artifacts.
8 - PredictionStage (Optional)
        Run inference on test data or new unseen data.
"""
import os

from source.config import Config
from source.pipeline_context import PipelineContext, PipelineData
from stages.data_loader_stage import DataLoaderStage
from stages.model_loader_stage import ModelLoaderStage
from stages.training_stage import TrainingStage


# TODO : Add default pipeline
class PipelineOrchestrator:
    """Manages the execution of pipeline stages, passing data between them."""

    STAGES = {
        "data_loader_stage": DataLoaderStage(),
        "model_loader_stage": ModelLoaderStage(),
        "training_stage": TrainingStage()
    }

    def __init__(self):
        self.stages = []

    # TODO
    # def create_pipeline_from_config(self, config=None):
    #     config = config
    #     if config is None:
    #         raise ValueError("Pipeline must have a valid configuration!")

    def _fetch_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
        return Config.from_yaml(config_path)

    def add_stage(self, stage):
        """Added stage must inherit from BasePipelineStage."""
        self.stages.append(stage)

    def run(self):
        # TODO : Remove
        # # Define the path to the config file
        # config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
        #
        # # Initialize ConfigManager with the YAML config and get the config
        # config = ConfigManager.initialize_from_yaml(config_path).config

        config = self._fetch_config()
        context = PipelineContext()
        # The output from each execute method becomes the input for the next stage.
        for stage in self.stages:
            # TODO : Remove and user logger
            print(f"***** Executing stage: {stage.__class__.__name__} *****")  # TODO: Remove of make cleaner.
            context = stage.execute(config, context)
