"""Pipeline Orchestrator for Machine Learning Workflows.

This module defines the `PipelineOrchestrator` class, which manages the execution of various stages in a machine
learning pipeline. Users can modify the order or add new stages as needed. Each stage is implemented as a separate
module, allowing easy extensibility.

Pipeline Stages :
1 - DataLoaderStage: Load the dataset and split into train/test loaders.
2 - AugmentationStage (planned): Apply augmentations to training data.
3 - ModelLoaderStage: Loads and initializes the selected model.
4 - TrainingStage: Train the model while saving training metrics.
5 - EvaluationStage: Compute validation/test metrics.
6 - VisualizationStage: Plot loss/accuracy curves.
7 - ModelStorageStage (planned): Save the trained model for future use.
8 - PredictionStage (planned): Runs inference on new/unseen data.
"""

from source.config import Config
from source.stages.base_pipeline_stage import BasePipelineStage
from source.pipeline_context import PipelineContext
from source.stages.data_loader_stage import DataLoaderStage
from source.stages.model_loader_stage import ModelLoaderStage
from source.stages.training_stage import TrainingStage
from source.stages.evaluation_stage import EvaluationStage
from source.stages.visualization_stage import VisualizationStage


# TODO : Add default pipeline. Useful for creating a pipeline without having to manually add stages.
class PipelineOrchestrator:
    """Manages the execution of sequential pipeline stages in a machine learning workflow."""

    # TODO : Add all stages. Useful for when adding stages from parsing a file, etc.
    STAGES = {
        "data_loader_stage": DataLoaderStage(),
        "model_loader_stage": ModelLoaderStage(),
        "training_stage": TrainingStage(),
        "evaluation_stage": EvaluationStage(),
        "visualization_stage": VisualizationStage()
    }

    def __init__(self):
        """Initializes an empty pipeline with no predefined stages."""
        self.stages = []

    def add_stage(self, stage: BasePipelineStage) -> None:
        """Adds a new stage to the pipeline.

        Args:
            stage (BasePipelineStage): The stage to be added. Must inherit from `BasePipelineStage`.
        """
        self.stages.append(stage)

    def run(self, config: Config) -> None:
        """Executes all pipeline stages sequentially.

        Args:
            config (Config): Configuration object containing pipeline parameters.
        """
        context = PipelineContext()

        # The output from each execute method becomes the input for the next stage.
        for stage in self.stages:
            print(f"***** Executing stage: {stage.__class__.__name__} *****")  # TODO : Replace with logging.
            context = stage.execute(config, context)
