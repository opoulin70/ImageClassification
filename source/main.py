import os
from source.pipeline_orchestrator import PipelineOrchestrator
from source.config import Config
from source.stages.data_loader_stage import DataLoaderStage
from source.stages.evaluation_stage import EvaluationStage
from source.stages.model_loader_stage import ModelLoaderStage
from source.stages.training_stage import TrainingStage
from source.stages.visualization_stage import VisualizationStage


def main():
    # Load the configuration from a YAML file.
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    config = Config.from_yaml(config_path)

    # Create the orchestrator and add the stages.
    orchestrator = PipelineOrchestrator()
    orchestrator.add_stage(DataLoaderStage())
    orchestrator.add_stage(ModelLoaderStage())
    orchestrator.add_stage(TrainingStage())
    orchestrator.add_stage(EvaluationStage())
    orchestrator.add_stage(VisualizationStage())

    # Execute the pipeline.
    orchestrator.run(config)


# TODO : Change this ?
if __name__ == "__main__":
    main()
