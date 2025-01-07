"""TODO"""

from pipeline_orchestrator import PipelineOrchestrator
from source.stages.data_loader_stage import DataLoaderStage
from source.stages.evaluation_stage import EvaluationStage
from source.stages.model_loader_stage import ModelLoaderStage
from source.stages.training_stage import TrainingStage
from source.stages.visualization_stage import VisualizationStage


def main():
    # # Define the path to the config file
    # config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    #
    # # Initialize ConfigManager with the YAML config and get the config
    # config_manager = ConfigManager()
    # instance = config_manager.get_instance()
    # config = ConfigManager.initialize_from_yaml(config_path).config
    #

    # TODO : add stages
    orchestrator = PipelineOrchestrator()
    orchestrator.add_stage(DataLoaderStage())
    orchestrator.add_stage(ModelLoaderStage())
    orchestrator.add_stage(TrainingStage())
    orchestrator.add_stage(EvaluationStage())
    orchestrator.add_stage(VisualizationStage())

    orchestrator.run()


# TODO : Change this ?
if __name__ == "__main__":
    main()
