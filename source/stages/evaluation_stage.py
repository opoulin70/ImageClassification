import torch
from source.config import Config
from source.pipeline_context import PipelineContext, PipelineData
from source.stages.base_pipeline_stage import BasePipelineStage


class EvaluationStage(BasePipelineStage):

    # TODO : Split this into different functions or make some sort of ConfigValidator utility class
    # TODO : Check everything
    def _verify_data(self, config: Config, context: PipelineContext):
        if config is None:
            raise ValueError("The Config is not initialized.")
        if PipelineContext is None:
            raise ValueError("The PipelineContext is not initialized.")

    # TODO : Use log
    def _show_evaluation_metrics(self, evaluation_metrics):
        print(f"Accuracy of the model on the test images: {evaluation_metrics.accuracy:.2f}%")

    def _evaluate_model(self, config: Config, model, test_loader):
        evaluation_metrics = PipelineData.EvaluationMetrics()

        model.eval()  # Set the model to evaluation mode
        correct_predictions = 0
        total_samples = len(test_loader.dataset)

        with torch.no_grad():
            for inputs, labels in test_loader:
                # Move data to GPU if available
                inputs = inputs.to(config.device)
                labels = labels.to(config.device)

                # Forward pass: compute outputs (predictions) from inputs
                outputs = model(inputs)

                # Get predicted classes (the index with the highest score)
                _, predicted = torch.max(outputs, dim=1)

                # Update correct predictions
                correct_predictions += (predicted == labels).sum().item()  # Count correct predictions

        # Calculate accuracy percentage
        evaluation_metrics.accuracy = (correct_predictions / total_samples) * 100

        return evaluation_metrics

    def execute(self, config: Config, context: PipelineContext):
        self._verify_data(config, context)

        context.data.evaluation_metrics = self._evaluate_model(
            config,
            context.data.model,
            context.data.data_loader.test_loader
        )

        self._show_evaluation_metrics(context.data.evaluation_metrics)

        return context
