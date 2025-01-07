import torch
from matplotlib import pyplot as plt

from source.config import Config
from source.pipeline_context import PipelineContext, PipelineData
from source.stages.base_pipeline_stage import BasePipelineStage


class VisualizationStage(BasePipelineStage):

    # TODO : Split this into different functions or make some sort of ConfigValidator utility class
    # TODO : Check everything
    def _verify_data(self, config: Config, context: PipelineContext):
        if config is None:
            raise ValueError("The Config is not initialized.")
        if PipelineContext is None:
            raise ValueError("The PipelineContext is not initialized.")

    def _plot_epoch_losses(self, epoch_losses):
        """Visualize the training loss over epochs."""
        title = "Epoch Losses"
        epochs = list(range(1, len(epoch_losses) + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, epoch_losses, marker='o', label='Loss')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(epochs)
        plt.legend()
        plt.grid(True)
        plt.show()

    def execute(self, config: Config, context: PipelineContext):
        self._verify_data(config, context)

        self._plot_epoch_losses(context.data.training_metrics.epoch_losses)

        return context


# def visualize_predictions(model, test_loader, num_images=4, device='cpu'):
#     model.eval()  # Set the model to evaluation mode
#
#     dataiter = iter(test_loader)  # Get an iterator for the test set
#     images, labels = next(dataiter)  # Get a batch of images and labels
#
#     # Make predictions
#     outputs = model(images.to(device))  # Forward pass
#     _, predicted = torch.max(outputs, 1)  # Get predicted class (highest score)
#
#     # Display the images with predicted and true labels
#     show_images(images)
#     for i in range(num_images):
#         # Display predicted and true labels
#         print(f"Predicted: {predicted[i].item()}, True label: {labels[i].item()}")
