from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineData:
    """Encapsulates data transferred between pipeline stages.

        --- Training Stage ---
        final_loss : Quantitative measure of how well the model fit the training data.
        training_time : Total time taken for training.

    --- Data Loader Stage ---
    data_loader: Stores dataset loaders and class information.

    --- Model Loader Stage ---
    model: The loaded model instance.

    --- Training Stage ---
    training_metrics: Stores training performance metrics.

    --- Evaluation Stage ---
    evaluation_metrics: Stores evaluation results.
    """

    @dataclass
    class DataLoader:
        """Holds dataset loaders for training, testing, and validation."""
        class_names: Optional[list[str]] = None
        train_loader: Optional[object] = None
        test_loader: Optional[object] = None
        validation_loader: Optional[object] = None

    @dataclass
    class TrainingMetrics:
        """Tracks training performance metrics."""
        epoch_losses: list[float] = field(default_factory=list)
        final_loss: float = 0.0
        training_time: float = 0.0

    @dataclass
    class EvaluationMetrics:
        """Stores evaluation metrics"""
        accuracy: float = 0.0

    # Data Loader stage
    data_loader: Optional[DataLoader] = None

    # Model Loader stage
    model: Optional[object] = None

    # Training stage
    training_metrics: Optional[TrainingMetrics] = None

    # Evaluation stage
    evaluation_metrics: Optional[EvaluationMetrics] = None


# TODO : Make it more secure by limiting what each stage can read/write ?
class PipelineContext:
    """Context object containing runtime data used by the different stages.

    Stores shared data across pipeline stages, enabling controlled data flow and tracking intermediate results.
    """

    def __init__(self, data: PipelineData = None):
        self.data = data if data is not None else PipelineData()
