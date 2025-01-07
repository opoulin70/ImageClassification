import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineData:
    """Data transfer object used to pass data between stages.

        --- Training Stage ---
        final_loss : Quantitative measure of how well the model fit the training data.
        training_time : Total time taken for training.
    """
    @dataclass
    class DataLoader:
        class_names: Optional[list[str]] = None
        train_loader: Optional[object] = None
        test_loader: Optional[object] = None
        validation_loader: Optional[object] = None

    @dataclass
    class TrainingMetrics:
        epoch_losses: list[float] = field(default_factory=list)
        final_loss: float = 0.0
        training_time: float = 0.0

    @dataclass
    class EvaluationMetrics:
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
    """Context object containing runtime data used by the different stages."""

    def __init__(self, data: PipelineData = None):
        self.data = data if data is not None else PipelineData()
