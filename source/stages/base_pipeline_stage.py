"""Contains abstract base class used for other pipeline stages."""

from abc import ABC, abstractmethod
from source.pipeline_context import PipelineContext
from source.config import Config


class BasePipelineStage(ABC):
    """Abstract base class for all pipeline stages."""
    @abstractmethod
    def _verify_data(self, config: Config, context: PipelineContext):
        """Ensure the data is valid before executing the stage."""
        pass

    @abstractmethod
    def execute(self, config: Config, context: PipelineContext):
        """Execute the stage logic."""
        pass
