"""Contains abstract base class used for other pipeline stages."""

from abc import ABC, abstractmethod
from source.pipeline_context import PipelineContext
from source.config import Config


class BasePipelineStage(ABC):
    """Abstract base class for all pipeline stages."""
    @abstractmethod
    def execute(self, config: Config, context: PipelineContext) -> PipelineContext:
        """Execute the stage logic."""
        pass

    @abstractmethod
    def _verify_data(self, config: Config, context: PipelineContext) -> None:
        """Ensure the data is valid before executing the stage."""
        pass
