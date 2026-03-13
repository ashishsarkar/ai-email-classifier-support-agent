"""Base service class with common initialization and cleanup."""

from abc import ABC, abstractmethod
from loguru import logger


class BaseService(ABC):
    """Abstract base class for all services."""

    async def initialize(self) -> None:
        """Initialize service resources. Override in subclasses if needed."""
        logger.info(f"Initializing {self.__class__.__name__}")

    async def cleanup(self) -> None:
        """Cleanup service resources. Override in subclasses if needed."""
        logger.info(f"Cleaning up {self.__class__.__name__}")

    async def health_check(self) -> dict:
        """Check service health. Override in subclasses for specific checks."""
        return {"status": "healthy", "service": self.__class__.__name__}
