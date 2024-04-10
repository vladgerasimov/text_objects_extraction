from abc import ABC, abstractmethod

from src.core.models import ExtractedObjectsDict


class BaseObjectsExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> ExtractedObjectsDict: ...
