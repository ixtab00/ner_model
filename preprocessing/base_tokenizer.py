from abc import ABC, abstractmethod
from typing import Any

class Tokrnizer(ABC):
    def __init__(self, max_vocab_size: int):
        self.max_vocab_size = max_vocab_size

    @abstractmethod
    def tokenize(self, dataset: Any) -> Any:
        return None

    @abstractmethod
    def serealize(self, path: str) -> None:
        return None

    @abstractmethod
    def detokenize(self, data: Any) -> str:
        return None
    
    @abstractmethod
    def export_tables(self, path: str) -> None:
        return None