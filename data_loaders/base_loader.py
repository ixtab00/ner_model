from abc import ABC, abstractmethod

class DataLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load(self, path: str, verbose: bool):
        pass