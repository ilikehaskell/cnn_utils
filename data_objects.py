from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, List, Tuple
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class Stage(Enum):
    TRAIN = auto()
    DEV = auto()
    VAL = auto()
    TEST = auto()

class DataHolder:
    def __init__(self, name: str, dataloader: DataLoader, size: int, stage: Stage, dataset:Dataset = None) -> None:
        self.name=name
        self.dataloader=dataloader
        self.size=size
        self.stage=stage
        self.dataset=dataset

    @classmethod
    def from_dataset(cls, name:str, dataset:Dataset, stage:Stage, batch_size=32, shuffle=False) -> None:
        dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        size = len(dataset)

        return cls(name, dataloader, size, stage, dataset)
        
    def get_input_label_preds(self, model) -> List[Tuple[Any, Any, Any]]:
        return [(input, label, int(pred)) for (input, label), pred in zip(self.dataset, model(self))]

       