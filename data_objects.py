from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, List, Tuple
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from os import cpu_count

class Stage(Enum):
    TRAIN = auto()
    DEV = auto()
    VAL = auto()
    TEST = auto()

cpus = cpu_count()
class DataHolder:
    def __init__(self, name: str, dataloader: DataLoader, size: int, stage: Stage, dataset:Dataset = None) -> None:
        self.name=name
        self.dataloader=dataloader
        self.size=size
        self.stage=stage
        self.dataset=dataset

    @classmethod
    def from_dataset(cls, name:str, dataset:Dataset, stage:Stage, batch_size=32, shuffle=False) -> None:
        dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=cpus)
        size = len(dataset)

        return cls(name, dataloader, size, stage, dataset)
        
    def get_input_label_preds(self, model) -> List[Tuple[Any, Any, Any]]:
        return [(input, label, int(pred)) for (input, label), pred in zip(self.dataset, model(self))]

       