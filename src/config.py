from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ClassConfig:
    classes: List[str] = ("normal", "smoke_fire", "oil_leak", "conveyor_jam")


@dataclass(frozen=True)
class TrainConfig:
    image_size: int = 224
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 10
    weight_decay: float = 1e-4
    num_workers: int = 2
    data_root: Path = Path("data/processed")
    model_out: Path = Path("models/mobilenet_anomaly.pt")

