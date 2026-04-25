import os
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    context_len: int = 60
    pred_len: int = 5
    features: List[str] = field(default_factory=lambda: ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD"])
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 32
    parent_ticker: str = "^NSEI"
    child_tickers: List[str] = field(default_factory=lambda: ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFC.NS", "HDFCBANK.NS"])
    start_date: str = "2010-01-01"
    parent_epochs: int = 50
    child_epochs: int = 10
    transfer_strategy: str = "freeze"
    learning_rate: float = 0.001
    fine_tune_lr: float = 0.0001
    artifact_dir: str = field(default_factory=lambda: os.getenv("ARTIFACT_DIR", "outputs"))
    feast_online_store: str = field(default_factory=lambda: os.getenv("FEAST_ONLINE_STORE", "sqlite"))
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "127.0.0.1"))
    redis_port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    parent_dir: str = field(init=False)
    workdir: str = field(init=False)

    def __post_init__(self):
        self.workdir = os.getenv("WORKDIR", self.artifact_dir)
        self.parent_dir = os.getenv("PARENT_DIR", os.path.join(self.workdir, "parent"))

    @property
    def input_size(self) -> int:
        return len(self.features)
