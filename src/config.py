from dataclasses import dataclass,field
from typing import List
import torch

@dataclass
class Config:
    device:str='cuda' if torch.cuda.is_available() else 'cpu'
    context_len:int=60
    pred_len:int=5
    features:List[str]=field(default_factory=lambda:['Open','High','Low','Close','Volume','RSI','MACD'])
    hidden_size:int=128
    num_layers:int=2
    dropout:float=0.2
    batch_size:int=32
    parent_ticker: str='^NSEI'
    child_tickers: List[str]=field(default_factory=lambda:['TCS.NS','INFY.NS','RELIANCE.NS','HDFC.NS','HDFCBANK.NS'])
    start_date:str='2010-01-01'
    parent_epochs:int=50
    child_epochs:int=10
    transfer_strategy:str='freeze'
    learning_rate:float=0.001
    fine_tune_lr:float=0.0001
    parent_dir:str='outputs/parent'
    workdir:str='outputs'
    feast_online_store:str='sqlite'
    redis_host:str='127.0.0.1'
    redis_port:int=6379
    redis_db:int=0

    @property
    def input_size(self)->int:
        return len(self.features)
