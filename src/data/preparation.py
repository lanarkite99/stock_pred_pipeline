import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from src.data.ingestion import fetch_ohlcv
import pandas as pd
from src.config import Config
from src.exception import PipelineError

class StockDataset(Dataset):
    def __init__(self, data: pd.DataFrame, scaler: StandardScaler, context_len: int = 60, pred_len: int = 5):
        self.context_len=context_len
        self.pred_len=pred_len
        try:
            val=scaler.transform(data[Config().features].values)
            self.res=[]
            for t in range(self.context_len,len(data)-self.pred_len):
                past=val[t-self.context_len:t]
                fut=val[t:t+self.pred_len]
                if past.shape==(self.context_len,len(Config().features)) and fut.shape==(self.pred_len,len(Config().features)):
                    self.res.append((past,fut))
                else:
                    print(f"Invalid data shape at time {t}")
            if not self.res:
                raise PipelineError("No valid data found")
        except Exception as e:
            raise PipelineError("failed to create dataset") from e

    def __len__(self):
        return len(self.res)

    def __getitem__(self, idx):
        past,fut=self.res[idx]
        return torch.FloatTensor(past),torch.FloatTensor(fut)
    
    
    
    
    
    
