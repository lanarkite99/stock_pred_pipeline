import torch.nn as nn

from src.config import Config

class StockLSTM(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, num_layers=None,dropout=None,pred_len=None):
        super().__init__()
        config = Config()
        input_size = input_size or config.input_size
        hidden_size = hidden_size or config.hidden_size
        num_layers = num_layers or config.num_layers
        dropout = dropout if dropout is not None else config.dropout
        pred_len = pred_len or config.pred_len
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=dropout)
        self.fc1=nn.Linear(hidden_size,pred_len*input_size)
        self.pred_len=pred_len
        self.input_size=input_size

    def forward(self,x):
        lstm_out,_=self.lstm(x)
        lstm_out=lstm_out[:,-1,:]
        out=self.fc1(lstm_out)
        out=out.view(-1,self.pred_len,self.input_size)
        return out
    
    
    
    
    
    
    
