import torch.nn as nn


class FFNN(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p):
        super(FFNN, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x