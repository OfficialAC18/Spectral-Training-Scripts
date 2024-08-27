import torch
import torch.nn as nn
import torch.nn.functional as F


class ExVivoLSTM(nn.Module):
    def __init__(self,
                  hidden_size = 500,
                  num_layers = 3,
                  dp = 0.2,
                  l1=120,
                  l2=64,
                  init_input = torch.rand(1,900)):
        super(ExVivoLSTM, self).__init__()
        # We need to find the length of the input for the LSTM
        self.lstm_shape = init_input.shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dp = dp
        self.lstm1 = nn.LSTM(input_size = self.lstm_shape[-1],
                             hidden_size = self.hidden_size,
                             num_layers=self.num_layers,
                             batch_first=True)

        self.dropout = nn.Dropout1d(p=self.dp)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(l1)
        self.fc2 = nn.Linear(l1,l2)
        self.head = nn.Linear(l2,2)

    def forward(self, x):
        x, _ = self.lstm1(x, (torch.rand(self.num_layers,x.shape[0], self.hidden_size, device='cuda:0'),
                              torch.rand(self.num_layers,x.shape[0], self.hidden_size, device='cuda:0')))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.head(x)
        return x
    
 

