import torch
import torch.nn as nn
import torch.nn.functional as F

class ExVivoConv(nn.Module):
    def __init__(self, nc1 = 128,
                  k1 = 3,
                  nc2 = 64,
                  k2 = 3,
                  nc3 = 32,
                  k3 = 5,
                  nc4 = 32,
                  k4 = 5,
                  dp = 0.5,
                  l1=120,
                  l2=64,
                  init_input = torch.rand(1,900)):
        super(ExVivoConv, self).__init__()
        self.conv1 = nn.Conv1d(1,nc1,k1,)
        self.pool1 = nn.MaxPool1d(k1)
        self.bn1 = nn.BatchNorm1d(nc1)
        self.conv2 = nn.Conv1d(nc1, nc2, k2)
        self.pool2 = nn.MaxPool1d(k2)
        self.bn2 = nn.BatchNorm1d(nc2)
        self.conv3 = nn.Conv1d(nc2, nc3, k3)
        self.pool3 = nn.MaxPool1d(k3)
        self.bn3 = nn.BatchNorm1d(nc3)
        self.conv4 = nn.Conv1d(nc3, nc4, k4)
        self.pool4 = nn.MaxPool1d(k4)
        self.bn4 = nn.BatchNorm1d(nc4)
        self.dropout = nn.Dropout1d(p=dp)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(l1)
        self.fc2 = nn.Linear(l1,l2)
        self.head = nn.Linear(l2,5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.relu(x))
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.pool2(F.relu(x))
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.pool3(F.relu(x))
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.pool4(F.relu(x))
        x = self.bn4(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.head(x)
        return x
    

