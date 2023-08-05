import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, aggr

class GCN(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 32)
        self.conv4 = GCNConv(32, 16)
        # self.pool = aggr.MaxAggregation()
        # self.aggr = aggr.MLPAggregation(16, 128, 16, num_layers=1)
        self.fc1 = nn.Linear(16*16, 128)
        self.fc2 = nn.Linear(128, 256)

        self.edge_idx = torch.tensor([[0,0,0,0,1,1,2,2,3,4,4,4,4,5,5,5,6,6,7,8,8,8,8,9,9,10,10,11,12,12,12,13,13,14,14,15],
                                      [1,4,5,8,0,2,1,3,2,0,5,8,12,0,4,6,5,7,6,0,4,9,12,8,10,9,11,10,4,8,13,12,14,13,15,14]], dtype=torch.int64)

    def forward(self, obs):
        edge_idx = self.edge_idx.to(obs.device)
        x = self.conv1(obs, edge_idx).relu()
        x = self.conv2(x, edge_idx).relu()
        x = self.conv3(x, edge_idx).relu()
        x = self.conv4(x, edge_idx).relu()
        # x = self.pool(x).squeeze()
        # x = self.aggr(x)
        x = x.view(-1, x.size(1)*x.size(2))
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.elu(x)

        return x
