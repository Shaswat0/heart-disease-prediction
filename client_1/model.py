import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=32):
        super(GNNModel, self).__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 1)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)  # Global pooling
        x = self.fc(x)
        return torch.sigmoid(x)
