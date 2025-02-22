import torch
from torch_geometric.nn import GATConv, GraphConv, SAGEConv, TransformerConv, GCNConv
import torch.nn as nn
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, in_feats = 384, hid_feats = 128, out_feats = 64, n_class = 2, edge_dim = 1):
        super(GAT, self).__init__()
        self.fc1 = nn.Sequential(
            nn.LayerNorm(in_feats),
            nn.Linear(in_feats, hid_feats)
        )
        self.conv1 = GATConv(in_channels=hid_feats, out_channels=hid_feats)
        self.conv2 = GATConv(in_channels=hid_feats, out_channels= out_feats)
        self.fc2 = nn.Sequential(
            nn.LayerNorm(out_feats),
            nn.Linear(out_feats, n_class)
        )
        
        self.fc3 = nn.Sequential(                                                                                                                               
            nn.LayerNorm(in_feats),                                                                                                                            
            nn.Linear(in_feats, hid_feats) ,
            nn.LayerNorm(hid_feats),
            nn.Linear(hid_feats, out_feats),
            nn.LayerNorm(out_feats),
            nn.Linear(out_feats, n_class)                                                                                    
        )
        
        self.apply(self._initialize_weights)
        
    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            print(f'layer {m} initialized')
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x, edge_index, edge_attr):
        # inputs are features of nodes
        x = self.fc1(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
#         print(h.size())
#         output=F.relu(self.fc2(h))
        output = self.fc2(x)
        # output = self.fc3(x)
        return output
    
class GNN(torch.nn.Module):
    def __init__(self, in_feats = 384, hid_feats = 128, out_feats = 64, n_class = 2, edge_dim = 1):
    # def __init__(self, in_feats = 442, hid_feats = 128, out_feats = 64, n_class = 2, edge_dim = 1):
    # def __init__(self, in_feats = 20, hid_feats = 128, out_feats = 64, n_class = 2, edge_dim = 1):
        super(GNN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.LayerNorm(in_feats),
            nn.Linear(in_feats, hid_feats)
        )
        # self.conv1 = SAGEConv(in_channels=hid_feats, out_channels=hid_feats)
        # self.conv2 = SAGEConv(in_channels=hid_feats, out_channels= out_feats)

        self.conv1 = GraphConv(in_channels=hid_feats, out_channels=hid_feats)
        self.conv2 = GraphConv(in_channels=hid_feats, out_channels= out_feats)

        # self.conv1 = TransformerConv(in_channels=hid_feats, out_channels=hid_feats)
        # self.conv2 = TransformerConv(in_channels=hid_feats, out_channels= out_feats)

        # self.conv1 = GCNConv(in_channels=hid_feats, out_channels=hid_feats)
        # self.conv2 = GCNConv(in_channels=hid_feats, out_channels= out_feats)
        self.fc2 = nn.Sequential(
            nn.LayerNorm(out_feats),
            nn.Linear(out_feats, n_class)
        )

        self.apply(self._initialize_weights)
        
    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            print(f'layer {m} initialized')
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x, edge_index, edge_attr):
        # inputs are features of nodes
        x = self.fc1(x)
        # print(x.shape)
        # print(edge_attr.shape)
        # print(edge_index.shape)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
#         print(h.size())
#         output=F.relu(self.fc2(h))
        output = self.fc2(x)
        # output = self.fc3(x)
        return output