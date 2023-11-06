from torch_geometric.nn import GCNConv, VGAE
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn.models import InnerProductDecoder


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(GCNConv(in_channels=int(in_channels),out_channels=int(out_channels), dropout=0.5))
        self.num_layers = 2
        # hidden layers
        for l in range(1, self.num_layers):
            self.encoder.append(GCNConv(in_channels=int(in_channels), out_channels=int(out_channels),  dropout=0.5))
        #
        self.gcn_shared = GCNConv(in_channels=int(in_channels),out_channels=int(in_channels))
        self.gcn_mu = GCNConv(in_channels=int(in_channels),out_channels=int(out_channels))
        self.gcn_logvar = GCNConv(in_channels=int(in_channels),out_channels=int(out_channels))

    def forward(self, edge_emb_eq1, edge_index):
        for l in range(self.num_layers):
            edge_emb_eq1 = self.encoder[l](edge_emb_eq1, edge_index).flatten(1)

        x = F.relu(self.gcn_shared(edge_emb_eq1, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar


class VGraphAE(VGAE):
    def __init__(self, device, sym_dict={'log':0, 'Mul':1, 'exp':2, 'Add':3, 'Symbol':4, 'Pow':5, 'cos':6, 'Integer':7, 'sin':8}, input_dim=384, hidden_dim=384, feat_drop=0.5, heads=8, num_layers=2, args=None):
        super(VGraphAE, self).__init__(
            encoder=GCNEncoder(in_channels=int(input_dim), hidden_channels=None, out_channels=int(hidden_dim)).to(device),
            decoder=InnerProductDecoder().to(device))

        self.device = device
        self.activation = nn.LeakyReLU()
        self.num_layers = num_layers
        self.dim = hidden_dim

        # symbol
        self.symbol_dict = sym_dict
        if args.exp == 'recon_symbol':
            # symbolic representation
            self.symbol = torch.nn.Parameter(torch.tensor(np.random.uniform(-1,1, (len(self.symbol_dict), self.dim)), dtype=torch.float, requires_grad=True, device=self.device))
        else:
            self.embedding = nn.Embedding(sym_dict.size, self.dim)
            self.embedding.weight.data.uniform_(-1, 1)

        self.args = args

    def generate_edge_emb(self, node_l):
        if self.args.exp == 'recon_symbol':
            edge_emb = []
            for node in node_l:
                if node in self.symbol_dict:
                    edge_emb.append(self.symbol[self.symbol_dict[node]])
                else:
                    edge_emb.append(torch.tensor(np.random.uniform(-1,1, (self.dim)), dtype=torch.float, requires_grad=True, device=self.device))
            edge_emb = torch.stack(edge_emb, dim=0).to(self.device)
        elif self.args.exp == 'recon_natural':
            node_index = [self.symbol_dict.word2idx[w.strip()] if w.strip() in self.symbol_dict.word2idx else self.symbol_dict.unk for w in node_l]
            edge_emb = self.embedding(torch.tensor(node_index, device=self.device))
        else:
            edge_emb = None
            exit('only support recon_symbol and recon_natural')
        return edge_emb

    def forward(self, equation1):
        edge_emb_eq1 = self.generate_edge_emb(equation1["node_list"]).to(self.device)
        edge_index = equation1["edge_index"].to(self.device)
        z = self.encode(edge_emb_eq1, edge_index) # (node_num by hidden_size)
        rec_loss = -torch.log(self.decoder(z, edge_index, sigmoid=True) + 1e-15).mean()
        kl_loss = 1 / edge_emb_eq1.size(0) * self.kl_loss()
        loss = rec_loss + 0*kl_loss
        return loss, rec_loss, kl_loss

    def loss_and_z(self, equation1):
        edge_emb_eq1 = self.generate_edge_emb(equation1["node_list"]).to(self.device)
        edge_index = torch.tensor(equation1["edge_index"]).to(self.device)
        z = self.encode(edge_emb_eq1, edge_index) # (node_num by hidden_size)
        # print(self.decoder.forward_all(z).shape)
        rec_loss = -torch.log(self.decoder(z, edge_index, sigmoid=True) + 1e-15).mean()
        kl_loss = 1 / edge_emb_eq1.size(0) * self.kl_loss()
        loss = rec_loss + 0*kl_loss
        return loss, rec_loss, kl_loss, z