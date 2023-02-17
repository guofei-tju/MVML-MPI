from math import sqrt

import numpy as np
import torch
import torch.nn as nn
from dgllife.model.gnn.gat import GAT
from dgl.nn.pytorch import Set2Set
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import tokens_struct
from torch_geometric.nn import GraphNorm


class MVP(nn.Module):
    def __init__(self, num_classes, in_feats=64, hidden_feats=None, num_step_set2set=6,
                 num_layer_set2set=3, rnn_embed_dim=64, blstm_dim=128, blstm_layers=2, fp_2_dim=128, num_heads=4,
                 dropout=0.2, device='cpu'):
        super(MVP, self).__init__()
        self.device = device
        self.vocab = tokens_struct()
        if hidden_feats is None:
            hidden_feats = [64, 64]
        self.final_hidden_feats = hidden_feats[-1]
        self.norm_layer_module = nn.LayerNorm(self.final_hidden_feats).to(device)
        self.gnn = GNNModule(in_feats, hidden_feats, dropout, num_step_set2set, num_layer_set2set)
        self.rnn = RNNModule(self.vocab, rnn_embed_dim, blstm_dim, blstm_layers, self.final_hidden_feats, dropout,
                             bidirectional=True, device=device)
        self.fp_mlp = FPNModule(fp_2_dim, self.final_hidden_feats)
        self.conv = nn.Sequential(nn.Conv2d(12, 12, kernel_size=3), nn.ReLU(),
                                  nn.Dropout(dropout))
        dim_k = self.final_hidden_feats * num_heads
        dim_v = self.final_hidden_feats * num_heads
        dim_in = self.final_hidden_feats
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)
        self.norm_layer = nn.LayerNorm((self.final_hidden_feats - 2) * self.num_heads).to(device)
        self.mlp = nn.Sequential(
            nn.Linear((self.final_hidden_feats - 2) * self.num_heads, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, smiles, graphs, atom_feats, fp_t):
        # get graph input
        batch_size = smiles['smiles'].size(0)
        smiles_x = self.norm_layer_module(self.rnn(smiles)).view(batch_size, 1, -1)
        graph_x = self.norm_layer_module(self.gnn(graphs, atom_feats)).view(batch_size, 1, -1)
        fp_x = self.norm_layer_module(self.fp_mlp(fp_t)).view(batch_size, 1, -1)
        in_tensor = torch.cat([smiles_x, graph_x, fp_x], dim=1)
        batch, n, dim_in = in_tensor.shape
        assert dim_in == self.dim_in
        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head
        q = self.linear_q(in_tensor).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(in_tensor).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(in_tensor).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)
        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
        att = torch.matmul(dist, v)
        out = self.conv(att).view(batch_size, -1)
        out = self.mlp(out)
        return out

    def predict(self, smiles, graphs, atom_feats, fp_t):
        return self.sigmoid(self.forward(smiles, graphs, atom_feats, fp_t))


class GNNModule(nn.Module):
    def __init__(self, in_feats=64, hidden_feats=None, dropout=0.2, num_step_set2set=6,
                 num_layer_set2set=3):
        super(GNNModule, self).__init__()
        self.conv = GAT(in_feats, hidden_feats)
        self.readout = Set2Set(input_dim=hidden_feats[-1],
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.norm = GraphNorm(hidden_feats[-1] * 2)
        self.fc = nn.Sequential(nn.Linear(hidden_feats[-1] * 2, hidden_feats[-1]), nn.ReLU(),
                                nn.Dropout(p=dropout))

    def forward(self, graphs, atom_feats):
        # get graph input
        node_x = self.conv(graphs, atom_feats)
        graph_x = self.readout(graphs, node_x)
        out = self.norm(graph_x)
        out = self.fc(out)
        return out


class RNNModule(nn.Module):
    def __init__(self, vocab, embed_dim, blstm_dim, num_layers, out_dim=2, dropout=0.2, bidirectional=True,
                 device='cpu'):
        super(RNNModule, self).__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.blstm_dim = blstm_dim
        self.hidden_size = blstm_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.bidirectional = bidirectional
        self.device = device
        self.num_dir = 1
        if self.bidirectional:
            self.num_dir += 1
        self.embeddings = nn.Embedding(vocab.tokens_length, self.embed_dim, padding_idx=vocab.pad)
        self.rnn = nn.LSTM(self.embed_dim, self.blstm_dim, num_layers=self.num_layers,
                           bidirectional=self.bidirectional, dropout=dropout,
                           batch_first=True)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Sequential(nn.Linear(self.blstm_dim, self.out_dim), nn.ReLU(), nn.Dropout(p=dropout))
        if self.bidirectional:
            self.norm_layer = nn.LayerNorm(2 * self.blstm_dim).to(device)
            self.fc = nn.Sequential(nn.Linear(2 * self.blstm_dim, self.out_dim), nn.ReLU(),
                                    nn.Dropout(p=dropout))

    def forward(self, batch):
        smiles = batch["smiles"]
        seq_lens = batch['seq_len']
        x = self.embeddings(smiles.long())
        packed_input = pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        packed_output, states = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out_forward = output[range(len(output)), np.array(seq_lens) - 1, :self.blstm_dim]
        out_reverse = output[:, 0, self.blstm_dim:]
        text_fea = torch.cat((out_forward, out_reverse), 1)
        out = self.fc(text_fea)
        return out


class FPNModule(nn.Module):
    def __init__(self, fp_2_dim, out_feats, dropout=0.2):
        super(FPNModule, self).__init__()
        self.fp_2_dim = fp_2_dim
        self.dropout_fpn = dropout
        self.out_feats = out_feats
        self.fp_dim = 2513
        self.fc1 = nn.Linear(self.fp_dim, self.fp_2_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.out_feats)
        self.dropout = nn.Dropout(p=self.dropout_fpn)

    def forward(self, smiles):
        fpn_out = self.fc1(smiles)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        return fpn_out
