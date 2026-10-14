"""点特征空间: 多层图卷积网络 GCN (论文 Model Construction)

H^(1) = sigma(A_w X_f W_0)
H^(l) = sigma(A_w H^(l-1) W_{l-1})
F_p   = H^(L)
邻接矩阵 A_w 使用对称归一化 D^{-1/2}(A+I)D^{-1/2} 保证训练稳定。
"""
import torch
import torch.nn as nn


class GraphConv(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, adj_norm):
        # x: (B, N, C), adj_norm: (B, N, N)
        h = torch.bmm(adj_norm, x)          # 邻居聚合: A_w H
        h = self.lin(h)                     # W H
        h = self.norm(h + x)                # 残差 + 归一化
        return self.drop(self.act(h))


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [GraphConv(hidden_dim, dropout) for _ in range(n_layers)])

    @staticmethod
    def normalize_adj(adj):
        """adj: (B, N, N) 含自环 -> 对称归一化 D^{-1/2} A D^{-1/2}"""
        deg = adj.sum(dim=-1).clamp(min=1e-6)
        d_inv_sqrt = deg.pow(-0.5)
        d_mat = torch.diag_embed(d_inv_sqrt)
        return torch.bmm(torch.bmm(d_mat, adj), d_mat)

    def forward(self, node_feat, adj):
        """
        node_feat: (B, N, D)  adj: (B, N, N)
        返回节点嵌入 F_p: (B, N, hidden)
        """
        h = self.in_proj(node_feat)
        adj_norm = self.normalize_adj(adj)
        for conv in self.convs:
            h = conv(h, adj_norm)
        return h
