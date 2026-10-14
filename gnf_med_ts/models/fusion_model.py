"""整体模型: 双空间特征提取 + 可学习头融合 + 多目标解码 (论文 Model Construction)

点特征空间:  GCN 提取时间节点嵌入  F_p
边权重空间:  门控卷积注意力提取  F_e(按时间保留)
融合:        F_total = H_fp * F_p + H_fe * F_e    (H_fp, H_fe 为可学习权重头)

分类任务 (死亡/临床表型): 对 F_total 按时间分解, 前馈网络映射, BCE 损失
回归任务 (住院时长):      取最后时间点的 F_total, 前馈网络映射, L1 损失

消融 (ablate): 'point' -> 仅边权重空间; 'edge' -> 仅点特征空间
"""
import torch
import torch.nn as nn

from .gcn import GCN
from .gated_attention import EdgeGatedAttention


class GNFFusionModel(nn.Module):
    def __init__(self, cfg, node_feat_dim, n_cls_tasks, n_reg_tasks=1):
        super().__init__()
        self.cfg = cfg
        self.gcn = GCN(node_feat_dim, cfg.gcn_hidden, cfg.gcn_layers,
                       cfg.gcn_dropout)
        self.edge_net = EdgeGatedAttention(cfg.edge_layers, cfg.edge_conv_channels,
                                           cfg.n_heads, cfg.attn_dropout)
        self.edge_time_proj = nn.Linear(cfg.edge_conv_channels, cfg.gcn_hidden)

        # ---- 可学习头: 对点/边特征做时间维度的加权重构 ----
        self.H_fp = nn.Linear(cfg.gcn_hidden, cfg.gcn_hidden, bias=False)
        self.H_fe = nn.Linear(cfg.gcn_hidden, cfg.gcn_hidden, bias=False)
        # 融合门控 (可学习标量头, 对应论文中的 H_fp / H_fe 加权)
        self.gate_fp = nn.Parameter(torch.tensor(0.5))
        self.gate_fe = nn.Parameter(torch.tensor(0.5))

        # ---- 分类解码器: 时间序列重构 -> FFN -> 各目标 logits ----
        self.cls_head = nn.Sequential(
            nn.Linear(cfg.gcn_hidden, cfg.gcn_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.gcn_dropout),
            nn.Linear(cfg.gcn_hidden, n_cls_tasks))

        # ---- 回归解码器: 最后时间点特征 -> FFN ----
        self.reg_head = nn.Sequential(
            nn.Linear(cfg.gcn_hidden, cfg.gcn_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.gcn_dropout),
            nn.Linear(cfg.gcn_hidden, n_reg_tasks))

    def forward(self, node_feat, adj, we, we_mask, n_time, last_idx,
                ablate=None):
        """
        node_feat: (B, N, D)  adj: (B, N, N)
        we/we_mask: (B, 1, T, F)   n_time: (B,)  last_idx: (B,)
        ablate: None | 'point' | 'edge'  (消融实验)
        返回 cls_logits (B, n_cls_tasks), reg_out (B, n_reg_tasks),
             time_weights (B, T)  # 可解释性: 各时间点权重
        """
        B = node_feat.size(0)
        T_max = we.size(2)

        # ---- 点特征空间: 取时间节点部分 (前 n_time 个节点) ----
        node_emb = self.gcn(node_feat, adj)                     # (B, N, H)
        idx = torch.arange(T_max, device=node_feat.device)
        time_mask = (idx.unsqueeze(0) < n_time.unsqueeze(1))    # (B, T_max)
        node_emb_t = node_emb[:, :T_max] * time_mask.unsqueeze(-1)

        # ---- 边权重空间: (B, T, C) -> 投影对齐 ----
        edge_emb_t = self.edge_time_proj(self.edge_net(we, we_mask))
        edge_emb_t = edge_emb_t * time_mask.unsqueeze(-1)

        # 消融: 屏蔽对应特征空间
        if ablate == 'point':
            node_emb_t = node_emb_t * 0.0
        if ablate == 'edge':
            edge_emb_t = edge_emb_t * 0.0

        # ---- 可学习头的时间加权重构 (softmax 归一化, 可解释) ----
        w_fp = torch.softmax(self.H_fp(node_emb_t).mean(-1).masked_fill(
            ~time_mask, float('-inf')), dim=1)                  # (B, T)
        w_fe = torch.softmax(self.H_fe(edge_emb_t).mean(-1).masked_fill(
            ~time_mask, float('-inf')), dim=1)

        # ---- 融合: F_total = H_fp*F_p + H_fe*F_e (分类: 全局上下文) ----
        g_fp, g_fe = torch.sigmoid(self.gate_fp), torch.sigmoid(self.gate_fe)
        ctx_p = (w_fp.unsqueeze(-1) * node_emb_t).sum(1)        # (B, H)
        ctx_e = (w_fe.unsqueeze(-1) * edge_emb_t).sum(1)
        cls_total = g_fp * ctx_p + g_fe * ctx_e
        cls_logits = self.cls_head(cls_total)

        # ---- 回归: 最后时间点特征 (论文: 关键考虑最终时刻特征向量) ----
        li = last_idx.clamp(max=T_max - 1)
        reg_p = node_emb_t[torch.arange(B, device=node_emb.device), li]
        reg_e = edge_emb_t[torch.arange(B, device=edge_emb.device), li]
        reg_total = g_fp * reg_p + g_fe * reg_e
        reg_out = self.reg_head(reg_total)

        return cls_logits, reg_out, w_fp
