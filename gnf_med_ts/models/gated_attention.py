"""边权重空间: 门控卷积注意力网络 (论文 Model Construction)

第 1 层:
  W_EF = Sigmoid(h_1 * W_E + b_1)              # 遗忘门, 逐元素仿射
  C_EF = Conv2d(W_EF),  C_E = Conv2d(W_E)
  A_E(1) = Attention(C_EF, C_E, C_E)           # 多头注意力, 查询来自遗忘特征

第 l 层 (l > 1):
  C_AE(l) = Conv2d( Sigmoid(h_l * AvgPool(A_E(l-1))) )
  A_E(l) = Attention(C_AE(l), A_E(l-1), C_E)

输出: F_e = AveragePooling(A_E(L))  (保留时间维, 供分类任务按时间重构)
"""
import torch
import torch.nn as nn


class GatedConvAttentionLayer(nn.Module):
    def __init__(self, channels, n_heads, first_layer, dropout=0.1):
        super().__init__()
        self.first = first_layer
        # 遗忘门权重 (第一层对 W_E 做仿射; 后续层对平均池化后的特征做仿射)
        self.gate_weight = nn.Parameter(torch.randn(1) * 0.02)
        self.gate_bias = nn.Parameter(torch.zeros(1))
        self.conv_q = nn.Conv2d(channels, channels, kernel_size=3,
                                padding=1, bias=True)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=3,
                                padding=1, bias=True)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=n_heads,
                                         dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, we, prev_attn=None):
        """
        we:      (B, 1, T, F) 归一化边权矩阵 W_E
        prev_attn: (B, T, F, C) 上一层注意力输出 A_E(l-1)
        返回: (B, T, F, C)
        """
        B, _, T, Fdim = we.shape
        C = self.conv_q.in_channels

        if self.first:
            gate = torch.sigmoid(self.gate_weight * we + self.gate_bias)   # W_EF
            q_in = gate.expand(B, C, T, Fdim)                              # 广播到 C 通道
            kv_in = we.expand(B, C, T, Fdim)
        else:
            pooled = prev_attn.mean(dim=(1, 2), keepdim=True)              # AvgPool -> (B,1,1,C)
            gate = torch.sigmoid(self.gate_weight * pooled + self.gate_bias)
            q_in = gate.permute(0, 3, 1, 2).expand(B, C, T, Fdim)
            kv_in = prev_attn                                              # A_E(l-1) 作 K/V

        c_q = self.conv_q(q_in)                                            # C_EF / C_AE
        c_v = self.conv_v(kv_in)                                           # C_E
        q = c_q.permute(0, 2, 3, 1).reshape(B, T * Fdim, C)
        k = c_v.permute(0, 2, 3, 1).reshape(B, T * Fdim, C)
        v = c_v.permute(0, 2, 3, 1).reshape(B, T * Fdim, C)
        attn, _ = self.mha(q, k, v)                                        # Attention(Q,K,V)
        out = attn.reshape(B, T, Fdim, C)
        if not self.first and prev_attn is not None:                       # 残差连接稳定深层训练
            out = out + prev_attn
        return self.drop(self.norm(out))


class EdgeGatedAttention(nn.Module):
    """L 层堆叠的边权重特征提取器"""
    def __init__(self, n_layers, channels, n_heads, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.layers = nn.ModuleList([
            GatedConvAttentionLayer(channels, n_heads, first_layer=(i == 0),
                                    dropout=dropout)
            for i in range(n_layers)])

    def forward(self, we, we_mask):
        """
        we: (B, 1, T, F), we_mask: (B, 1, T, F)
        返回按时间聚合前的特征: (B, T, C) (mask 平均池化掉特征维)
        """
        we = we * we_mask                                   # 缺失位置置零, 无噪声
        prev = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                prev = layer(we)
            else:
                prev = layer(we, prev_attn=prev)
        # prev: (B, T, F, C) -> 对特征维做掩码平均池化, 保留时间维 (供时间序列重构)
        B, T, Fdim, C = prev.shape
        m = we_mask.permute(0, 2, 3, 1)                     # (B, T, F, 1)
        pooled = (prev * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)
        return pooled                                       # (B, T, C)
