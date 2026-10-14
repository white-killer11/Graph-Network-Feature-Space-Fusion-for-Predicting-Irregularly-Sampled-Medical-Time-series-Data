"""
Configuration for Graph Network Feature-space Fusion (GNF) model
论文: Graph Network Feature Space Fusion for Predicting Irregularly Sampled
      Medical Time-Series Data (JMIR Med Inform 2026;14:e81145)
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ---------- 图结构 ----------
    n_features: int = 42            # F_m: 特征通道数 (论文中为 42 个临床特征)
    max_time_nodes: int = 128       # T_m 上限: 时间窗口内最多保留的不同测量时间点数
    n_kl_bins: int = 32             # KL 散度离散分布的 bin 数 M
    time_window_hours: float = 24.0 # 计算时间范围 Tm(小时), 可为 6/12/24/36/48

    # ---------- 节点/边特征 ----------
    node_dim: int = 128             # 节点特征投影维度 D
    edge_hidden: int = 64           # 边权重空间卷积通道数

    # ---------- GCN (点特征空间) ----------
    gcn_hidden: int = 128
    gcn_layers: int = 3             # L 层 GCN
    gcn_dropout: float = 0.1

    # ---------- 门控卷积注意力 (边权重空间) ----------
    edge_conv_channels: int = 64
    edge_layers: int = 3            # L 层 门控卷积注意力
    n_heads: int = 4                # 多头注意力头数
    attn_dropout: float = 0.1

    # ---------- 任务 ----------
    n_phenotypes: int = 20          # 多表型分类目标数 (可按需调整)
    phenotype_names: List[str] = field(default_factory=list)

    # ---------- 训练 ----------
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    seed: int = 42
