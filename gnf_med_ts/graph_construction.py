"""
图网络数据结构构建 (论文 Methods - Graph Network Construction)

每个测量行为三维: (t, f, v) = (测量时间, 测量特征, 测量值)
- 时间节点 Pt: 特征向量为该时间与其余所有测量时间的负距离
- 特征节点 Pf: 特征向量为该特征与其余特征分布之间的 KL 散度
- 连接边 Ew: 测量值 v 构成时间节点与特征节点之间的边权
  *边仅在实际测量存在时存在*, 从而避免缺失值插补带来的噪声

防数据泄漏: 预测时刻 Tm 的图仅使用严格早于 Tm 的测量计算 KL 散度与边权。
稀疏数据处理:
  n=0 (无测量)      -> 均匀分布
  n=1 (单次测量)    -> 以该值为中心的高斯分布
  n>=2             -> 高斯核 KDE 估计分布
"""
import numpy as np
from typing import Dict, List, Tuple

EPS = 1e-10


# ---------------------------------------------------------------- KL 散度 ----
def estimate_feature_distribution(values: np.ndarray,
                                  n_bins: int,
                                  value_range: Tuple[float, float]) -> np.ndarray:
    """将一个特征通道在窗口内的可用测量值估计为离散概率分布 (论文 4 页)。"""
    lo, hi = value_range
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = (hi - lo) / n_bins
    n = len(values)

    if n == 0:
        return np.ones(n_bins, dtype=np.float64) / n_bins
    if n == 1:
        sigma = max(bin_width, 1e-6)
        p = np.exp(-0.5 * ((centers - values[0]) / sigma) ** 2)
    else:
        sigma = float(np.std(values))
        if sigma < 1e-6:                       # 所有值相同 -> 退化高斯
            sigma = max(bin_width, 1e-6)
        p = np.zeros(n_bins, dtype=np.float64)
        for v in values:
            p += np.exp(-0.5 * ((centers - v) / sigma) ** 2)
        p /= n
    p = np.clip(p, EPS, None)
    return p / p.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p||q) = sum p*log(p/q), 含数值稳定处理。"""
    p = np.clip(p, EPS, 1.0); p = p / p.sum()
    q = np.clip(q, EPS, 1.0); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_feature_node_vectors(measurements: Dict[int, np.ndarray],
                                 n_bins: int,
                                 value_range: Tuple[float, float],
                                 n_features: int) -> np.ndarray:
    """
    构造所有特征节点的特征向量  S_pf = {F_1^pf, ..., F_Fm^pf}
    F_b^pf = [ KL(f_b, f_1), KL(f_b, f_2), ..., KL(f_b, f_Fm) ]
    measurements: {feature_idx: array of measured values within window}
    返回: (Fm, Fm) 的 KL 矩阵(行归一化, 映射到特征相关空间)
    """
    dists = [estimate_feature_distribution(measurements.get(f, np.array([])),
                                           n_bins, value_range)
             for f in range(n_features)]
    kl_mat = np.zeros((n_features, n_features), dtype=np.float64)
    for b in range(n_features):
        for j in range(n_features):
            if b == j:
                kl_mat[b, j] = 0.0
            else:
                kl_mat[b, j] = kl_divergence(dists[b], dists[j])
    # 行归一化, 保证特征分布落在特征相关空间, 减小对训练的影响
    row_max = kl_mat.max(axis=1, keepdims=True)
    kl_mat = kl_mat / np.maximum(row_max, EPS)
    return kl_mat


def compute_time_node_vectors(times: np.ndarray) -> np.ndarray:
    """
    构造所有时间节点的特征向量  S_pt = {F_1^pt, ..., F_a^pt}
    F_a^pt = [ -||t_a - t_1||, ..., -||t_a - t_m|| ]   (时间差嵌入空间)
    返回: (Tm, Tm)
    """
    n = len(times)
    diff = np.abs(times[:, None] - times[None, :])
    return -diff.astype(np.float64)


# ------------------------------------------------------------ 图整体构建 ----
class GraphBuilder:
    """
    将单个患者的 (时间, 特征, 值) 不规则序列转换为图网络张量:
      node_feat : (N, D)   N = Tm + Fm 个节点的特征矩阵 (时间差向量与KL向量
                           维度不同时以 0 填充到公共维度 D)
      adj       : (N, N)   邻接/边权矩阵, 块结构 [[0, V], [V^T, 0]],
                           仅在实际测量存在处非零 (无插补噪声)
      we        : (Tm, Fm) 边权矩阵 V (每个特征通道独立 min-max 归一化)
      we_mask   : (Tm, Fm) 测量存在掩码
    """

    def __init__(self, n_features: int, n_kl_bins: int = 32,
                 value_range: Tuple[float, float] = (0.0, 1.0),
                 max_time_nodes: int = 128):
        self.n_features = n_features
        self.n_kl_bins = n_kl_bins
        self.value_range = value_range
        self.max_time_nodes = max_time_nodes

    def __call__(self, records: np.ndarray,
                 pred_time: float) -> Dict[str, np.ndarray]:
        """
        records: (n_records, 3) 数组, 每行 [t, f, v], t 单位为小时
        pred_time: 预测时刻 Tm, 仅使用 t < pred_time 的记录 (防泄漏)
        """
        rec = records[records[:, 0] < pred_time]          # 严格早于预测时刻
        if len(rec) == 0:
            # 无可用记录 -> 构造空图(调用方应保证窗口内有数据)
            t_nodes = np.array([0.0])
        else:
            # 合并相同时间戳, 最多保留 max_time_nodes 个时间节点
            times = np.unique(rec[:, 0])
            if len(times) > self.max_time_nodes:
                times = times[:self.max_time_nodes]
                rec = rec[np.isin(rec[:, 0], times)]
            t_nodes = times

        Tm, Fm = len(t_nodes), self.n_features
        t_index = {t: i for i, t in enumerate(t_nodes)}

        # ---- 边权矩阵: 同一 (t, f) 若多次测量取均值 ----
        we = np.zeros((Tm, Fm), dtype=np.float64)
        cnt = np.zeros((Tm, Fm), dtype=np.float64)
        for t, f, v in rec:
            i = t_index[t]
            we[i, int(f)] += v
            cnt[i, int(f)] += 1.0
        we_mask = (cnt > 0).astype(np.float64)
        we = np.divide(we, cnt, out=np.zeros_like(we), where=cnt > 0)

        # 每个特征通道 min-max 归一化 (缺失位置保持 0)
        for f in range(Fm):
            vals = we[:, f][we_mask[:, f] > 0]
            if len(vals) > 1 and vals.max() > vals.min():
                we[:, f] = np.where(we_mask[:, f] > 0,
                                    (we[:, f] - vals.min()) / (vals.max() - vals.min()),
                                    0.0)
            elif len(vals) == 1:
                we[:, f] = np.where(we_mask[:, f] > 0, 0.5, 0.0)

        # ---- 节点特征 ----
        meas = {f: we[:, f][we_mask[:, f] > 0] for f in range(Fm)}
        kl_mat = compute_feature_node_vectors(meas, self.n_kl_bins,
                                              self.value_range, Fm)
        t_mat = compute_time_node_vectors(t_nodes)

        D = max(Tm, Fm)
        node_feat = np.zeros((Tm + Fm, D), dtype=np.float64)
        node_feat[:Tm, :Tm] = t_mat                   # 时间节点: 时间差嵌入
        node_feat[Tm:, :Fm] = kl_mat                  # 特征节点: KL 嵌入

        # ---- 邻接矩阵 (含自环, 对称归一化在模型内完成) ----
        adj = np.zeros((Tm + Fm, Tm + Fm), dtype=np.float64)
        adj[:Tm, Tm:] = we
        adj[Tm:, :Tm] = we.T
        adj = adj + np.eye(Tm + Fm)

        last_idx = int(np.max(np.nonzero(we_mask.sum(axis=1))[0])) if we_mask.any() else 0

        return dict(node_feat=node_feat.astype(np.float32),
                    adj=adj.astype(np.float32),
                    we=we.astype(np.float32),
                    we_mask=we_mask.astype(np.float32),
                    n_time=Tm, n_feat=Fm, last_idx=last_idx)
