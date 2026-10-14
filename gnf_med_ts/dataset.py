"""
数据集与批处理 (PyTorch Dataset + 合成数据生成器)

真实数据接口: 期望一个 DataFrame / 数组, 每行 [patient_id, t, f, v, ...],
配合标签 (death, phenotypes, stay_hours) 即可直接替换 SyntheticGenerator。
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from graph_construction import GraphBuilder


class MedicalGraphDataset(Dataset):
    """
    将每位患者的记录构建为图。每个样本:
      node_feat (Nmax, D), adj (Nmax, Nmax), we/we_mask (Tmax, Fmax),
      n_time, last_idx, labels {death, pheno, stay}
    """

    def __init__(self, records_list, labels_list, builder: GraphBuilder,
                 pred_time: float):
        self.records = records_list      # list of (n_rec, 3) ndarray [t, f, v]
        self.labels = labels_list        # list of dict(death=0/1, pheno=np.array, stay=float)
        self.builder = builder
        self.pred_time = pred_time
        self.cache = [None] * len(records_list)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if self.cache[idx] is None:
            g = self.builder(self.records[idx], self.pred_time)
            g['death'] = np.float32(self.labels[idx]['death'])
            g['pheno'] = np.asarray(self.labels[idx]['pheno'], dtype=np.float32)
            g['stay'] = np.float32(self.labels[idx]['stay'])
            self.cache[idx] = g
        return self.cache[idx]


def collate_graphs(batch):
    """变长图 padding 到批内最大尺寸"""
    Tm = max(b['n_time'] for b in batch)
    Fm = max(b['n_feat'] for b in batch)
    N = Tm + Fm
    D = batch[0]['node_feat'].shape[1]
    B = len(batch)

    node_feat = np.zeros((B, N, D), dtype=np.float32)
    adj = np.zeros((B, N, N), dtype=np.float32)
    we = np.zeros((B, Tm, Fm), dtype=np.float32)
    we_mask = np.zeros((B, Tm, Fm), dtype=np.float32)
    n_time = np.zeros(B, dtype=np.int64)
    last_idx = np.zeros(B, dtype=np.int64)
    death = np.zeros(B, dtype=np.float32)
    stay = np.zeros(B, dtype=np.float32)
    pheno = np.stack([b['pheno'] for b in batch])

    for i, b in enumerate(batch):
        t, f = b['n_time'], b['n_feat']
        node_feat[i, :t + f, :b['node_feat'].shape[1]] = b['node_feat']
        adj[i, :t + f, :t + f] = b['adj']
        we[i, :t, :f] = b['we']
        we_mask[i, :t, :f] = b['we_mask']
        n_time[i] = t
        last_idx[i] = b['last_idx']
        death[i] = b['death']
        stay[i] = b['stay']

    return dict(node_feat=torch.from_numpy(node_feat),
                adj=torch.from_numpy(adj),
                we=torch.from_numpy(we).unsqueeze(1),       # (B,1,Tm,Fm)
                we_mask=torch.from_numpy(we_mask).unsqueeze(1),
                n_time=torch.from_numpy(n_time),
                last_idx=torch.from_numpy(last_idx),
                death=torch.from_numpy(death),
                pheno=torch.from_numpy(pheno),
                stay=torch.from_numpy(stay))


# ------------------------------------------------------------ 合成数据 ----
class SyntheticGenerator:
    """
    生成与论文设定一致的不规则采样 ICU 时间序列:
    - 每位患者 42 个特征通道, 采样间隔不均、大量缺失
    - 死亡/表型标签与部分关键特征的真实变化模式相关
    - 住院时长 (stay_hours) 与病情严重程度相关
    """

    def __init__(self, n_patients=2000, n_features=42, seed=42):
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        half = n_patients // 2
        death = np.concatenate([np.ones(half), np.zeros(n_patients - half)])
        rng.shuffle(death)

        records_list, labels_list = [], []
        stay_min, stay_max = 12.0, 720.0     # 归一化参照 (最小-最大)
        for i in range(n_patients):
            d = death[i]
            severity = rng.beta(2 + 2 * d, 5 - 2 * d)          # 死亡组更重
            n_meas = rng.integers(30, 300)
            t = np.sort(rng.uniform(0, 48, size=n_meas))       # 不规则时间
            f = rng.integers(0, n_features, size=n_meas)       # 随机特征
            base = rng.normal(0, 1, n_features)
            # 关键特征 (0..4) 随时间恶化/好转, 其余为噪声
            trend = np.where(f < 5, (0.05 + 0.15 * d) * t, 0.0)
            v = base[f] + trend + rng.normal(0, 0.3, n_meas) + severity
            records_list.append(np.stack([t, f, v], axis=1).astype(np.float32))

            pheno = (rng.random(20) < 0.1 + 0.4 * severity * d).astype(np.float32)
            stay = float(np.clip((1 - severity) * (stay_max - stay_min)
                                 * (0.5 if d else 1.0) + stay_min,
                                 stay_min, stay_max))
            labels_list.append(dict(death=d, pheno=pheno,
                                    stay=(stay - stay_min) / (stay_max - stay_min)))
        self.records_list = records_list
        self.labels_list = labels_list
        self.stay_min, self.stay_max = stay_min, stay_max
