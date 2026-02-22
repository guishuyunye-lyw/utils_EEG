# utils_EEG

EEG 数据分析工具库，服务于间隔学习 vs 集中学习的记忆巩固研究（pre10 项目）。

## 模块说明

### 核心分析模块

| 模块 | 用途 |
|------|------|
| `utils_cluster_analysis.py` | 时空聚类分析：cluster 合并、sum-t 计算、evoked 对象构建 |
| `utils_sumtPermu.py` | Sum-t 置换检验：基于 p 值的 cluster 检测与置换分布计算 |
| `utils_tfcePermu.py` | TFCE 置换检验：Threshold-Free Cluster Enhancement 的 max/min 置换分布 |
| `cluster_level_back.py` | MNE cluster-level 统计备份模块（从 MNE-Python 复制，含自定义修改） |
| `decoding_parallel.py` | 多被试并行 EEG 解码（封装 neurora） |
| `stuff.py` | 通用工具函数：信号处理、绘图、t 值计算等 |
| `stuff_nps_dist_ttest_whl_evoked.py` | NPS（神经模式相似性）距离分析与 t 检验 |

### Notebook 工作流

| Notebook | 用途 |
|----------|------|
| `prepare_permutation_csvs.ipynb` | Step 1: 从 Mplus 原始 CSV 生成 6 组分析所需的 3 个 CSV 文件 |
| `find_clusters_6groups.ipynb` | Step 2: 对 real data 和 permutation data 分 6 组找 cluster |
| `utils_sumtPermu.ipynb` | 旧版 sum-t 置换检验流程（4 组，手动切换） |
| `utils_tfcePermu.ipynb` | TFCE 置换检验流程 |
| `拿到a b ab的z p.ipynb` | 从 Mplus 结果提取 a/b/ab 路径的 z 值和 p 值 |

### 辅助工具

| 模块 | 用途 |
|------|------|
| `ai_variable_inspector.py` | Jupyter 变量结构检查（支持 minimal/auto/normal/full 模式） |
| `inspect_for_ai.py` | AI 检查点工具（变量分析） |
| `save_a_data.py` | 导出中介模型 a 路径结果到 CSV（独立脚本） |
| `save_filtered_data.py` | 导出筛选/重命名后的结果到 CSV（独立脚本） |

## 6 组置换检验流程

```
Mplus 原始 CSV
    │
    ▼
prepare_permutation_csvs.ipynb    ← Step 1: 生成 _a.csv, _ab_a大于0.csv, _ab_a小于等于0.csv
    │
    ▼
find_clusters_6groups.ipynb       ← Step 2: 6 组 × (real + permutation) 找 cluster
    │
    ▼
输出:
  - Real data: pickle (clustersT, clusterT_sum)
  - Permutation: min_max_cluster_sums_{group_id}.csv
```

### 6 组定义

| # | group_id | CSV | tail | 含义 |
|---|----------|-----|------|------|
| 1 | a_tail1 | `_a.csv` | +1 | 正的 a 路径效应 |
| 2 | a_tail-1 | `_a.csv` | -1 | 负的 a 路径效应（分散学习带来更多激活） |
| 3 | ab_a_gt_0_tail1 | `_ab_a大于0.csv` | +1 | 正的间接效应（分散学习有更多理解加成） |
| 4 | ab_a_le_0_tail1 | `_ab_a小于等于0.csv` | +1 | 正的间接效应（a≤0 子集） |
| 5 | ab_a_gt_0_tail-1 | `_ab_a大于0.csv` | -1 | 负的间接效应（a>0 子集） |
| 6 | ab_a_le_0_tail-1 | `_ab_a小于等于0.csv` | -1 | 负的间接效应（集中学习有更多理解加成） |

## 依赖

- numpy, scipy, pandas, mne, matplotlib, seaborn
- neurora（解码模块）
- 自定义 `mne.stats.cluster_level_backup`（修改版 MNE cluster 模块）

## 数据路径

- Mplus 结果: `D:\LYW\R for permutation t\Mplus\results\`
- Cluster 输出: `D:\LYW\pre10\data\permutation_summary_between\`
- EEG 数据: `D:\LYW\pre10\data\`
