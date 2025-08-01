"""
时空聚类分析工具函数库
用于处理 spaced-massed 学习研究中的聚类分析、sum_t 计算和 evoked 对象创建

主要功能：
1. 聚类数据提取和过滤
2. sum_t 计算和可视化
3. evoked 对象创建和操作
4. 中介分析数据准备
"""

import numpy as np
import pandas as pd
import mne
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union
import warnings


def filter_clusters_by_time_window(clusters, time_start: int, time_end: int) -> List[Tuple]:
    """
    根据时间窗口筛选聚类数据
    
    Parameters:
    -----------
    clusters : list of tuples
        聚类数据，每个元组包含 (x_array, y_array)
    time_start : int
        时间窗口起始点
    time_end : int
        时间窗口结束点
        
    Returns:
    --------
    filtered_clusters : list of tuples
        筛选后的聚类数据
    """
    filtered_clusters = []
    
    for x_array, y_array in clusters:
        # 创建时间窗口掩码
        mask = (x_array >= time_start) & (x_array <= time_end)
        
        # 应用掩码筛选数据
        filtered_x = x_array[mask]
        filtered_y = y_array[mask]
        
        # 只保留非空结果
        if filtered_x.size > 0 and filtered_y.size > 0:
            filtered_clusters.append((filtered_x, filtered_y))
    
    return filtered_clusters


def extract_cluster_timepoints_by_step(clusters, step: int = 1) -> Dict[int, List[Tuple]]:
    """
    按步长提取聚类的时间点数据
    
    Parameters:
    -----------
    clusters : list of tuples
        聚类数据
    step : int
        时间步长，默认为1（逐个时间点）
        
    Returns:
    --------
    timepoint_clusters : dict
        键为时间区间起始点，值为该区间内的聚类数据列表
    """
    timepoint_clusters = {}
    
    for x_array, y_array in clusters:
        # 获取时间点范围
        min_time = int(np.min(x_array))
        max_time = int(np.max(x_array))
        
        # 按步长分割时间区间
        for start_time in range(min_time, max_time + 1, step):
            end_time = start_time + step
            
            # 筛选当前时间区间的数据
            mask = (x_array >= start_time) & (x_array < end_time)
            filtered_x = x_array[mask]
            filtered_y = y_array[mask]
            
            if filtered_x.size > 0:
                if start_time not in timepoint_clusters:
                    timepoint_clusters[start_time] = []
                timepoint_clusters[start_time].append((filtered_x, filtered_y))
    
    return timepoint_clusters


def calculate_sum_t_from_t_values(t_values: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    从 t 值计算累积 sum_t
    
    Parameters:
    -----------
    t_values : np.ndarray
        t 值数组，可以是 1D 或 2D
    axis : int
        计算累积和的轴，默认为0（时间轴）
        
    Returns:
    --------
    sum_t : np.ndarray
        累积 t 值
    """
    return np.cumsum(t_values, axis=axis)


def create_evoked_from_statistical_values(stat_values: np.ndarray, 
                                         info: mne.Info, 
                                         times: np.ndarray,
                                         comment: str = 'Statistical values') -> mne.Evoked:
    """
    从统计值创建 MNE evoked 对象
    
    Parameters:
    -----------
    stat_values : np.ndarray
        统计值数组，形状为 (n_channels, n_times) 或 (n_times,)
    info : mne.Info
        MNE info 对象
    times : np.ndarray
        时间点数组
    comment : str
        evoked 对象的注释
        
    Returns:
    --------
    evoked : mne.Evoked
        创建的 evoked 对象
    """
    # 确保数据是 2D 数组
    if stat_values.ndim == 1:
        # 如果是 1D，扩展到所有通道
        n_channels = len(info['ch_names'])
        stat_values = np.tile(stat_values, (n_channels, 1))
    
    # 创建 evoked 对象
    evoked = mne.EvokedArray(
        data=stat_values,
        info=info,
        tmin=times[0],
        comment=comment,
        nave=1
    )
    
    return evoked


def create_sum_t_evoked_from_spaced_massed(dissimilarity_spaced: np.ndarray,
                                          dissimilarity_massed: np.ndarray,
                                          info: mne.Info,
                                          times: np.ndarray) -> Tuple[mne.Evoked, mne.Evoked, np.ndarray]:
    """
    从 spaced-massed 对比创建 sum_t evoked 对象
    
    Parameters:
    -----------
    dissimilarity_spaced : np.ndarray
        spaced 条件数据
    dissimilarity_massed : np.ndarray
        massed 条件数据
    info : mne.Info
        MNE info 对象
    times : np.ndarray
        时间点数组
        
    Returns:
    --------
    evoked_t : mne.Evoked
        原始 t 值的 evoked 对象
    evoked_sum_t : mne.Evoked
        sum_t 的 evoked 对象
    t_values : np.ndarray
        原始 t 值
    """
    # 计算 t 值
    t_values, p_values = ttest_rel(dissimilarity_spaced, dissimilarity_massed, axis=0)
    
    # 处理多维数据
    if t_values.ndim > 1:
        t_values = np.mean(t_values, axis=0) if t_values.shape[0] > 1 else t_values.squeeze()
    
    # 计算 sum_t
    sum_t_values = calculate_sum_t_from_t_values(t_values)
    
    # 创建 evoked 对象
    evoked_t = create_evoked_from_statistical_values(
        t_values, info, times, comment='spaced-massed t-values'
    )
    
    evoked_sum_t = create_evoked_from_statistical_values(
        sum_t_values, info, times, comment='spaced-massed sum_t'
    )
    
    return evoked_t, evoked_sum_t, t_values


def average_cluster_dataframes(dfs_in_cluster: List[pd.DataFrame]) -> pd.DataFrame:
    """
    对聚类内的多个 DataFrame 进行平均处理
    
    Parameters:
    -----------
    dfs_in_cluster : list of pd.DataFrame
        聚类内的 DataFrame 列表
        
    Returns:
    --------
    df_cluster_avg : pd.DataFrame
        平均后的 DataFrame
    """
    if not dfs_in_cluster:
        raise ValueError("输入的 DataFrame 列表为空")
    
    # 提取并堆叠 'distance' 列
    distance_arrays = [df['distance'].values for df in dfs_in_cluster]
    distance_stack = np.vstack(distance_arrays)
    
    # 计算平均值
    mean_distance = np.mean(distance_stack, axis=0)
    
    # 创建结果 DataFrame
    df_cluster_avg = dfs_in_cluster[0].copy()
    df_cluster_avg['distance'] = mean_distance
    
    return df_cluster_avg


def extract_cluster_data_from_dict(updated_dfs_dict: Dict, 
                                  cluster_timepoints: List[int], 
                                  cluster_channels: List[int]) -> List[pd.DataFrame]:
    """
    从数据字典中提取聚类对应的 DataFrame
    
    Parameters:
    -----------
    updated_dfs_dict : dict
        嵌套字典，结构为 {channel: {time: DataFrame}}
    cluster_timepoints : list of int
        聚类包含的时间点
    cluster_channels : list of int
        聚类包含的通道索引
        
    Returns:
    --------
    dfs_in_cluster : list of pd.DataFrame
        聚类内的 DataFrame 列表
    """
    dfs_in_cluster = []
    
    for ch_idx in cluster_channels:
        for time_idx in cluster_timepoints:
            if ch_idx in updated_dfs_dict and time_idx in updated_dfs_dict[ch_idx]:
                df = updated_dfs_dict[ch_idx][time_idx]
                dfs_in_cluster.append(df)
    
    return dfs_in_cluster


def prepare_cluster_for_mediation(updated_dfs_dict: Dict,
                                 cluster_x: np.ndarray,
                                 cluster_y: np.ndarray) -> pd.DataFrame:
    """
    为中介分析准备聚类数据
    
    Parameters:
    -----------
    updated_dfs_dict : dict
        数据字典
    cluster_x : np.ndarray
        聚类的时间点索引
    cluster_y : np.ndarray
        聚类的通道索引
        
    Returns:
    --------
    df_for_mediation : pd.DataFrame
        准备好的中介分析数据
    """
    # 获取唯一的时间点和通道
    unique_timepoints = np.unique(cluster_x).tolist()
    unique_channels = np.unique(cluster_y).tolist()
    
    # 提取聚类数据
    dfs_in_cluster = extract_cluster_data_from_dict(
        updated_dfs_dict, unique_timepoints, unique_channels
    )
    
    # 平均处理
    df_for_mediation = average_cluster_dataframes(dfs_in_cluster)
    
    return df_for_mediation


def plot_t_values_and_sum_t(t_values: np.ndarray, 
                           sum_t_values: np.ndarray, 
                           times: np.ndarray,
                           title_prefix: str = "Statistical Analysis") -> plt.Figure:
    """
    绘制 t 值和 sum_t 的对比图
    
    Parameters:
    -----------
    t_values : np.ndarray
        原始 t 值
    sum_t_values : np.ndarray
        累积 t 值
    times : np.ndarray
        时间点数组
    title_prefix : str
        图标题前缀
        
    Returns:
    --------
    fig : plt.Figure
        matplotlib 图形对象
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 绘制原始 t 值
    ax1.plot(times, t_values, 'b-', linewidth=2)
    ax1.set_title(f'{title_prefix}: T-values (Spaced vs Massed)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('T-value')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # 绘制累积 t 值
    ax2.plot(times, sum_t_values, 'r-', linewidth=2)
    ax2.set_title(f'{title_prefix}: Cumulative T-values (sum_t)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Cumulative T-value')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
