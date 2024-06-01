'''

想要改成函数包

'''


import neurora.rsa_plot

'''
虽然这是不经济的，但是还是搬用这个方法试一试，这是你现在手里能找到的唯一的方法了。
'''

import sys
print(sys.executable)
import os
import numpy as np
import mne
from neurora.decoding import tbyt_decoding_kfold
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
from scipy.stats import ttest_1samp, ttest_rel
from neurora.stuff import get_affine, get_bg_ch2, get_bg_ch2bet, correct_by_threshold, \
    clusterbased_permutation_1d_1samp_1sided, clusterbased_permutation_2d_1samp_1sided, \
    clusterbased_permutation_1d_1samp_2sided, clusterbased_permutation_2d_2sided, smooth_1d,get_cluster_index_1d_1sided, show_progressbar, get_cluster_index_1d_2sided
from decimal import Decimal
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.stats import pearsonr

# 定义一个函数来处理读取epochs的重复逻辑
# 可以选取某些特定通道
def load_epochs(sub_ids, list_epochs_allSubs, base_data_path, channels_field):
    '''


    读取epochs数据，并将读取的epochs对象添加到列表中。


    参数:
    sub_ids (list): 包含所有被试的ID的列表。
    list_epochs (list): 包含所有被试的epochs对象的列表。
    base_data_path (str): 数据存放路径的前缀。
    channels_all (list): 包含所有被试的通道名称的列表。


    '''

    for sub_id in sub_ids:
        # 为每个sub_id更新data_path
        data_path = base_data_path + sub_id + '-epo.fif'

        # 读取epochs数据
        epochs_sub = mne.read_epochs(fname=data_path, preload=True)

        epochs_sub = epochs_sub.pick(picks=channels_field)  # 如果需要，选择特定的通道

        epochs_sub.equalize_event_counts(method='mintime')  # 和最少试次的条件对齐

        # 将读取的epochs对象添加到列表中
        list_epochs_allSubs.append(epochs_sub)


def format_channel_names(channel_names):
    """
    需求场景: 当你需要复制 channels 名称的时候

    格式化通道名称列表，使每个通道名被单引号包围，并以逗号分隔。

    参数:
    channel_names (list): 通道名称的列表。

    返回:
    str: 格式化后的通道名称字符串。
    """
    # 使用列表推导和字符串的 join 方法来创建格式化的字符串
    formatted_names = ', '.join(f"'{name}'" for name in channel_names)
    return f"[{formatted_names}]"


def organize_eeg_data_by_subject_and_condition(T1_sub_ids, list_epochs_One, list_epochs_M2, list_epochs_S2,
                                               channels_all):
    """
    其实就是把每个被试 三个条件的数据 合并成一个数组，然后返回

    处理多个条件下的 EEG 数据，并组织为特定格式的数组。

    参数:
        T1_sub_ids (list): 被试的标识符列表。
        list_epochs_One (list): 第一个条件下每个被试的 Epochs 对象列表。
        list_epochs_M2 (list): 第二个条件下每个被试的 Epochs 对象列表。
        list_epochs_S2 (list): 第三个条件下每个被试的 Epochs 对象列表。
        channels_all (list): 所有EEG通道的列表。

    返回:
        list: 每个被试处理后的数据列表，数据格式为 [n_cons=3, n_subs=1, n_trials, n_chls, n_ts=501]。
    """
    num_channel = len(channels_all)
    list_subdata = []

    for sub_index in range(len(T1_sub_ids)):
        num_trials = len(list_epochs_One[sub_index].events)
        num_times = list_epochs_One[sub_index].get_data().shape[2]
        subdata = np.zeros((3, num_trials, num_channel, num_times), dtype=np.float32)

        for condition_index, epochs_list in enumerate([list_epochs_One, list_epochs_M2, list_epochs_S2]):
            epochs = epochs_list[sub_index]
            data = epochs.get_data(picks='eeg')
            subdata[condition_index] = data # every condition

        list_subdata.append(subdata)

    return list_subdata


def sequential_decoding_analysis(list_subdata, decoding_params):
    """
     参数:
        list_subdata (list of ndarray): 每个被试的数据列表，其中每个元素是一个 ndarray，
            数据格式为 [n_cons, n_subs=1, n_trials, n_chls, n_ts]，其中：
            - n_cons 是条件数，
            - n_subs 是被试数（通常为1，因为数据已预分配到每个被试），
            - n_trials 是每种条件下的试验数，
            - n_chls 是通道数，
            - n_ts 是时间点数。

        decoding_params (dict): 用于解码的配置参数字典，包括：
            - n (int): 每个样本的重复次数。
            - navg (int): 需要平均的邻近样本数。
            - time_opt (str): 时间优化选项，如 'average'。
            - time_win (int): 时间窗大小（单位：毫秒）。
            - time_step (int): 时间窗步长（单位：毫秒）。
            - nfolds (int): 交叉验证的折数。
            - nrepeats (int): 交叉验证重复次数。
            - normalization (bool): 是否进行数据标准化。
            - pca (bool): 是否应用 PCA。
            - pca_components (float): 保留的 PCA 组件的比例或数量。
            - smooth (bool): 是否对数据进行平滑处理。

    返回:
        tuple: 包含两个列表，分别存储每个被试的条件对比（ONE vs M2 和 ONE vs S2）解码结果：
            - decoding_results_one_m2 (list of ndarray): 每个元素是一个 ndarray((1,n_ts))，表示每个被试的 ONE vs M2 解码精度。
            - decoding_results_one_s2 (list of ndarray): 每个元素是一个 ndarray，表示每个被试的 ONE vs S2 解码精度。
    """
    decoding_results_one_m2 = []
    decoding_results_one_s2 = []

    for sub_data in list_subdata:
        print("decoding:", sub_data)
        one_data = sub_data[0, 0]  # ONE condition data
        m2_data = sub_data[1, 0]  # M2 condition data
        s2_data = sub_data[2, 0]  # S2 condition data

        for condition_pair, result_list in [((one_data, m2_data), decoding_results_one_m2),
                                            ((one_data, s2_data), decoding_results_one_s2)]:
            combined_data = np.concatenate(condition_pair, axis=0)
            labels = np.zeros(combined_data.shape[0])
            labels[len(condition_pair[0]):] = 1

            data_expanded = np.expand_dims(combined_data, axis=0)
            labels_expanded = np.expand_dims(labels, axis=0)

            # 通过字典展开传递参数
            acc = tbyt_decoding_kfold(
                data=data_expanded,
                labels=labels_expanded,
                **decoding_params
            )
            result_list.append(acc)

    return decoding_results_one_m2, decoding_results_one_s2


print("导入成功")


def save_arrays_to_csv(results_m2, results_s2, filename_m2, filename_s2, shape):
    """将解码结果重塑并保存到 CSV 文件中"""
    array_m2 = np.array(results_m2).reshape(shape)
    array_s2 = np.array(results_s2).reshape(shape)
    np.savetxt(filename_m2, array_m2, delimiter=",")
    np.savetxt(filename_s2, array_s2, delimiter=",")


def fit_lmer_model(data, formula=None):
    """
    Fits a linear mixed effects model to the data using R's lmerTest package.

    Parameters:
    - data: Pandas DataFrame containing the data.
    - formula: A string representing the model formula in R's syntax.

    Returns:
    - The summary of the fitted model as an R object.
    """
    # Check if the formula is valid
    # print('if there is NaN in data, please remove it first. nan becase subjects not use all')
    # print(data.isna().sum())  # Sum of NAs in each column
    data.dropna(inplace=True)  # Drop rows with NAs
    # print('after dropna')
    # print(data.isna().sum())  # Sum of NAs in each column

    # Ensure automatic conversion between Pandas DataFrame and R DataFrame
    pandas2ri.activate()

    # Convert the Pandas DataFrame to an R DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)

    # Pass the data to the R environment
    ro.globalenv['r_data'] = r_data
    #
    # Fit the model in R
    x = ro.r('''
    library(lmerTest)
    fit <- lmer(''' + formula + ''', r_data)
    summary(fit)$coefficients["categoryS", c("t value", "Pr(>|t|)")]
    ''')

    # Deactivate the Pandas to R DataFrame conversion
    pandas2ri.deactivate()

    # print(x)
    return x[0], x[1]


def clusterbased_permutation_1d_1samp_2sided_mixmodel_2007(
                                                            p_threshold=0.05, clusterp_threshold=0.05,
                                                           n_threshold=2,
                                                           iter=1000,
                                                           updated_dataframes_list=None,
                                                           model_formula=None,
                                                           x=450
                                                           ):
    """
    1-sample & 2-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results : array
        A result matrix.
        The shape of results should be [n_subs, x]. n_subs represents the number of subjects.
    level : float. Default is 0.
        An expected value in null hypothesis. (Here, results > level)
    p_threshold : float. Default is 0.05.
        The threshold of p-values.
    clusterp_threshold : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    n_threshold : int. Default is 2.
        The threshold of number of values in one cluster (number of values per cluster > n_threshold).
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    ps : float
        The permutation test resultz, p-values.
        The shape of ps is [x]. The values in ps should be 0 or 1 or -1, which represent not significant point or
        significantly greater point or significantly less point after cluster-based permutation test, respectively.
    """
    ps = np.zeros([x])
    ps2 = np.zeros([x])
    ts = np.zeros([x])
    cluster_p_values_pos = [] # 用来存储每个cluster的p值 = index/iter
    cluster_p_values_neg = []
    print(len(updated_dataframes_list))
    print(x)

    for t in range(x):

        ts[t], p = fit_lmer_model(updated_dataframes_list[t], model_formula)
        # print(p)
        ps2[t] = p # ps2 record original p-values

        # 应该算单边的
        if p / 2 < p_threshold and ts[t] > 0:
            ps[t] = 1
        if p / 2 < p_threshold and ts[t] < 0:
            ps[t] = -1

    # cluster_n1 就是正的情况.
    cluster_index1, cluster_n1, cluster_index2, cluster_n2 = get_cluster_index_1d_2sided(ps)

    '''
    改成标准做法
    https://www.nature.com/articles/s41562-020-0901-2#Sec10
    '''
    if cluster_n1 != 0:
        cluster_ts = np.zeros([cluster_n1])
        for i in range(cluster_n1):
            for t in range(x):
                if cluster_index1[t] == i + 1:
                    cluster_ts[i] = cluster_ts[i] + ts[t]

        permu_ts = np.zeros([iter])  # 用来存储1000个T值
        # chance = np.full([nsubs], level)
        print("\nPermutation test")

        for i in range(iter):
            i_ps = np.zeros([x])
            i_ts = np.zeros([x])
            # i_results = results[np.random.permutation(results.shape[0])]
            for t in range(x):
                '''
                shuffle
                '''
                data_sample = updated_dataframes_list[t]
                shuffled_data = data_sample.groupby('subId', group_keys=False).apply(
                    lambda x: x.assign(category=x['category'].sample(frac=1).values))
                '''
                算t p
                '''
                i_ts[t], p = fit_lmer_model(shuffled_data, model_formula)
                '''
                单边转换
                '''
                if i_ts[t] > 0:  # 分边处理
                    p = p / 2
                else:
                    p = 1  # 认为最大,没有意义.
                # 单边?
                if p < p_threshold and i_ts[t] > 0:
                    i_ps[t] = 1
                else:
                    i_ps[t] = 0

            # 标注好显著iter的index,有1 有2,
            # 给出iter 的cluster数量
            i_cluster_index, i_cluster_n = get_cluster_index_1d_1sided(i_ps)
            if i_cluster_n != 0:
                i_cluster_ts = np.zeros([i_cluster_n])
                for j in range(i_cluster_n):
                    for t in range(x):
                        if i_cluster_index[t] == j + 1:
                            i_cluster_ts[j] = i_cluster_ts[j] + i_ts[t]
                permu_ts[i] = np.max(i_cluster_ts)
            else:
                permu_ts[i] = np.max(i_ts)
            show_progressbar("Calculating", (i + 1) * 100 / iter)
            if i == (iter - 1):
                print("\nCluster-based permutation test finished!\n")

        for i in range(cluster_n1):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1 - clusterp_threshold):
                for t in range(x):
                    if cluster_index1[t] == i + 1:
                        ps[t] = 0
            cluster_p_values_pos.append(index / iter) # 存储每个cluster的p值 = index/iter

    if cluster_n2 != 0:
        cluster_ts = np.zeros([cluster_n2])
        for i in range(cluster_n2):
            for t in range(x):
                if cluster_index2[t] == i + 1:
                    cluster_ts[i] = cluster_ts[i] + ts[t]

        permu_ts = np.zeros([iter])  # 用来存储1000个T值
        # chance = np.full([nsubs], level)
        print("\nPermutation test")

        for i in range(iter):
            i_ps = np.zeros([x])
            i_ts = np.zeros([x])
            # i_results = results[np.random.permutation(results.shape[0])]
            for t in range(x):
                data_sample = updated_dataframes_list[t]
                shuffled_data = data_sample.groupby('subId', group_keys=False).apply(
                    lambda x: x.assign(category=x['category'].sample(frac=1).values))
                i_ts[t], p = fit_lmer_model(shuffled_data, model_formula)
                if i_ts[t] < 0:  # 分边处理
                    p = p / 2
                else:
                    p = 1  # 认为最大,没有意义.
                # 单边?
                if p < p_threshold and i_ts[t] < 0:
                    i_ps[t] = 1
                else:
                    i_ps[t] = 0
            # 标注好显著的index,和cluster的个数?
            i_cluster_index, i_cluster_n = get_cluster_index_1d_1sided(i_ps)  # 双边标注
            if i_cluster_n != 0:
                i_cluster_ts = np.zeros([i_cluster_n])
                for j in range(i_cluster_n):
                    for t in range(x):
                        if i_cluster_index[t] == j + 1:
                            i_cluster_ts[j] = i_cluster_ts[j] + i_ts[t]
                permu_ts[i] = np.max(i_cluster_ts)
            else:
                permu_ts[i] = np.max(i_ts)
            show_progressbar("Calculating", (i + 1) * 100 / iter)
            if i == (iter - 1):
                print("\nCluster-based permutation test finished!\n")

        for i in range(cluster_n2):
            index = 0
            for j in range(iter):
                if cluster_ts[i] < permu_ts[j]: # should be <
                    index = index + 1
            # when cannot reject H0, then p = 0
            if index < iter * (1 - clusterp_threshold):
                for t in range(x):
                    if cluster_index2[t] == i + 1:
                        ps[t] = 0
            cluster_p_values_neg.append(index / iter) # 存储每个cluster的p值 = index/iter


    newps = np.zeros([x + 2])
    newps[1:x + 1] = ps

    for i in range(x):
        if newps[i + 1] == 1 and newps[i] != 1:
            index = 0
            while newps[i + 1 + index] == 1:
                index = index + 1
            if index < n_threshold:
                newps[i + 1:i + 1 + index] = 0

        if newps[i + 1] == -1 and newps[i] != -1:
            index = 0
            while newps[i + 1 + index] == -1:
                index = index + 1
            if index < n_threshold:
                newps[i + 1:i + 1 + index] = 0

    ps = newps[1:x + 1]

    print("p-values:", ps2)

    return ps, ts, ps2, cluster_p_values_pos, cluster_p_values_neg

def plot_tbyt_diff_decoding_acc(acc1, acc2, start_time=0, end_time=1, time_interval=0.01, chance=0.5, p=0.05, cbpt=True,
                                clusterp=0.05, stats_time=[0, 1], color1='r', color2='b', label1='Condition1',
                                label2='Condition2', xlim=[0, 1], ylim=[0.4, 0.8], xlabel='Time (s)',
                                ylabel='Decoding Accuracy', figsize=[6.4, 3.6], x0=0, ticksize=12, fontsize=16,
                                markersize=2, legend_fontsize=14, title=None, title_fontsize=16, avgshow=False,
                                updated_dataframes_list=None, iter=1000, model_formula=None):
    """
    Plot the differences of time-by-time decoding accuracies between two conditions
    Parameters
    ----------
    acc1 : array
        The decoding accuracies under condition1.
        The size of acc1 should be [n_subs, n_ts]. n_subs, n_ts represent the number of subjects and number of
        time-points.
    acc2 : array
        The decoding accuracies under condition2.
        The size of acc2 should be [n_subs, n_ts]. n_subs, n_ts represent the number of subjects and number of
        time-points.
    start_time : int or float. Default is 0.
        The start time.
    end_time : int or float. Default is 1.
        The end time.
    time_interval : float. Default is 0.01.
        The time interval between two time samples.
    chance : float. Default is 0.5.
        The chance level.
    p : float. Default is 0.05.
        The threshold of p-values.
    cbpt : bool True or False. Default is True.
        Conduct cluster-based permutation test or not.
    clusterp : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    stats_time : array or list [stats_time1, stats_time2]. Default os [0, 1].
        Time period for statistical analysis.
    color1 : matplotlib color or None. Default is 'r'.
        The color for the curve under condition1.
    color2 : matplotlib color or None. Default is 'r'.
        The color for the curve under condition2.
    label1 : string-array. Default is 'Condition1'.
        The Label of acc1's condition.
    label2 : string-array. Default is 'Condition2'.
        The Label of acc2's condition.
    xlim : array or list [xmin, xmax]. Default is [0, 1].
        The x-axis (time) view lims.
    ylim : array or list [ymin, ymax]. Default is [0.4, 0.8].
        The y-axis (decoding accuracy) view lims.
    xlabel : string. Default is 'Time (s)'.
        The label of x-axis.
    ylabel : string. Default is 'Decoding Accuracy'.
        The label of y-axis.
    figsize : array or list, [size_X, size_Y]. Default is [6.4, 3.6].
        The size of the figure.
    x0 : float. Default is 0.
        The Y-axis is at x=x0.
    ticksize : int or float. Default is 12.
        The size of the ticks.
    fontsize : int or float. Default is 16.
        The fontsize of the labels.
    markersize : int or float. Default is 2.
        The size of significant marker.
    legend_fontsize : int or float. Default is 14.
        The fontsize of the legend.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    avgshow : boolen True or False. Default is False.
        Show the averaging decoding accuracies or not.
    """

    print('update 2')

    if len(np.shape(acc1)) != 2 or len(np.shape(acc2)) != 2:
        return "Invalid input!"

    nsubs, nts = np.shape(acc1)
    tstep = float(Decimal((end_time - start_time) / nts).quantize(Decimal(str(time_interval))))

    if tstep != time_interval:
        return "Invalid input!"

    delta1 = (stats_time[0] - start_time) / tstep - int((stats_time[0] - start_time) / tstep)
    delta2 = (stats_time[1] - start_time) / tstep - int((stats_time[1] - start_time) / tstep)
    if delta1 == 0:
        stats_time1 = int((stats_time[0] - start_time) / tstep)
    else:
        stats_time1 = int((stats_time[0] - start_time) / tstep) + 1
    if delta2 == 0:
        stats_time2 = int((stats_time[1] - start_time) / tstep)
    else:
        stats_time2 = int((stats_time[1] - start_time) / tstep) + 1

    yminlim = ylim[0]
    ymaxlim = ylim[1]

    avg1 = np.average(acc1, axis=0)
    err1 = np.zeros([nts])
    for t in range(nts):
        err1[t] = np.std(acc1[:, t], ddof=1) / np.sqrt(nsubs)

    avg2 = np.average(acc2, axis=0)
    err2 = np.zeros([nts])
    for t in range(nts):
        err2[t] = np.std(acc2[:, t], ddof=1) / np.sqrt(nsubs)

    if cbpt == True:

        ps1_stats = clusterbased_permutation_1d_1samp_1sided(acc1[:, stats_time1:stats_time2], level=chance,
                                                             p_threshold=p, clusterp_threshold=clusterp, iter=iter)
        ps1 = np.zeros([nts])
        ps1[stats_time1:stats_time2] = ps1_stats
        ps2_stats = clusterbased_permutation_1d_1samp_1sided(acc2[:, stats_time1:stats_time2], level=chance,
                                                             p_threshold=p, clusterp_threshold=clusterp, iter=iter)
        print(stats_time1, stats_time2, time_interval)
        print((stats_time2 - stats_time1) / time_interval)

        ps2 = np.zeros([nts])
        ps2[stats_time1:stats_time2] = ps2_stats
        ps_stats, ts_stats, pss_stats, cluster_p_values_pos, cluster_p_values_neg = clusterbased_permutation_1d_1samp_2sided_mixmodel_2007(
                                                                          p_threshold=p,
                                                                          clusterp_threshold=clusterp, iter=iter,
                                                                          updated_dataframes_list=updated_dataframes_list[
                                                                                                  stats_time1:stats_time2],
                                                                          x=stats_time2 - stats_time1,
            model_formula=model_formula
                                                                          # 因为你就给了updated_dataframes_list的是统计区间数据 而不是所有数据.
                                                                          )
        print(pss_stats)
        ps = np.zeros([nts])
        ps[stats_time1:stats_time2] = ps_stats

    else:
        ps1 = np.zeros([nts])
        ps2 = np.zeros([nts])
        ps = np.zeros([nts])
        for t in range(nts):
            if t >= stats_time1 and t < stats_time2:
                ps1[t] = ttest_1samp(acc1[:, t], chance, alternative="greater")[1]
                ps2[t] = ttest_1samp(acc2[:, t], chance, alternative="greater")[1]
                if ps1[t] < p:
                    ps1[t] = 1
                else:
                    ps1[t] = 0
                if ps2[t] < p:
                    ps2[t] = 1
                else:
                    ps2[t] = 0
                if ttest_rel(acc1[:, t], acc2[:, t], alternative="greater")[1] < p / 2:
                    ps[t] = 1
                elif ttest_rel(acc1[:, t], acc2[:, t], alternative="less")[1] < p / 2:
                    ps[t] = -1
                else:
                    ps[t] = 0

    for t in range(nts):
        if ps1[t] == 1:
            plt.plot(t * tstep + start_time + 0.5 * tstep, (ymaxlim - yminlim) * 0.95 + yminlim, 's', color=color1,
                     alpha=0.8,
                     markersize=markersize)
        if ps2[t] == 1:
            plt.plot(t * tstep + start_time + 0.5 * tstep, (ymaxlim - yminlim) * 0.91 + yminlim, 's', color=color2,
                     alpha=0.8,
                     markersize=markersize)
        if ps[t] == 1:
            xi = [t * tstep + start_time, t * tstep + tstep + start_time]
            ymin = [avg2[t] + err2[t]]
            ymax = [avg1[t] - err1[t]]
            plt.fill_between(xi, -10, 10, facecolor="grey", alpha=0.2)
        if ps[t] == -1:
            xi = [t * tstep + start_time, t * tstep + tstep + start_time]
            ymin = [avg1[t] + err1[t]]
            ymax = [avg2[t] - err2[t]]
            plt.fill_between(xi, -10, 10, facecolor="yellow", alpha=0.2)

    fig = plt.gcf()
    fig.set_size_inches(figsize[0], figsize[1])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["left"].set_position(("data", x0))
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["bottom"].set_position(("data", chance))


    # x = np.arange(start_time + 0.5 * tstep, end_time + 0.5 * tstep, tstep)
    x = np.arange(start_time + 0.5 * tstep, end_time - 0.5 * tstep, tstep)

    if avgshow is True:
        plt.plot(x, avg1, color=color1, alpha=0.95)
        plt.plot(x, avg2, color=color2, alpha=0.95)


    print(f"Length of x: {len(x)}")
    print(f"Length of avg1: {len(avg1)}")
    print(f"Length of err1: {len(err1)}")

    plt.fill_between(x, avg1 + err1, avg1 - err1, facecolor=color1, alpha=0.75, label=label1)
    plt.fill_between(x, avg2 + err2, avg2 - err2, facecolor=color2, alpha=0.75, label=label2)
    # plt.fill_between(x, -10, 10, facecolor='grey', alpha=0.2, label=label2)
    plt.ylim(yminlim, ymaxlim)
    plt.xlim(xlim[0], xlim[1])
    plt.tick_params(labelsize=ticksize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.legend()
    ax = plt.gca()
    leg = ax.get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=legend_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.show()

    return ps1, ps2, ps, ts_stats, pss_stats, cluster_p_values_pos, cluster_p_values_neg

'''

exclude the static processs from the analysis.

'''
def plot_tbyt_diff_decoding_acc_withoutStatic(acc1, acc2, start_time=0, end_time=1, time_interval=0.01, chance=0.5, stats_time=[0, 1],
                                color1='r', color2='b', label1='Condition1',
                                label2='Condition2', xlim=[0, 1], ylim=[0.4, 0.8], xlabel='Time (s)',
                                ylabel='STPS', figsize=[6.4, 3.6], x0=0, ticksize=12, fontsize=16,
                                markersize=2, legend_fontsize=14, title=None, title_fontsize=16, avgshow=False,
                                ps=None, ps1=None, ps2=None):
    """
    Plot the differences of time-by-time decoding accuracies between two conditions
    Parameters
    ----------
    acc1 : array
        The decoding accuracies under condition1.
        The size of acc1 should be [n_subs, n_ts]. n_subs, n_ts represent the number of subjects and number of
        time-points.
    acc2 : array
        The decoding accuracies under condition2.
        The size of acc2 should be [n_subs, n_ts]. n_subs, n_ts represent the number of subjects and number of
        time-points.
    start_time : int or float. Default is 0.
        The start time.
    end_time : int or float. Default is 1.
        The end time.
    time_interval : float. Default is 0.01.
        The time interval between two time samples.
    chance : float. Default is 0.5.
        The chance level.
    p : float. Default is 0.05.
        The threshold of p-values.
    cbpt : bool True or False. Default is True.
        Conduct cluster-based permutation test or not.
    clusterp : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    stats_time : array or list [stats_time1, stats_time2]. Default os [0, 1].
        Time period for statistical analysis.
    color1 : matplotlib color or None. Default is 'r'.
        The color for the curve under condition1.
    color2 : matplotlib color or None. Default is 'r'.
        The color for the curve under condition2.
    label1 : string-array. Default is 'Condition1'.
        The Label of acc1's condition.
    label2 : string-array. Default is 'Condition2'.
        The Label of acc2's condition.
    xlim : array or list [xmin, xmax]. Default is [0, 1].
        The x-axis (time) view lims.
    ylim : array or list [ymin, ymax]. Default is [0.4, 0.8].
        The y-axis (decoding accuracy) view lims.
    xlabel : string. Default is 'Time (s)'.
        The label of x-axis.
    ylabel : string. Default is 'Decoding Accuracy'.
        The label of y-axis.
    figsize : array or list, [size_X, size_Y]. Default is [6.4, 3.6].
        The size of the figure.
    x0 : float. Default is 0.
        The Y-axis is at x=x0.
    ticksize : int or float. Default is 12.
        The size of the ticks.
    fontsize : int or float. Default is 16.
        The fontsize of the labels.
    markersize : int or float. Default is 2.
        The size of significant marker.
    legend_fontsize : int or float. Default is 14.
        The fontsize of the legend.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    avgshow : boolen True or False. Default is False.
        Show the averaging decoding accuracies or not.
    """

    if len(np.shape(acc1)) != 2 or len(np.shape(acc2)) != 2:
        return "Invalid input!"

    nsubs, nts = np.shape(acc1)
    tstep = float(Decimal((end_time - start_time) / nts).quantize(Decimal(str(time_interval))))

    if tstep != time_interval:
        return "Invalid input!"

    delta1 = (stats_time[0] - start_time) / tstep - int((stats_time[0] - start_time) / tstep)
    delta2 = (stats_time[1] - start_time) / tstep - int((stats_time[1] - start_time) / tstep)
    if delta1 == 0:
        stats_time1 = int((stats_time[0] - start_time) / tstep)
    else:
        stats_time1 = int((stats_time[0] - start_time) / tstep) + 1
    if delta2 == 0:
        stats_time2 = int((stats_time[1] - start_time) / tstep)
    else:
        stats_time2 = int((stats_time[1] - start_time) / tstep) + 1

    yminlim = ylim[0]
    ymaxlim = ylim[1]

    avg1 = np.average(acc1, axis=0)
    err1 = np.zeros([nts])
    for t in range(nts):
        err1[t] = np.std(acc1[:, t], ddof=1) / np.sqrt(nsubs)

    avg2 = np.average(acc2, axis=0)
    err2 = np.zeros([nts])
    for t in range(nts):
        err2[t] = np.std(acc2[:, t], ddof=1) / np.sqrt(nsubs)

    print('\nSignificant time-windows for condition 1:')
    for t in range(nts):
        if t == 0 and ps1[t] == 1:
            print(str(int(start_time * 1000)) + 'ms to ', end='')
        if t > 0 and ps1[t] == 1 and ps1[t - 1] == 0:
            print(str(int((start_time + t * tstep) * 1000)) + 'ms to ', end='')
        if t < nts - 1 and ps1[t] == 1 and ps1[t + 1] == 0:
            print(str(int((start_time + (t + 1) * tstep) * 1000)) + 'ms')
        if t == nts - 1 and ps1[t] == 1:
            print(str(int(stats_time[1] * 1000)) + 'ms')

    print('\nSignificant time-windows for condition 2:')
    for t in range(nts):
        if t == 0 and ps2[t] == 1:
            print(str(int(start_time * 1000)) + 'ms to ', end='')
        if t > 0 and ps2[t] == 1 and ps2[t - 1] == 0:
            print(str(int((start_time + t * tstep) * 1000)) + 'ms to ', end='')
        if t < nts - 1 and ps2[t] == 1 and ps2[t + 1] == 0:
            print(str(int((start_time + (t + 1) * tstep) * 1000)) + 'ms')
        if t == nts - 1 and ps2[t] == 1:
            print(str(int(stats_time[1] * 1000)) + 'ms')

    print('\nSignificant time-windows for condition 1 > condition 2:')
    for t in range(nts):
        if t == 0 and ps[t] == 1:
            print(str(int(start_time * 1000)) + 'ms to ', end='')
        if t > 0 and ps[t] == 1 and ps2[t - 1] < 1:
            print(str(int((start_time + t * tstep) * 1000)) + 'ms to ', end='')
        if t < nts - 1 and ps[t] == 1 and ps[t + 1] < 1:
            print(str(int((start_time + (t + 1) * tstep) * 1000)) + 'ms')
        if t == nts - 1 and ps[t] == 1:
            print(str(int(stats_time[1] * 1000)) + 'ms')

    print('\nSignificant time-windows for condition 2 > condition 1:')
    for t in range(nts):
        if t == 0 and ps[t] == -1:
            print(str(int(start_time * 1000)) + 'ms to ', end='')
        if t > 0 and ps[t] == -1 and ps2[t - 1] > -1:
            print(str(int((start_time + t * tstep) * 1000)) + 'ms to ', end='')
        if t < nts - 1 and ps[t] == -1 and ps[t + 1] > -1:
            print(str(int((start_time + (t + 1) * tstep) * 1000)) + 'ms')
        if t == nts - 1 and ps[t] == -1:
            print(str(int(stats_time[1] * 1000)) + 'ms')

    for t in range(nts):
        if ps1[t] == 1:
            plt.plot(t * tstep + start_time + 0.5 * tstep, (ymaxlim - yminlim) * 0.95 + yminlim, 's', color=color1,
                     alpha=0.8,
                     markersize=markersize)
        if ps2[t] == 1:
            plt.plot(t * tstep + start_time + 0.5 * tstep, (ymaxlim - yminlim) * 0.91 + yminlim, 's', color=color2,
                     alpha=0.8,
                     markersize=markersize)
        if ps[t] == 1:
            xi = [t * tstep + start_time, t * tstep + tstep + start_time]
            ymin = [avg2[t] + err2[t]]
            ymax = [avg1[t] - err1[t]]
            plt.fill_between(xi, -10, 10, facecolor="grey", alpha=0.2)
        if ps[t] == -1:
            xi = [t * tstep + start_time, t * tstep + tstep + start_time]
            ymin = [avg1[t] + err1[t]]
            ymax = [avg2[t] - err2[t]]
            plt.fill_between(xi, -10, 10, facecolor="yellow", alpha=0.2)

    fig = plt.gcf()
    fig.set_size_inches(figsize[0], figsize[1])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["left"].set_position(("data", x0))
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["bottom"].set_position(("data", chance))
    x = np.arange(start_time + 0.5 * tstep, end_time + 0.5 * tstep, tstep)
    x = np.arange(start_time + 0.5 * tstep, end_time - 0.5 * tstep, tstep)
    if avgshow is True:
        plt.plot(x, avg1, color=color1, alpha=0.95)
        plt.plot(x, avg2, color=color2, alpha=0.95)
    plt.fill_between(x, avg1 + err1, avg1 - err1, facecolor=color1, alpha=0.75, label=label1)
    plt.fill_between(x, avg2 + err2, avg2 - err2, facecolor=color2, alpha=0.75, label=label2)
    # plt.fill_between(x, -10, 10, facecolor='grey', alpha=0.2, label=label2)
    plt.ylim(yminlim, ymaxlim)
    plt.xlim(xlim[0], xlim[1])
    plt.tick_params(labelsize=ticksize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.legend()
    ax = plt.gca()
    leg = ax.get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=legend_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.show()

    return ps1, ps2, ps
'''

part for calculate distance 
including vectorize data and  calculate distance series 

'''


def limtozero(value):
    """ Ensure that negative values are set to zero. """
    return max(value, 0)
def condition_difference_score_permutations(data, time_opt='features',
                                            time_win=50, time_step=5,
                                            method='correlation', use_abs=False,
                                            return_type='all'):
    """
    Calculate distances representing the differences between two conditions in EEG data across time windows,
    considering all possible pairings of trials between the two conditions using specified distance metric.

    Parameters:
    - data (numpy.ndarray): The input data with shape [n_conds=2, n_trials, n_chls, n_ts]
        where n_conds must be 2, n_trials is the number of trials, n_chls is the number of channels,
        and n_ts is the number of time samples.
    - time_opt (str): Option to preprocess the data; 'average' to average all time points,
        'features' to use the reshaped feature vectors. Default is 'features'.
    - time_win (int): Length of the time window over which to calculate the distance.
    - time_step (int): Step size to move the window across the time samples.
    - method (str): Method to calculate distance between trials ('euclidean' or 'correlation').
    - use_abs (bool): If using 'correlation', whether to use the absolute value of the correlation coefficient.
    - return_type (str): Determines the return value; 'mean' returns the average distance per window,
        'all' returns an array of all computed distances per window.

    Returns:
    - numpy.ndarray: Depending on return_type, either an array of mean distances per window or a nested
      array of all distances for each window.
    """
    n_trials = data.shape[1]
    n_chls = data.shape[2]
    n_ts = data.shape[3]
    n_windows = (n_ts - time_win) // time_step + 1
    results = []

    for window_idx in range(n_windows):
        start_idx = window_idx * time_step
        end_idx = start_idx + time_win
        data_window = data[:, :, :, start_idx:end_idx]

        if time_opt == 'average':
            data_preprocessed = data_window.mean(axis=3)
        elif time_opt == 'features':
            data_preprocessed = data_window.reshape(2, n_trials, n_chls * time_win)

        distances = []
        for i in range(n_trials):
            for j in range(i + 1,n_trials):
                if method == 'correlation':
                    r, _ = pearsonr(data_preprocessed[0, i], data_preprocessed[1, j])
                    # distance = 1 - np.abs(r) if use_abs else 1 - r
                    distance = r
                elif method == 'euclidean':
                    distance = np.linalg.norm(data_preprocessed[0, i] - data_preprocessed[1, j])
                distances.append(distance)

        if return_type == 'mean':
            mean_distance = np.mean(distances)
            results.append(mean_distance)
        elif return_type == 'all':
            results.append(distances)

    return np.array(results)

# Example usage:
# data = np.random.rand(2, 10, 5, 100)  # Simulated data: [2 conditions, 10 trials, 5 channels, 100 time samples]
# mean_distances = condition_difference_score_permutations(data, return_type='mean')
# all_distances = condition_difference_score_permutations(data, return_type='all')
#
#

def normalize_across_comparisons(data, normalize=True):
    """
    Compute distances for two comparisons (Cond1 vs Cond2 and Cond1 vs Cond3),
    and optionally normalize these distances based on the global min and max from both sets.

    Parameters:
    - data: array, input data containing conditions.
    - normalize: bool, if True, normalize the distances; if False, return raw distances.

    Returns:
    - Tuple of arrays, normalized or raw distances for the two comparisons.
    """
    # Extract data subsets for the two comparisons
    cond1_vs_cond2 = np.array([data[0], data[1]])
    cond1_vs_cond3 = np.array([data[0], data[2]])

    # Calculate distances for both comparisons
    distances1 = condition_difference_score_permutations(cond1_vs_cond2)
    distances2 = condition_difference_score_permutations(cond1_vs_cond3)

    if normalize:
        # Combine distances to find the global min and max
        combined_distances = np.concatenate((distances1, distances2))
        min_distance = np.min(combined_distances)
        max_distance = np.max(combined_distances)

        # Normalize distances based on global min and max
        if max_distance != min_distance:
            normalized_distances1 = (distances1 - min_distance) / (max_distance - min_distance)
            normalized_distances2 = (distances2 - min_distance) / (max_distance - min_distance)
        else:
            normalized_distances1 = np.zeros_like(distances1)
            normalized_distances2 = np.zeros_like(distances2)

        return normalized_distances1, normalized_distances2
    else:
        # Return raw distances if normalization is not requested
        return distances1, distances2
# Example use
# Assuming data is your EEG dataset with the shape (3, n_trials, n_chls, n_ts)
# data = np.random.rand(3, 100, 32, 500)  # Simulated EEG data
# normalized_distances1, normalized_distances2 = normalize_across_comparisons(data)
# print("Normalized scores for Condition 1 vs Condition 2:", normalized_distances1)
# print("Normalized scores for Condition 1 vs Condition 3:", normalized_distances2)


'''

don't aveage the decoding accuracy for each trial, remain original data.

'''
def normalize_across_comparisons_trial(data, normalize=True):

    print(data)

    # Extract data subsets for the two comparisons
    cond1_vs_cond2 = np.array([data[0], data[1]])
    cond1_vs_cond3 = np.array([data[0], data[2]])

    # Calculate distances for both comparisons
    distances1 = condition_difference_score_permutations(cond1_vs_cond2)
    distances2 = condition_difference_score_permutations(cond1_vs_cond3)

    if normalize:
        # Normalize each set of distances for each window
        normalized_distances1 = [np.array(d) / max(d) if max(d) != 0 else np.zeros_like(d) for d in distances1]
        normalized_distances2 = [np.array(d) / max(d) if max(d) != 0 else np.zeros_like(d) for d in distances2]
        return normalized_distances1, normalized_distances2
    else:
        return distances1, distances2



#%%
'''
acc数据整理 for 统计permutation

just for mix model situation.

the updated_dataframes_list is a list of dataframes, 
each dataframe is the pivoted data with the final column updated.
structure of the list of dataframes is:
n_ts * (n_sub*2 = 56, 9+1 = 10)
'''


def save_arrays_to_csv(results_m2, results_s2, filename_m2, filename_s2, shape):
    """将解码结果重塑并保存到 CSV 文件中"""
    array_m2 = np.array(results_m2).reshape(shape)
    array_s2 = np.array(results_s2).reshape(shape)
    np.savetxt(filename_m2, array_m2, delimiter=",")
    np.savetxt(filename_s2, array_s2, delimiter=",")

def load_and_integrate_data(file_spaced, file_massed, sub_ids, num_samples):
    """
    Load data from CSV files and integrate into a DataFrame.

    Parameters:
    - file_spaced: str, path to the CSV file containing spaced data.
    - file_massed: str, path to the CSV file containing massed data.
    - sub_ids: list, list of subject IDs corresponding to the rows in the CSV files.
    - num_samples: int, number of sample columns in the CSV files.

    Returns:
    - integrated_data: DataFrame, a DataFrame with data integrated from both files.
    """
    data_spaced = pd.read_csv(file_spaced, header=None)
    data_massed = pd.read_csv(file_massed, header=None)
    columns = ['subId', 'category'] + [f'sample_{i+1}' for i in range(num_samples)]
    integrated_data = pd.DataFrame(columns=columns)

    for sub_id in sub_ids:
        row_s = [sub_id, 'S'] + data_spaced.loc[sub_ids.index(sub_id)].tolist()
        integrated_data = integrated_data._append(pd.Series(row_s, index=columns), ignore_index=True)
        row_m = [sub_id, 'M'] + data_massed.loc[sub_ids.index(sub_id)].tolist()
        integrated_data = integrated_data._append(pd.Series(row_m, index=columns), ignore_index=True)

    return integrated_data

def update_dataframes(data_frame, integrated_data, num_samples):
    """根据 integrated_data 更新 data_frame 并存储每个更新过的副本"""
    updated_dataframes_list = []
    for i in range(1, num_samples+1):
        df_copy = data_frame.copy()
        sample_column = f'sample_{i}'
        for index, row in df_copy.iterrows():
            match_row = integrated_data[(integrated_data['subId'] == row['subId']) & (integrated_data['category'] == row['category'])]
            if not match_row.empty and sample_column in match_row.columns:
                df_copy.at[index, 'final'] = match_row.iloc[0][sample_column]
        updated_dataframes_list.append(df_copy)
    return updated_dataframes_list

# # 设定文件路径和参数
# file_spaced = 'I:\\pycharmProject\\pre10\\analysis\\nps\\中间结果\\array_acc_spaced.csv'
# file_massed = 'I:\\pycharmProject\\pre10\\analysis\\nps\\中间结果\\array_acc_massed.csv'
# file_filtered = 'I:\\pycharmProject\\pre10\\统计和建模\\filtered_pivoted_data_2024年03月14日.csv'
#
# # 假设这些是你的解码结果和受试者ID列表
# # decoding_results_one_m2 = [...]
# # decoding_results_one_s2 = [...]
# # T1_sub_ids = [...]
#
# # 调用函数
# # save_arrays_to_csv(decoding_results_one_m2, decoding_results_one_s2, file_massed, file_spaced, (, 236))
# # integrated_data = load_and_integrate_data(file_spaced, file_massed, T1_sub_ids)
# # filtered_pivoted_data = pd.read_csv(file_filtered)
# # updated_dataframes_list = update_dataframes(filtered_pivoted_data, integrated_data, 236)


