from neurora.rdm_cal import eegRDM
import numpy as np
import pandas as pd
import numpy as np
from neurora.rsa_plot import plot_tbyt_diff_decoding_acc
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
from scipy.stats import ttest_1samp, ttest_rel
from nilearn import plotting, datasets, surface
import nibabel as nib
from neurora.stuff import get_affine, get_bg_ch2, get_bg_ch2bet, correct_by_threshold, \
    clusterbased_permutation_1d_1samp_1sided, clusterbased_permutation_2d_1samp_1sided, \
    clusterbased_permutation_1d_1samp_2sided, clusterbased_permutation_2d_2sided, smooth_1d, get_cluster_index_1d_1sided
import mne
from decimal import Decimal
import nibabel as nib
import numpy as np
import os
import math
from scipy.stats import spearmanr, pearsonr, kendalltau, ttest_1samp, ttest_rel, ttest_ind
from skimage.measure import label
import sys
from neurora.stuff import get_cluster_index_1d_2sided,show_progressbar
def eegdataOfOneField(channel_fieldn, epoch1, epoch2):
    """
    作者：LYW

    功能：从全脑数据中拿出来某个区域的数据

    变量：把这个field对应的channel组成的string list传进来

    返回值：crop好的eegdata

    """
    epock1_fieldn = epoch1.copy().pick_channels(channel_fieldn)  # 加copy是因为会在自身修改。
    epock2_fieldn = epoch2.copy().pick_channels(channel_fieldn)
    num_chan_picked = len(channel_fieldn)
    '''进行这两个条件之间STPS的计算'''
    # shape of eegdate: [n_cons, n_subs, n_chls, n_ts]
    eegdata = np.zeros([2, 1, num_chan_picked, 501], dtype=np.float32)
    # 把T1 T2合并起来，但是是建立一个新的维度：conditions，并不是直接拼接到一起。
    # 所以事先定好结构，然后赋值进去
    eegdata[0] = epock1_fieldn.data
    eegdata[1] = epock2_fieldn.data
    # 为了使用RSA函数，增加一个维度trial维度，两种condition下都只有一个trail（evoked）
    eegdata = np.reshape(eegdata, [2, 1, 1, num_chan_picked, 501])

    return eegdata

# 这个函数就是针对100ms的时间窗的
# 450 50 is specific to the 100ms time window.
# resolutioin is 500,so every 2ms is 1 sample.

def OneLineOfSTPS(eegdata):
    """
    :arg:把evoked数据根据时间切块，每一个时间块都有对应的一个RSA，然后取RSA矩阵中的cell，作为这行STPS的值。
    :param eegdata: 2个条件下的eegdata,是
    :return: 只是十个值组成的一行的STPS
    """
    # 这里450是因为减掉了最前面的0.1基线部分。+
    OneLineOf_STPS = np.zeros([1, 450], dtype=np.float32)
    m = 0
    for m in range(450):
        n = m + 1
        ndarray = eegdata
        #用这个函数间接地求了矩阵之间的 差异值才对 Representational Dissimilarity Matrix
        #这里用的是time_win是默认值？
        #这里用的eegRDM最简单按的情况，相当于只是使用了计算 dissimilarity的部分。
        #n=0的时候，0到50，对应就是0ms到100ms的时间窗口。第一个算出来，应该放在-0.1s位置上。
        rdm = eegRDM(ndarray[:, :, :, :, n:n + 50],sub_opt=0)
        #50是和100ms的窗口保持一致。
        # rdm = eegRDM(ndarray[:, :, :, :, n:n + 50], time_opt=1, time_win=50)
        OneLineOf_STPS[0, m] = rdm[0, 1]
    return OneLineOf_STPS


'''

ttestRel 

'''
def clusterbased_permutation_1d_1samp_2sided_ttestRel(acc1, acc2,
                                                           level=0, p_threshold=0.05, clusterp_threshold=0.05,
                                                           n_threshold=10,
                                                           iter=1000,
                                                           nsubs=28,
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
    '''
    find out clusters of significant points
    '''

    ps = np.zeros([x])
    ts = np.zeros([x])
    for t in range(x):
        # print(t)
        ts[t], p = ttest_rel(acc1[:, t], acc2[:, t]) # paired t-test for each time point.
        # Perform a one-tailed t-test (data1 is less than data2)
        # ts[t], p  = ttest_ind(acc1[:, t], acc2[:, t], alternative='less')
        # print(ts[t])
        # print(p)
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
        chance = np.full([nsubs], level)
        print("\nPermutation test")

        for i in range(iter):
            i_ps = np.zeros([x])
            i_ts = np.zeros([x])
            # i_results = results[np.random.permutation(results.shape[0])]
            for t in range(x):
                '''
                shuffle
                '''
                # Shuffle the data
                combined = np.concatenate((acc1[:, t], acc2[:, t]))
                np.random.shuffle(combined)
                # Split the shuffled array back into new 'acc1' and 'acc2'
                new_acc1 = combined[:nsubs]
                new_acc2 = combined[nsubs:]
                '''
                算t p
                '''
                i_ts[t], p = ttest_ind(new_acc1, new_acc2, equal_var=False)

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

    if cluster_n2 != 0:
        cluster_ts = np.zeros([cluster_n2])
        for i in range(cluster_n2):
            for t in range(x):
                if cluster_index1[t] == i + 1:
                    cluster_ts[i] = cluster_ts[i] + ts[t]

        permu_ts = np.zeros([iter])  # 用来存储1000个T值
        chance = np.full([nsubs], level)
        print("\nPermutation test")

        for i in range(iter):
            i_ps = np.zeros([x])
            i_ts = np.zeros([x])
            # i_results = results[np.random.permutation(results.shape[0])]
            for t in range(x):
                '''
                shuffle
                '''
                # Shuffle the data
                combined = np.concatenate((acc1[:, t], acc2[:, t]))
                np.random.shuffle(combined)
                # Split the shuffled array back into new 'acc1' and 'acc2'
                new_acc1 = combined[:nsubs]
                new_acc2 = combined[nsubs:]
                '''
                算t p
                '''
                i_ts[t], p = ttest_ind(new_acc1, new_acc2, equal_var=False)

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
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1 - clusterp_threshold):
                for t in range(x):
                    if cluster_index2[t] == i + 1:
                        ps[t] = 0

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

    print(cluster_n1)
    print(cluster_n2)
    return ps


def plot_tbyt_diff_decoding_acc_ttest(acc1, acc2, start_time=0, end_time=1, time_interval=0.01, chance=0.5, p=0.05, cbpt=True,
                                clusterp=0.05, stats_time=[0, 1], color1='r', color2='b', label1='Condition1',
                                label2='Condition2', xlim=[0, 1], ylim=[0.4, 0.8], xlabel='Time (s)',
                                ylabel='Decoding Accuracy', figsize=[6.4, 3.6], x0=0, ticksize=12, fontsize=16,
                                markersize=2, legend_fontsize=14, title=None, title_fontsize=16, avgshow=False,
                                iter=1000, ):
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
        # ps_stats, ts_stats, pss_stats, cluster_p_values_pos, cluster_p_values_neg = clusterbased_permutation_1d_1samp_2sided_ttestind_2007(
        #                                                                   p_threshold=p,
        #                                                                   clusterp_threshold=clusterp, iter=iter,
        #                                                                   updated_dataframes_list=updated_dataframes_list[
        #                                                                                           stats_time1:stats_time2],
        #                                                                   x=stats_time2 - stats_time1-1,
        #     model_formula=model_formula
        #                                                                   # 因为你就给了updated_dataframes_list的是统计区间数据 而不是所有数据.
        #                                                                   )
        ps_stats = clusterbased_permutation_1d_1samp_2sided_ttestRel(acc1[:, stats_time1:stats_time2],
                                                                          acc2[:, stats_time1:stats_time2], level=0,
                                                                          p_threshold=p,
                                                                          clusterp_threshold=clusterp, iter=100,
                                                                          x=350
                                                                          # 因为你就给了updated_dataframes_list的是统计区间数据 而不是所有数据.
                                                                          )
        #
        # print(pss_stats)
        ps = np.zeros([nts])
        ps[stats_time1:stats_time2] = ps_stats

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
    # ax.spines["bottom"].set_position(("data", chance))
    ax.spines["bottom"].set_position(("data", 0)) # 底部坐标轴


    x = np.arange(start_time + 0.5 * tstep, end_time + 0.5 * tstep, tstep)
    # x = np.arange(start_time + 0.5 * tstep, end_time - 0.5 * tstep, tstep)

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

    return ps1, ps2, ps