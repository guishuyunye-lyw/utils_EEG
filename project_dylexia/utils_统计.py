
import numpy as np
import os.path as op
from pandas import read_csv
import mne
from mne.io import read_raw_fif
from mne.datasets import visual_92_categories
from neurora.nps_cal import nps
from neurora.rdm_cal import eegRDM_bydecoding
from neurora.rdm_corr import rdm_correlation_spearman
from neurora.corr_cal_by_rdm import rdms_corr
from neurora.rsa_plot import plot_rdm, plot_corrs_by_time, plot_nps_hotmap, plot_corrs_hotmap
from neurora.stuff import show_progressbar,get_cluster_index_1d_2sided
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel
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
    clusterbased_permutation_1d_1samp_2sided, clusterbased_permutation_2d_2sided, smooth_1d,get_cluster_index_1d_1sided
from decimal import Decimal

#%%


' a function for 2-sided cluster based permutation test for 1-D results '
'''
注意 你需要把ttest_rel  换成  ttest_ind
'''

def clusterbased_permutation_1d_2sided(results1, results2, p_threshold=0.05, clusterp_threshold=0.05, n_threshold=2,
                                       iter=1000):

    """
    2-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results1 : array
        A result matrix under condition1.
        The shape of results1 should be [n_subs, x]. n_subs represents the number of subjects.
    results2 : array
        A result matrix under condition2.
        The shape of results2 should be [n_subs, x]. n_subs represents the number of subjects. (Here, results1 >
        results2)
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

    nsubs, x = np.shape(results1)

    ps = np.zeros([x])
    ts = np.zeros([x])
    for t in range(x):
        # ts[t], p = ttest_rel(results1[:, t], results2[:, t])
        ts[t], p = ttest_ind(results1[:, t], results2[:, t])
        if p < p_threshold and ts[t] > 0:
            ps[t] = 1
        if p < p_threshold and ts[t] < 0:
            ps[t] = -1

    cluster_index1, cluster_n1, cluster_index2, cluster_n2 = get_cluster_index_1d_2sided(ps)

    if cluster_n1 != 0:
        cluster_ts = np.zeros([cluster_n1])
        for i in range(cluster_n1):
            for t in range(x):
                if cluster_index1[t] == i + 1:
                    cluster_ts[i] = cluster_ts[i] + ts[t]

        permu_ts = np.zeros([iter])
        print("\nPermutation test\n")
        print("Side 1 begin:")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n1])
            for j in range(cluster_n1):
                for t in range(x):
                    if cluster_index1[t] == j + 1:
                        v = np.hstack((results1[:, t], results2[:, t]))
                        vshuffle = np.random.permutation(v)
                        v1 = vshuffle[:nsubs]
                        v2 = vshuffle[nsubs:]
                        permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="greater")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nSide 1 finished!\n")

        for i in range(cluster_n1):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1-clusterp_threshold):
                for t in range(x):
                    if cluster_index1[t] == i + 1:
                        ps[t] = 0

    if cluster_n2 != 0:
        cluster_ts = np.zeros([cluster_n2])
        for i in range(cluster_n2):
            for t in range(x):
                if cluster_index2[t] == i + 1:
                    cluster_ts[i] = cluster_ts[i] + ts[t]

        permu_ts = np.zeros([iter])
        print("Side 2 begin:\n")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n2])
            for j in range(cluster_n2):
                for t in range(x):
                    if cluster_index2[t] == j + 1:
                        v = np.hstack((results1[:, t], results2[:, t]))
                        vshuffle = np.random.permutation(v)
                        v1 = vshuffle[:nsubs]
                        v2 = vshuffle[nsubs:]
                        # 改成独立样本t检验，或者是别的。。
                        # permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="less")[0]
                        permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_ind(v1, v2, alternative="less")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nSide 2 finished!\n")
                print("Cluster-based permutation test finished!\n")

        for i in range(cluster_n2):
            index = 0
            for j in range(iter):
                if cluster_ts[i] < permu_ts[j]:
                    index = index + 1
            if index < iter * (1-clusterp_threshold):
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

    return ps



def plot_tbyt_diff_decoding_acc(acc1, acc2, start_time=0, end_time=1, time_interval=0.01, chance=0.5, p=0.05, cbpt=True,
                                clusterp=0.05, stats_time=[0, 1], color1='r', color2='b', label1='Condition1',
                                label2='Condition2', xlim=[0, 1], ylim=[0.4, 0.8], xlabel='Time (s)',
                                ylabel='Decoding Accuracy', figsize=[6.4, 3.6], x0=0, ticksize=12, fontsize=16,
                                markersize=2, legend_fontsize=14, title=None, title_fontsize=16, avgshow=False):
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
    time_interval : float. Default is 0.binary.
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

    # 不在要求一致性，不做配对测试
    # if len(np.shape(acc1)) != 2 or len(np.shape(acc2)) != 2:
    #     return "Invalid input!"

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
                                                             p_threshold=p, clusterp_threshold=clusterp, iter=1000)
        ps1 = np.zeros([nts])
        ps1[stats_time1:stats_time2] = ps1_stats
        ps2_stats = clusterbased_permutation_1d_1samp_1sided(acc2[:, stats_time1:stats_time2], level=chance,
                                                             p_threshold=p, clusterp_threshold=clusterp, iter=1000)
        ps2 = np.zeros([nts])
        ps2[stats_time1:stats_time2] = ps2_stats
        # 这里直接用差值代替了acc1[:, stats_time1:stats_time2]-acc2[:, stats_time1:stats_time2]
        # ps_stats = clusterbased_permutation_1d_1samp_2sided(acc1[:, stats_time1:stats_time2] -
        #                                                     acc2[:, stats_time1:stats_time2], level=0, p_threshold=p,
        #                                                     clusterp_threshold=clusterp, iter=1000)
        # 尝试直接换成clusterbased_permutation_1d_2sided
        ps_stats = clusterbased_permutation_1d_2sided(acc1[:, stats_time1:stats_time2],acc2[:, stats_time1:stats_time2],
                                                      p_threshold=0.05, clusterp_threshold=0.05, n_threshold=3,
                                       iter=1000)

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
                # if ttest_rel(acc1[:, t], acc2[:, t], alternative="greater")[1] < p / 2:
                if ttest_ind(acc1[:, t], acc2[:, t], alternative="greater")[1] < p / 2:
                    ps[t] = 1
                # elif ttest_rel(acc1[:, t], acc2[:, t], alternative="less")[1] < p / 2:
                elif ttest_ind(acc1[:, t], acc2[:, t], alternative="less")[1] < p / 2:
                    ps[t] = -1
                else:
                    ps[t] = 0

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
            # plt.fill_between(xi, ymax, ymin, facecolor="grey", alpha=0.2)
            plt.fill_between(xi, -10, 10, facecolor="yellow", alpha=0.2)
        if ps[t] == -1:
            xi = [t * tstep + start_time, t * tstep + tstep + start_time]
            ymin = [avg1[t] + err1[t]]
            ymax = [avg2[t] - err2[t]]
            # plt.fill_between(xi, ymax, ymin, facecolor="grey", alpha=0.2)
            plt.fill_between(xi, -10, 10, facecolor="grey", alpha=0.2)


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
    if avgshow is True:
        plt.plot(x, avg1, color=color1, alpha=0.95)
        plt.plot(x, avg2, color=color2, alpha=0.95)
    plt.fill_between(x, avg1 + err1, avg1 - err1, facecolor=color1, alpha=0.75, label=label1)
    plt.fill_between(x, avg2 + err2, avg2 - err2, facecolor=color2, alpha=0.75, label=label2)
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


def clusterbased_permutation_1d_1samp_2sided_ttestind_2007(acc1, acc2,
                                                           level=0, p_threshold=0.05, clusterp_threshold=0.05,
                                                           n_threshold=2,
                                                           iter=1000,
                                                           updated_dataframes_list=None,
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
    ps = np.zeros([x])
    ts = np.zeros([x])
    for t in range(x):
        print(t)
        ts[t], p = ttest_ind(acc1[:, t], acc2[:, t], equal_var=False)
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


def plot_tbyt_diff_decoding_acc(acc1, acc2, start_time=0, end_time=1, time_interval=0.01, chance=0.5, p=0.05, cbpt=True,
                                clusterp=0.05, stats_time=[0, 1], color1='r', color2='b', label1='Condition1',
                                label2='Condition2', xlim=[0, 1], ylim=[0.4, 0.8], xlabel='Time (s)',
                                ylabel='Decoding Accuracy', figsize=[6.4, 3.6], x0=0, ticksize=12, fontsize=16,
                                markersize=2, legend_fontsize=14, title=None, title_fontsize=16, avgshow=False,
                                ):
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
    time_interval : float. Default is 0.binary.
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

    if cbpt == True:

        ps1_stats = clusterbased_permutation_1d_1samp_1sided(acc1[:, stats_time1:stats_time2], level=chance,
                                                             p_threshold=p, clusterp_threshold=clusterp, iter=100)
        ps1 = np.zeros([nts])
        ps1[stats_time1:stats_time2] = ps1_stats
        ps2_stats = clusterbased_permutation_1d_1samp_1sided(acc2[:, stats_time1:stats_time2], level=chance,
                                                             p_threshold=p, clusterp_threshold=clusterp, iter=100)
        ps2 = np.zeros([nts])
        ps2[stats_time1:stats_time2] = ps2_stats
        ps_stats = clusterbased_permutation_1d_1samp_2sided_ttestind_2007(acc1[:, stats_time1:stats_time2],
                                                                          acc2[:, stats_time1:stats_time2], level=0,
                                                                          p_threshold=p,
                                                                          clusterp_threshold=clusterp, iter=100,
                                                                          x=199,# x=391
                                                                          # 因为你就给了updated_dataframes_list的是统计区间数据 而不是所有数据.
                                                                          )
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
        if ps[t] == 1:  # spaced 大于 massed
            xi = [t * tstep + start_time, t * tstep + tstep + start_time]
            ymin = [avg2[t] + err2[t]]
            ymax = [avg1[t] - err1[t]]
            plt.fill_between(xi, -10, 10, facecolor="yellow", alpha=0.2)
        if ps[t] == -1:
            xi = [t * tstep + start_time, t * tstep + tstep + start_time]
            ymin = [avg1[t] + err1[t]]
            ymax = [avg2[t] - err2[t]]
            plt.fill_between(xi, -10, 10, facecolor="grey", alpha=0.2)

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
        # ps_stats, ts_stats, pss_stats, cluster_p_values_pos, cluster_p_values_neg = clusterbased_permutation_1d_1samp_2sided_ttestind_2007(
        #                                                                   p_threshold=p,
        #                                                                   clusterp_threshold=clusterp, iter=iter,
        #                                                                   updated_dataframes_list=updated_dataframes_list[
        #                                                                                           stats_time1:stats_time2],
        #                                                                   x=stats_time2 - stats_time1-1,
        #     model_formula=model_formula
        #                                                                   # 因为你就给了updated_dataframes_list的是统计区间数据 而不是所有数据.
        #                                                                   )
        ps_stats = clusterbased_permutation_1d_1samp_2sided_ttestind_2007(acc1[:, stats_time1:stats_time2],
                                                                          acc2[:, stats_time1:stats_time2], level=0,
                                                                          p_threshold=p,
                                                                          clusterp_threshold=clusterp, iter=100,
                                                                          x=157
                                                                          # 因为你就给了updated_dataframes_list的是统计区间数据 而不是所有数据.
                                                                          )
        #
        # print(pss_stats)
        ps = np.zeros([nts])
        ps[stats_time1:stats_time2] = ps_stats


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

'''

exclude the static part from the analysis.

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


# 这个就不是为decoding准备，少一步。
# 不需要返回label，只要给出
def prepare_epochs_data_con(epochs_all, conds, channels_all):
    num_trials = len(epochs_all[conds[0]].events[:,1])
    subdata = np.zeros((len(conds), num_trials, len(channels_all), 1001), dtype=np.float32)
    for i, cond in enumerate(conds):
        epochs = epochs_all[cond]
        data = epochs.get_data(picks='eeg')
        subdata[i] = data
    subdata = np.reshape(subdata, (len(conds), 1, num_trials, len(channels_all), 1001))
    return subdata

def prepare_epochs_data(epochs_all, conds, channels_all):
    num_trials = len(epochs_all[conds[0]].events[:,1])
    subdata = np.zeros((len(conds), num_trials, len(channels_all), 1001), dtype=np.float32)
    sublabel = np.array([])
    for i, cond in enumerate(conds):
        epochs = epochs_all[cond]
        data = epochs.get_data(picks='eeg')
        label_cond = epochs.events[:,2]
        sublabel = np.append(sublabel, label_cond)
        subdata[i] = data
    sublabel = sublabel.reshape(1, -1)  # This ensures sublabel is (1, 1344) if total labels count is 1344
    subdata = np.reshape(subdata, (len(conds), 1, num_trials, len(channels_all), 1001))
    data_decode = np.reshape(subdata, (1, len(conds) * num_trials, len(channels_all), 1001))
    return data_decode, sublabel

def load_and_preprocess_data(sub_ids, file_path, channels_all, conds):
    list_subdata = []
    list_sublabel = []
    for sub_id in sub_ids:
        data_path = f"{file_path}{sub_id}_RSA-epo.fif"
        epochs_all = mne.read_epochs(fname=data_path).pick(picks=channels_all)
        epochs_all.equalize_event_counts(method='mintime')  # Align to the minimum condition
        subdata, sublabel = prepare_epochs_data(epochs_all, conds, channels_all)
        list_subdata.append(subdata)
        list_sublabel.append(sublabel)
    del epochs_all
    return list_subdata, list_sublabel



#%%

'''
数据结构： n_subs * n_t * 2（corr,p），需要变[n_subs, n_ts]
'''

# corrs_dd_fuyin = np.load('./RSA/rdms/corrs_dd_fuyin.npy')
# corrs_dd_yindiao = np.load('./RSA/rdms/corrs_dd_yindiao.npy')
#
# corrs_td_fuyin = np.load('./RSA/rdms/corrs_td_fuyin.npy')
# corrs_td_yindiao = np.load('./RSA/rdms/corrs_td_yindiao.npy')
#
# # 提取相关性数据（假设相关性数据位于每个数组的最后一个维度的第一个位置）
# corr_dd_fuyin = corrs_dd_fuyin[:, :, 0]
# corr_dd_yindiao = corrs_dd_yindiao[:, :, 0]
# corr_td_fuyin = corrs_td_fuyin[:, :, 0]
# corr_td_yindiao = corrs_td_yindiao[:, :, 0]
# #%%
# plot_tbyt_diff_decoding_acc(corr_td_yindiao,corr_dd_yindiao, start_time=-0.2, end_time=0.755, time_interval=0.005, chance=0, p=0.05, cbpt=True,
#                                 clusterp=0.05, stats_time=[-0.2, 0.8], color1='r', color2='b', label1='td',
#                                 label2='dd', xlim=[0, 1], ylim=[-0.2, 0.2], xlabel='Time (s)',
#                                 ylabel='RSA CORR', figsize=[8, 3.6], x0=0, ticksize=12, fontsize=16,
#                                 markersize=2, legend_fontsize=14, title=None, title_fontsize=16, avgshow=False)
#
# #%%
#
# plot_tbyt_diff_decoding_acc(corr_td_fuyin,corr_dd_fuyin, start_time=-0.2, end_time=0.755, time_interval=0.005, chance=0, p=0.05, cbpt=True,
#                                 clusterp=0.1, stats_time=[-0.2, 0.8], color1='r', color2='b', label1='td',
#                                 label2='dd', xlim=[0, 1], ylim=[-0.2, 0.2], xlabel='Time (s)',
#                                 ylabel='RSA CORR', figsize=[8, 3.6], x0=0, ticksize=12, fontsize=16,
#                                 markersize=2, legend_fontsize=14, title=None, title_fontsize=16, avgshow=False)

#%%

