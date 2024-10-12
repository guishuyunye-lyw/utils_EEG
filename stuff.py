'''

想要改成函数包

'''
from imageio.v3 import improps

# from 行为数据整理.initialtest数据整理2023年10月17日 import sub_ids

print("reload stuff module")

import neurora.rsa_plot

'''
虽然这是不经济的，但是还是搬用这个方法试一试，这是你现在手里能找到的唯一的方法了。
'''

import sys
print(sys.executable)
import os
import time
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
from scipy.spatial.distance import pdist, squareform
from mne_rsa import searchlight


# model_formula = 'distance ~ category + logic1 + RT1 + (1|subId)'
model_formula = 'distance ~ category_cond2 + logicalScore1_cond2 + RT1_cond2 + (1|subId_cond1)+ (1|wordPairs)'



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

        # useless?
        # epochs_sub.equalize_event_counts(method='mintime')  # 和最少试次的条件对齐

        # 将读取的epochs对象添加到列表中
        list_epochs_allSubs.append(epochs_sub)

# utils_EEG/stuff.py
'''
update 2024年10月6日

for testing requirement (the distance calculation is too slow)
add a new parameter num_epochs to limit the number of epochs loaded
'''
def load_epochs(sub_ids, list_epochs_allSubs, base_data_path, channels_field,
                num_epochs=None):
    '''
    Load epochs data for a list of subjects and optionally limit the number of epochs for testing.

    Parameters:
    - sub_ids (list): List of subject IDs.
    - list_epochs_allSubs (list): List to append the loaded epochs objects.
    - base_data_path (str): Base path to the directory containing the epochs files.
    - channels_field (list): List of channel names to pick from the epochs.
    - num_epochs (int, optional): Number of epochs to load per subject. If None, load all epochs.
    '''
    # print num_epochs
    print("分界线")
    print("num_epochs:", num_epochs)

    for sub_id in sub_ids:
        # Construct the file path for the subject's epochs file
        data_path = os.path.join(base_data_path, f"{sub_id}-epo.fif")

        # Read the epochs data
        epochs_sub = mne.read_epochs(fname=data_path, preload=True)

        # Pick the specified channels
        epochs_sub = epochs_sub.pick(picks=channels_field)

        # Limit the number of epochs if num_epochs is specified
        if num_epochs is not None:
            epochs_sub = epochs_sub[:num_epochs]

        # Convert data to float32
        # avoid memory problem in parallel processing.
        epochs_sub._data = epochs_sub._data.astype(np.float32)

        # Append the epochs object to the list
        list_epochs_allSubs.append(epochs_sub)

'''
replace epochs data with power data (frequency)
'''
def load_epochs_tfr(sub_ids, list_epochs_allSubs, base_data_path, channels_field, freqs, n_cycles, decim=1, average=False):
    '''
    读取epochs数据，并将读取的epochs对象添加到列表中。

    参数:
    sub_ids (list): 包含所有被试的ID的列表。
    list_epochs_allSubs (list): 包含所有被试的epochs对象的列表。
    base_data_path (str): 数据存放路径的前缀。
    channels_field (list): 包含所有被试的通道名称的列表。
    freqs (array): 要分析的频率数组。
    n_cycles (int or array): 每个频率的周期数，可以是一个常数或一个数组。
    decim (int): 下采样因子，默认值为3。
    average (bool): 是否对每个epoch的TFR结果求平均，默认值为False。
    '''

    for sub_id in sub_ids:
        # 为每个sub_id更新data_path
        data_path = base_data_path + sub_id + '-epo.fif'

        # 读取epochs数据
        epochs_sub = mne.read_epochs(fname=data_path, preload=True)

        # 选择特定的通道
        epochs_sub = epochs_sub.pick(picks=channels_field)

        # 和最少试次的条件对齐
        epochs_sub.equalize_event_counts(method='mintime')

        # 计算TFR
        power = mne.time_frequency.tfr_morlet(epochs_sub, freqs=freqs, n_cycles=n_cycles,
                                              use_fft=True, return_itc=False, decim=decim, n_jobs=1, average=average)

        # 提取功率数据
        power_data = power.data

        # 平均频率维度的功率数据
        averaged_power_data = power_data.mean(axis=2)  # Shape will be (n_epochs, n_channels, n_times)

        # Copy original epochs and replace the data
        new_epochs = epochs_sub.copy()
        new_epochs._data = averaged_power_data

        # 将新的epochs对象添加到列表中
        list_epochs_allSubs.append(new_epochs)


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


def generate_channel_patches(file_path: str, spatial_radius: float) -> dict:
    # Load the epoch data for the searchlight analysis
    # exclude eog!!!!
    epochs = mne.read_epochs(file_path, preload=True, )  # include eog!!!!
    epochs.drop_channels(['VEOG'])

    # Get data shape
    shape = epochs.get_data().shape

    # Create a distance matrix using the channel locations
    picks = mne.pick_types(epochs.info, meg=True, eeg=True, eog=False, stim=False)
    pos = [epochs.info['chs'][pick]['loc'][:3] for pick in picks]
    dist = squareform(pdist(pos))

    # Initialize the searchlight generator
    sl = searchlight(shape=shape, dist=dist, spatial_radius=spatial_radius, temporal_radius=None)

    # Generate the patches from the searchlight generator
    field_dict = {}
    for index, patch in zip(picks, sl):
        central_channel = epochs.ch_names[index]
        surrounding_channels = [epochs.ch_names[idx] for idx in patch[1]]
        field_dict[central_channel] = surrounding_channels

    return field_dict


def organize_eeg_data_by_subject_and_condition(T1_sub_ids, list_epochs_One, list_epochs_M2, list_epochs_S2,
                                               field_channels):
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
    num_channel = len(field_channels)
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

'''
2024年10月6日 update:
I don't need to equalize the trials between 3 conditions.
because, I only need to calculate distances between trials of different conditions.

The mix-effect model don't require same numbers of values for calculation.

so the function below 
'''

def organize_eeg_data_by_subject_and_condition(T1_sub_ids, list_epochs_One, list_epochs_M2, list_epochs_S2,
                                               channels_all, equalize_trials=False):
    """
    Organizes EEG data for multiple subjects across multiple conditions, handling varying numbers of trials per condition.

    Parameters:
        T1_sub_ids (list): List of subject identifiers.
        list_epochs_One (list): List of Epochs objects for each subject under the first condition.
        list_epochs_M2 (list): List of Epochs objects for each subject under the second condition.
        list_epochs_S2 (list): List of Epochs objects for each subject under the third condition.
        channels_all (list): List of all EEG channels to include.
        equalize_trials (bool): If True, equalizes the number of trials across conditions by random sampling.

    Returns:
        list_subdata (list): A list where each element corresponds to one subject.
            Each subject's data is a list of arrays, one for each condition.
            Each condition array has shape (n_trials, n_channels, n_times),
            where n_trials may be equalized if equalize_trials is True.
    """
    import numpy as np

    num_channel = len(channels_all)
    list_subdata = []

    for sub_index in range(len(T1_sub_ids)):
        # List to hold data arrays for all conditions for a subject
        subdata = []
        trial_counts = []

        # Collect data and trial counts for each condition
        for epochs_list in [list_epochs_One, list_epochs_M2, list_epochs_S2]:
            epochs = epochs_list[sub_index]
            data = epochs.get_data(picks='eeg')  # Shape: (n_trials, n_channels, n_times)
            subdata.append(data)
            trial_counts.append(data.shape[0])

        if equalize_trials:
            # Find the minimum number of trials across conditions
            min_trials = min(trial_counts)
            equalized_subdata = []
            for data in subdata:
                n_trials = data.shape[0]
                if n_trials > min_trials:
                    # Randomly select min_trials trials
                    indices = np.random.choice(n_trials, min_trials, replace=False)
                    equalized_data = data[indices]
                else:
                    equalized_data = data  # No change needed
                equalized_subdata.append(equalized_data)
            subdata = equalized_subdata

        list_subdata.append(subdata)

    return list_subdata

'''
update 2024年10月6日

1. Modify organize_eeg_data_by_subject_and_condition to Include Metadata
First, we'll update organize_eeg_data_by_subject_and_condition to extract and return the metadata associated with each epoch. This way, we can use the metadata in subsequent functions.
Updated Function

Explanation
Return Structure:
Instead of returning just the data, we now return a list where each element (per subject) is a dictionary containing:
'data': A list of data arrays for each condition.
'metadata': A list of DataFrames for each condition, containing the metadata for each trial.
Extracting Metadata:
We check if epochs.metadata is available. If so, we reset the index to ensure proper alignment and append it to the metadata list.
Handling Missing Metadata:
If metadata is not available, we create a placeholder DataFrame with a 'trial_index' column.
'''
def organize_eeg_data_by_subject_and_condition(T1_sub_ids, list_epochs_One, list_epochs_M2, list_epochs_S2,
                                               channels_all):
    """
    Organizes EEG data for multiple subjects across multiple conditions, including metadata.

    Parameters:
        T1_sub_ids (list): List of subject identifiers.
        list_epochs_One (list): List of Epochs objects for each subject under the first condition.
        list_epochs_M2 (list): List of Epochs objects for each subject under the second condition.
        list_epochs_S2 (list): List of Epochs objects for each subject under the third condition.
        channels_all (list): List of all EEG channels.

    Returns:
        list_subdata (list): List of dictionaries for each subject, each containing:
            - 'data': List of arrays for each condition, with shape (n_trials, n_channels, n_timepoints).
            - 'metadata': List of DataFrames for each condition, containing metadata for each trial.
    """
    num_channel = len(channels_all)
    list_subdata = []

    for sub_index in range(len(T1_sub_ids)):
        subdata = {'data': [], 'metadata': []}

        for epochs_list in [list_epochs_One, list_epochs_M2, list_epochs_S2]:
            epochs = epochs_list[sub_index]
            # Get EEG data
            data = epochs.get_data(picks='eeg')  # Shape: (n_trials, n_channels, n_timepoints)
            subdata['data'].append(data)

            # Get metadata
            if epochs.metadata is not None:
                metadata = epochs.metadata.reset_index(drop=True)
                subdata['metadata'].append(metadata)
            else:
                # If no metadata is available, create a placeholder DataFrame
                n_trials = data.shape[0]
                metadata = pd.DataFrame({'trial_index': range(n_trials)})
                subdata['metadata'].append(metadata)

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

'''
why don't use package of python?

'''
# Ensure automatic conversion between Pandas DataFrame and R DataFrame
# extract it out from function to avoid run repeatedly.
pandas2ri.activate()
def fit_lmer_model_R(data, formula=None):
    print("R version")

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
    # extract it out from function to avoid run repeatedly.
    # pandas2ri.activate()
    # Set R's temporary directory to another drive
    # decrease the usage of C drive
    ro.r('Sys.setenv(TMPDIR = "D:/temp")')

    # Convert the Pandas DataFrame to an R DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)

    # Pass the data to the R environment
    ro.globalenv['r_data'] = r_data
    #
    # Fit the model in R
    # no hat, so return t directly.
    x = ro.r('''
    library(lmerTest)
    fit <- lmer(''' + formula + ''', r_data)
    summary(fit)$coefficients["category_cond2S", c("t value", "Pr(>|t|)")]
    ''')
###################################################################################
    # # Fit the model in R and extract coefficient estimate and standard error
    # # I want to calculate the t value using python
    # # if the result is same with R, then I can use python to hat t value
    # r_output = ro.r(f'''
    #     library(lmerTest)
    #     fit <- lmer({formula}, r_data)
    #     model_summary <- summary(fit)
    #     coef_estimate <- model_summary$coefficients["categoryS", "Estimate"]
    #     std_error <- model_summary$coefficients["categoryS", "Std. Error"]
    #     t_value <- model_summary$coefficients["categoryS", "t value"]
    #     p_value <- model_summary$coefficients["categoryS", "Pr(>|t|)"]
    #     c(coef_estimate, std_error, t_value, p_value)
    #     ''')
    # # Extract values from R output
    # coef_estimate = r_output[0]
    # std_error = r_output[1]
    # r_t_value = r_output[2]
    # p_value = r_output[3]
    #
    # # Calculate the t-value using Python
    # python_t_value = coef_estimate / std_error
    #
    # # Print the comparison results
    # print(f"R-calculated t-value: {r_t_value}, Python-calculated t-value: {python_t_value}")
    # print(f"R-calculated p-value: {p_value}")
    ###################################################################################

    return x[0], x[1]
'''
for hat+tfce clustering

I need to add sigma parameter ..

It is hard to add hat to mixmodel ... 
this function is not right need to change it...


the difference from former function is that 
the return value is coef_estimate, std_error, but not t_value, p_value
'''

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

def fit_lmer_model_R_hat(data, formula=None, term='category_cond2S'):
    """
    Fits a linear mixed effects model to the data using R's lmerTest package, with proportional variance adjustment
    to avoid over-optimistic statistics in low-variance regions.

    Parameters:
    - data: Pandas DataFrame containing the data.
    - model_formula: A string representing the model formula in R's syntax.
    - term: The term of interest for which t and p values are returned.
    
    Returns:
    - The t-value and p-value for the specified term.
    """

    # print("R version with proportional variance adjustment")
    # Drop rows with missing values in place to avoid copying the data
    data.dropna(inplace=True)

    # Set R's temporary directory to another drive to save memory
    ro.r('Sys.setenv(TMPDIR = "D:/temp")')

    # Convert the Pandas DataFrame to an R DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)

    # Pass the data to the R environment
    ro.globalenv['r_data'] = r_data

    # # Use efficient conversion and avoid keeping unnecessary data in memory
    # r_output = ro.r(f'''
    #         library(lmerTest)
    #         fit <- lmer({formula}, r_data)
    #         model_summary <- summary(fit)
    #         coef_estimate <- model_summary$coefficients["categoryS", "Estimate"]
    #         std_error <- model_summary$coefficients["categoryS", "Std. Error"]
    #         t_value <- model_summary$coefficients["categoryS", "t value"]
    #         p_value <- model_summary$coefficients["categoryS", "Pr(>|t|)"]
    #         c(coef_estimate, std_error, t_value, p_value)
    #         ''')

    # r_output = ro.r(f'''
    #         library(lmerTest)
    #         fit <- lmer({formula}, r_data)
    #         model_summary <- summary(fit)
    #         coef_estimate <- model_summary$coefficients["categoryS", "Estimate"]
    #         std_error <- model_summary$coefficients["categoryS", "Std. Error"]
    #         t_value <- model_summary$coefficients["categoryS", "t value"]
    #         p_value <- model_summary$coefficients["categoryS", "Pr(>|t|)"]
    #         c(coef_estimate, std_error, t_value, p_value)
    #         ''')

    # Construct the R code, inserting the term variable
    r_output = ro.r(f'''
            library(lmerTest)
            fit <- lmer({formula}, r_data)
            model_summary <- summary(fit)
            coef_estimate <- model_summary$coefficients["{term}", "Estimate"]
            std_error <- model_summary$coefficients["{term}", "Std. Error"]
            t_value <- model_summary$coefficients["{term}", "t value"]
            p_value <- model_summary$coefficients["{term}", "Pr(>|t|)"]
            c(coef_estimate, std_error, t_value, p_value)
            ''')

    # Extract values from R output
    coef_estimate = r_output[0]
    std_error = r_output[1]
    # r_t_value = r_output[2]
    # p_value = r_output[3]



    return coef_estimate, std_error

'''
oudated method to smooth t value
try a traditional way to smooth t value
easily to inflate variance in low-variance regions
'''
def fit_lmer_model_inflate(data, formula=None, inflate_variance=True, variance_threshold=1e-8):
    """
    Fits a linear mixed effects model to the data using R's lmerTest package, with variance adjustments to avoid over-optimistic statistics in low-variance regions.

    Parameters:
    - data: Pandas DataFrame containing the data.
    - formula: A string representing the model formula in R's syntax.
    - inflate_variance: Boolean indicating whether to inflate low variance regions. Default is False.
    - variance_threshold: Threshold below which variance inflation is applied. Default is 1e-8.

    Returns:
    - The summary of the fitted model as an R object.
    """

    print("R version with variance adjustment")

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Optionally inflate variance in low-variance regions
    if inflate_variance:
        # Calculate variance for each column
        variance = data.var()
        low_variance_cols = variance[variance < variance_threshold].index
        if not low_variance_cols.empty:
            print(f"Inflating variance for columns: {list(low_variance_cols)}")
            # Add a small constant to low-variance columns
            data[low_variance_cols] += np.random.normal(0, np.sqrt(variance_threshold),
                                                        size=data[low_variance_cols].shape)

    # Set R's temporary directory to another drive to save memory
    ro.r('Sys.setenv(TMPDIR = "D:/temp")')

    # Convert Pandas DataFrame to R DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)

    # Pass the data to the R environment
    ro.globalenv['r_data'] = r_data

    # # Fit the model in R
    # fit_result = ro.r(f'''
    # library(lmerTest)
    # fit <- lmer({formula}, r_data)
    # summary(fit)
    # ''')

    # Fit the model in R
    x = ro.r('''
        library(lmerTest)
        fit <- lmer(''' + formula + ''', r_data)
        summary(fit)$coefficients["categoryS", c("t value", "Pr(>|t|)")]
        ''')

    # Deactivate the Pandas to R DataFrame conversion
    # pandas2ri.deactivate()

    print(x)

    return x[0], x[1]



# Example usage:
# fit_lmer_model(data, formula="response ~ category + (1|subject)", inflate_variance=True, variance_threshold=1e-6)


'''
I need df(ziyoudu) to calculate t-threshold
'''
import scipy.stats
def fit_lmer_model_t_threshold(data, formula, alpha=0.05):
    """
    Calculates the critical t-threshold for a given alpha level using R's lmerTest package.

    Parameters:
    - data: Pandas DataFrame containing the data.
    - formula: A string representing the model formula in R's syntax.
    - alpha: Significance level for the threshold.

    Returns:
    - t-threshold value.
    """
    # Fit the model to get the degrees of freedom
    data.dropna(inplace=True)  # Drop rows with NAs

    # Ensure automatic conversion between Pandas DataFrame and R DataFrame
    pandas2ri.activate()

    # Convert the Pandas DataFrame to an R DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)

    # Pass the data to the R environment
    ro.globalenv['r_data'] = r_data

    # Fit the model in R and get the degrees of freedom
    result = ro.r('''
    library(lmerTest)
    fit <- lmer(''' + formula + ''', r_data)
    summary(fit)$coefficients["category_cond2S", c("df")]
    ''')

    # Deactivate the Pandas to R DataFrame conversion
    pandas2ri.deactivate()

    # Extract degrees of freedom
    df = result[0]

    # Calculate the t-threshold
    t_threshold = scipy.stats.t.ppf(1 - alpha, df)

    return t_threshold

'''
the former version basing on R is too slow, so I need to change it to python version

'''
import statsmodels.formula.api as smf
import re
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats

'''
maybe this is wrong..
'''
def fit_lmer_model_py(data, formula=None):
    data.dropna(inplace=True)
    model = smf.mixedlm(formula, data, groups=data["subId"])
    result = model.fit()
    # print(result.summary())
    t_value = result.tvalues["category[T.S]"]
    p_value = result.pvalues["category[T.S]"]
    return t_value, p_value

"""
python version

the grammer of formula is different from R's lmer

so we need to parse the formula

problem: cannot convertgent in some point. 
"""
def fit_lmer_model_py(data, formula=None):
    print("python version")

    """
    Fits a linear mixed effects model to the data using statsmodels' MixedLM.

    Parameters:
    - data: Pandas DataFrame containing the data.
    - formula: A string representing the model formula in R's syntax.

    Returns:
    - t-value and p-value for the coefficient 'category'.
    """

    # Drop rows with NAs
    data = data.dropna()

    # Parse the formula to extract fixed effects, random effects, and grouping variable
    def parse_formula(formula):
        """
        Parses an R-style formula into fixed effects formula, random effects formula, and grouping variable.
        """
        # Split the formula into left and right parts
        left_right = formula.split('~')
        if len(left_right) != 2:
            raise ValueError("Formula must have a left and right side separated by '~'")
        response = left_right[0].strip()
        right = left_right[1].strip()

        # Use regex to find random effects terms
        # Random effects are specified as (random_effects | group)
        random_effects_matches = re.findall(r'\(([^|]+)\|([^)]+)\)', right)
        if not random_effects_matches:
            raise ValueError("Random effects not specified in the formula in the format '(re|group)'.")

        # Extract random effects formula and groups
        re_terms = []
        group_vars = []
        for re_part, group_var in random_effects_matches:
            re_terms.append(re_part.strip())
            group_vars.append(group_var.strip())

        # Remove random effects terms from the right side of the formula to get fixed effects
        fixed_right = re.sub(r'\([^|]+\|[^)]+\)', '', right)
        fixed_right = re.sub(r'\s+', ' ', fixed_right)  # Replace multiple spaces with single space
        fixed_right = fixed_right.strip()
        fixed_right = re.sub(r'^\+', '', fixed_right)  # Remove leading '+'
        fixed_right = re.sub(r'\+$', '', fixed_right)  # Remove trailing '+'
        fixed_right = fixed_right.strip()

        fixed_formula = response + ' ~ ' + fixed_right

        # For simplicity, assume only one group variable
        if len(set(group_vars)) > 1:
            raise ValueError("Multiple grouping variables detected. This function supports only one grouping variable.")

        groups = group_vars[0]

        # Random effects formula
        # For simplicity, we take the first random effects term
        re_formula = re_terms[0]

        return fixed_formula, re_formula, groups

    # Parse the formula
    fixed_formula, re_formula, group_var = parse_formula(formula)

    # Fit the model using statsmodels
    model = smf.mixedlm(fixed_formula, data=data, groups=data[group_var], re_formula=re_formula)
    result = model.fit()

    # Check if 'category[T.S]' exists in the result's parameters
    if 'category[T.S]' in result.params.index:
        tvalue = result.tvalues['category[T.S]']  # Use the exact name of the dummy variable
        pvalue = result.pvalues['category[T.S]']
    else:
        raise ValueError("Coefficient 'category[T.S]' not found in the model parameters.")

    print("t-value:", tvalue)
    print("p-value:", pvalue)

    return tvalue, pvalue

def fit_lmer_model_t_threshold_py(data, formula, alpha=0.05):
    """
    Calculates the critical t-threshold for a given alpha level using statsmodels' MixedLM.

    Parameters:
    - data: Pandas DataFrame containing the data.
    - formula: A string representing the model formula in R's syntax.
    - alpha: Significance level for the threshold.

    Returns:
    - t-threshold value.
    """
    # Drop rows with NAs
    data = data.dropna()

    # Parse the formula to extract fixed effects, random effects, and grouping variable
    def parse_formula(formula):
        """
        Parses an R-style formula into fixed effects formula, random effects formula, and grouping variable.
        """
        # Split the formula into left and right parts
        left_right = formula.split('~')
        if len(left_right) != 2:
            raise ValueError("Formula must have a left and right side separated by '~'")
        response = left_right[0].strip()
        right = left_right[1].strip()

        # Use regex to find random effects terms
        random_effects_matches = re.findall(r'\(([^|]+)\|([^)]+)\)', right)
        if not random_effects_matches:
            raise ValueError("Random effects not specified in the formula in the format '(re|group)'.")
        
        # Extract random effects formula and groups
        re_terms = []
        group_vars = []
        for re_part, group_var in random_effects_matches:
            re_terms.append(re_part.strip())
            group_vars.append(group_var.strip())

        # Remove random effects terms from the right side of the formula to get fixed effects
        fixed_right = re.sub(r'\([^|]+\|[^)]+\)', '', right)
        fixed_right = re.sub(r'\s+', ' ', fixed_right)  # Replace multiple spaces with single space
        fixed_right = fixed_right.strip()
        fixed_right = re.sub(r'^\+', '', fixed_right)  # Remove leading '+'
        fixed_right = re.sub(r'\+$', '', fixed_right)  # Remove trailing '+'
        fixed_right = fixed_right.strip()

        fixed_formula = response + ' ~ ' + fixed_right

        # For simplicity, assume only one group variable
        if len(set(group_vars)) > 1:
            raise ValueError("Multiple grouping variables detected. This function supports only one grouping variable.")

        groups = group_vars[0]

        # Random effects formula
        # For simplicity, we take the first random effects term
        re_formula = re_terms[0]

        return fixed_formula, re_formula, groups

    # Parse the formula
    fixed_formula, re_formula, group_var = parse_formula(formula)

    # Fit the model using statsmodels
    model = smf.mixedlm(fixed_formula, data=data, groups=data[group_var], re_formula=re_formula)
    result = model.fit()

    # Attempt to extract degrees of freedom
    # Note: statsmodels MixedLM does not provide degrees of freedom in the same way as lmerTest
    # Here we attempt to approximate degrees of freedom based on the number of groups and observations

    # Number of observations
    n_obs = model.nobs

    # Number of groups
    n_groups = data[group_var].nunique()

    # Degrees of freedom approximation
    # This is a simplistic approximation and may not be accurate
    df_approx = n_groups - 1

    # Alternatively, for fixed effects, you might use:
    df_resid = n_obs - result.params.size

    # However, this does not account for the complexity of random effects

    # Calculate the t-threshold using the approximated degrees of freedom
    t_threshold = scipy.stats.t.ppf(1 - alpha, df=df_approx)

    print("Approximated degrees of freedom:", df_approx)
    print("t-threshold for alpha =", alpha, ":", t_threshold)

    return t_threshold

'''
you have to change the function to fit the new data structure
t_value, _ = fit_lmer_model(df, model_formula)
t_value, _ = fit_lmer_model_py(df, model_formula)
'''
# Loop over the ndarray and apply the model
# add a progress bar.
from tqdm import tqdm  # Import the tqdm library
def apply_lmer_models_to_array(array_of_dfs, model_formula = 'distance ~ category + logic1 + RT1 + (1|subId)'):
    """
    Applies a linear mixeAd-effects model to each DataFrame stored in a 1D ndarray.

    Parameters:
        5733 = times(91) * channels(63) flatten
    - array_of_dfs: ndarray, shape (1, 5733) containing pandas DataFrames.
    - model_formula: str, formula to be used in the linear mixed-effects model.

    Returns:
    - ndarray of shape (5733,) containing the t-values from the model fitting.
    """
    # Initialize an array to store the t-values, array_of_dfs has shape (1, 5733)
    results = np.zeros(array_of_dfs.shape[1], dtype=float)  # Shape (5733,)

    # Loop over the second dimension of the array since it's now 1D in usage
    # for j in range(array_of_dfs.shape[1]):
    # Loop over the second dimension of the array with a progress bar
    for j in tqdm(range(array_of_dfs.shape[1]), desc="Processing DataFrames"):
        # Extract the DataFrame at the current index
        df = array_of_dfs[0, j]  # Access the DataFrame
        try:
            t_value, _ = fit_lmer_model_R(data = df, formula = model_formula)
            results[j] = t_value
        except Exception as e:
            print(f"Error processing DataFrame at index {j}: {e}")
            results[j] = np.nan  # Use NaN for failed model fittings

    return results

def apply_lmer_models_to_array_hat(
        array_of_dfs,
        model_formula = 'distance ~ category + logic1 + RT1 + (1|subId)',
        sigma = 0.001
                                   ):
    """
    Applies a linear mixeAd-effects model to each DataFrame stored in a 1D ndarray.

    Parameters:
        5733 = times(91) * channels(63) flatten
    - array_of_dfs: ndarray, shape (1, 5733) containing pandas DataFrames.
    - model_formula: str, formula to be used in the linear mixed-effects model.

    Returns:
    - ndarray of shape (5733,) containing the t-values from the model fitting.
    """
    # Initialize an array to store the coef_estimate, std_error, array_of_dfs has shape (1, 5733)
    results_t_smooth = np.zeros(array_of_dfs.shape[1], dtype=float)
    results_coef_estimate = np.zeros(array_of_dfs.shape[1], dtype=float)  # Shape (5733,)
    results_std_error = np.zeros(array_of_dfs.shape[1], dtype=float)  # Shape (5733,)

    # Loop over the second dimension of the array since it's now 1D in usage
    # for j in range(array_of_dfs.shape[1]):
    # Loop over the second dimension of the array with a progress bar
    for j in tqdm(range(array_of_dfs.shape[1]), desc="Processing DataFrames"):
        # Extract the DataFrame at the current index
        df = array_of_dfs[0, j]  # Access the DataFrame
        try:
            coef_estimate, std_error = fit_lmer_model_R_hat(data = df, formula = model_formula)
            results_coef_estimate[j] = coef_estimate
            results_std_error[j] = std_error
        except Exception as e:
            print(f"Error processing DataFrame at index {j}: {e}")
            results_coef_estimate[j] = np.nan  # Use NaN for failed model fittings
    max_std_error = np.max(results_std_error)

    results_t_original = results_coef_estimate / (results_std_error+max_std_error*sigma)
    results_t_smooth = results_coef_estimate / (results_std_error+max_std_error*sigma)

    # Store the results outside for comparison
    np.save('results_t_original.npy', results_t_original)
    np.save('results_t_smooth.npy', results_t_smooth)

    return results_t_smooth


'''
refine: 
add a parameter to limit the number of iterations for testing
add try except to capture keyboard interrupt(always unworkable in last version)
'''

def apply_lmer_models_to_array_hat(
        array_of_dfs,
        model_formula=model_formula,
        sigma=0.001,
        test_iterations=10  # Add a parameter to limit the number of iterations for testing
):
    """
    Applies a linear mixed-effects model to each DataFrame stored in a 1D ndarray.

    Parameters:
        5733 = times(91) * channels(63) flatten
    - array_of_dfs: ndarray, shape (1, 5733) containing pandas DataFrames.
    - model_formula: str, formula to be used in the linear mixed-effects model.
    - sigma: float, smoothing parameter.
    - test_iterations: int, number of iterations to run for testing.

    Returns:
    - ndarray of shape (5733,) containing the t-values from the model fitting.
    """
    # Initialize an array to store the coef_estimate, std_error, array_of_dfs has shape (1, 5733)
    results_t_smooth = np.zeros(array_of_dfs.shape[1], dtype=float)
    results_coef_estimate = np.zeros(array_of_dfs.shape[1], dtype=float)  # Shape (5733,)
    results_std_error = np.zeros(array_of_dfs.shape[1], dtype=float)  # Shape (5733,)

    # Loop over the second dimension of the array since it's now 1D in usage
    for j in tqdm(range(min(test_iterations, array_of_dfs.shape[1])), desc="Processing DataFrames"):
        # Check for interruption
        try:
            # Extract the DataFrame at the current index
            df = array_of_dfs[0, j]  # Access the DataFrame
            try:
                coef_estimate, std_error = fit_lmer_model_R_hat(data=df, formula=model_formula)
                results_coef_estimate[j] = coef_estimate
                results_std_error[j] = std_error
            except Exception as e:
                print(f"Error processing DataFrame at index {j}: {e}")
                results_coef_estimate[j] = np.nan  # Use NaN for failed model fittings
        except KeyboardInterrupt:
            print("Process interrupted by user.")
            break

    max_std_error = np.max(results_std_error)

    results_t_original = results_coef_estimate / results_std_error
    results_t_smooth = results_coef_estimate / (results_std_error + max_std_error * sigma)

    # Store the results outside for comparison
    np.save('results_t_original.npy', results_t_original)
    np.save('results_t_smooth.npy', results_t_smooth)

    return results_t_smooth

# Example usage with a limited number of iterations for testing
# array_of_dfs = ...  # Your ndarray of DataFrames
# results = apply_lmer_models_to_array_hat(array_of_dfs, test_iterations=10)

# parallel version
'''
you have to change the function to fit the new data structure
t_value, _ = fit_lmer_model(df, model_formula)
t_value, _ = fit_lmer_model_py(df, model_formula)
'''
'''
add some logging to the process_single_dataframe function to 
capture more information about the DataFrame being processed 
and any exceptions that might occur. 
之前出现过不收敛得情况 python..
'''
def process_single_dataframe(df, model_formula):
    """Helper function to process a single DataFrame."""
    try:
        # Log the shape and first few rows of the DataFrame
        print(f"Processing DataFrame with shape: {df.shape}")
        print(df.head())

        # t_value, _ = fit_lmer_model(df, model_formula)
        print("python version")
        t_value, _ = fit_lmer_model_R(df, model_formula) # R version
        # t_value, _ = fit_lmer_model_py(df, model_formula)  # if python version mix model func
        return t_value
    except Exception as e:
        # Log the exception and the DataFrame that caused it
        print(f"Error processing DataFrame: {e}")
        print(f"DataFrame that caused the error:\n{df}")
        return np.nan


from joblib import Parallel, delayed
import os
# Calculate the number of cores to use, leaving 2 cores free
total_cores = os.cpu_count()
n_jobs = max(1, total_cores - 3)  # Ensure at least 2 core is used
# 设置 JOBLIB_TEMP_FOLDER 环境变量 to avoid too much C: drive usage
'''
log the index and DataFrame that caused the NaN value:
'''

def apply_lmer_models_to_array_parallel(array_of_dfs, model_formula = model_formula, n_jobs=n_jobs):
    """
    Applies a linear mixed-effects model to each DataFrame stored in a 1D ndarray in parallel.

    Parameters:
        5733 = times(91) * channels(63) flatten
    - array_of_dfs: ndarray, shape (1, 5733) containing pandas DataFrames.
    - model_formula: str, formula to be used in the linear mixed-effects model.
    - n_jobs: int, number of parallel jobs. -1 uses all available cores.

    Returns:
    - ndarray of shape (5733,) containing the t-values from the model fitting.
    """
    print('parallel')
    test_iterations = 100
    # Initialize an array to store the t-values, array_of_dfs has shape (1, 5733)
    results = Parallel(n_jobs=n_jobs, temp_folder=r'D:\LYW\buff4parallel')(
        delayed(process_single_dataframe)(array_of_dfs[0, j], model_formula)
        # for j in tqdm(range(array_of_dfs.shape[1]), desc="Processing DataFrames")
        for j in tqdm(range(min(test_iterations, array_of_dfs.shape[1])), desc="Processing DataFrames")
    )

    # Check for NaN values and log the index and DataFrame
    for idx, result in enumerate(results):
        if np.isnan(result):
            print(f"NaN value found at index {idx}")
            print(f"DataFrame at index {idx}:\n{array_of_dfs[0, idx]}")
            # Export the DataFrame to an Excel file
            df_with_nan = array_of_dfs[0, idx]
            df_with_nan.to_excel(f"DataFrame_with_NaN_at_index_{idx}.xlsx", index=False)
            print(f"DataFrame with NaN exported to DataFrame_with_NaN_at_index_{idx}.xlsx")

    return np.array(results)
'''
_hat  parallel version

To align your parallel version with the _hat function and address the convergence issues, you need to modify the apply_lmer_models_to_array_parallel function to:

Return coef_estimate and std_error from process_single_dataframe, instead of just t_value.
Collect these values in arrays, similar to how it's done in the _hat function.
Compute results_t_original and results_t_smooth using the formula from the _hat function.
Add logging to capture more information about each DataFrame and any exceptions.
'''
def process_single_dataframe_hat(df, model_formula):
    """Helper function to process a single DataFrame."""
    try:
        # Log the shape and first few rows of the DataFrame
        print(f"Processing DataFrame with shape: {df.shape}")
        # print(df.head())

        # Compute coef_estimate and std_error using fit_lmer_model_R_hat
        coef_estimate, std_error = fit_lmer_model_R_hat(
            data=df, formula=model_formula, term='category_cond2S'
        )
        # coef_estimate, _ = fit_lmer_model_R_hat(data=df, formula=model_formula)
        return coef_estimate, std_error
    except Exception as e:
        # Log the exception and the DataFrame that caused it
        print(f"Error processing DataFrame: {e}")
        print(f"DataFrame that caused the error:\n{df}")
        return np.nan, np.nan

def apply_lmer_models_to_array_parallel_hat(array_of_dfs,
                                            model_formula= model_formula,
                                            sigma=0.001,
                                            n_jobs=n_jobs):
    """
    Applies a linear mixed-effects model to each DataFrame stored in a 1D ndarray in parallel.

    Parameters:
        - array_of_dfs: ndarray, shape (1, 5733) containing pandas DataFrames.
        - model_formula: str, formula to be used in the linear mixed-effects model.
        - sigma: float, smoothing parameter as in apply_lmer_models_to_array_hat
        - n_jobs: int, number of parallel jobs.

    Returns:
    - ndarray of shape (5733,) containing the t-values from the model fitting.
    """
    print('Starting parallel processing')

    # Use Parallel to process DataFrames in parallel
    # Collect results as list of tuples (coef_estimate, std_error)
    results = Parallel(n_jobs=n_jobs, temp_folder=r'D:\LYW\buff4parallel')(
        delayed(process_single_dataframe_hat)(array_of_dfs[0, j], model_formula)
        for j in tqdm(range(array_of_dfs.shape[1]), desc="Processing DataFrames")
    )

    # Initialize arrays to store the results
    coef_estimates = np.zeros(array_of_dfs.shape[1], dtype=float)
    std_errors = np.zeros(array_of_dfs.shape[1], dtype=float)

    # Unpack results into separate arrays
    for idx, (coef_estimate, std_error) in enumerate(results):
        if np.isnan(coef_estimate) or np.isnan(std_error):
            print(f"NaN value found at index {idx}")
            print(f"DataFrame at index {idx}:\n{array_of_dfs[0, idx]}")
            # Export the DataFrame to an Excel file for inspection
            df_with_nan = array_of_dfs[0, idx]
            df_with_nan.to_excel(f"DataFrame_with_NaN_at_index_{idx}.xlsx", index=False)
            print(f"DataFrame with NaN exported to DataFrame_with_NaN_at_index_{idx}.xlsx")
            coef_estimates[idx] = np.nan
            std_errors[idx] = np.nan
        else:
            coef_estimates[idx] = coef_estimate
            std_errors[idx] = std_error

    # Compute max_std_error while ignoring NaNs
    max_std_error = np.nanmax(std_errors)
    print(f"Max std_error (ignoring NaNs): {max_std_error}")

    # Compute t-values, adding smoothing to avoid division by zero
    denominator = std_errors + max_std_error * sigma
    results_t_original = coef_estimates / std_errors
    results_t_smooth = coef_estimates / denominator

    # Save the results
    np.save('results_t_original.npy', results_t_original)
    np.save('results_t_smooth.npy', results_t_smooth)
    print('Results saved to results_t_original.npy and results_t_smooth.npy')

    return results_t_smooth

'''

 refine the parallel version of your function 
 to include a limit on the number of iterations for testing 
 and to handle keyboard interrupts gracefully.

'''
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed

model_formula = \
    'distance ~ category_cond2 + logicalScore1_cond2 + RT1_cond2 + (1|subId_cond1) + (1|wordPairs)'
# model_formula='distance ~ category + logic1 + RT1 + (1|subId)',

def apply_lmer_models_to_array_parallel_hat(array_of_dfs,
                                            model_formula= model_formula
                                    # for formal calculation, set test_iterations greater than n_times*n_channels(91 * 63 = 5733 for example/)
                                            ):
    """
    Applies a linear mixed-effects model to each DataFrame stored in a 1D ndarray in parallel.

    Parameters:
        - array_of_dfs: ndarray, shape (1, 5733) containing pandas DataFrames.
        - model_formula: str, formula to be used in the linear mixed-effects model.
        - sigma: float, smoothing parameter as in apply_lmer_models_to_array_hat
        - n_jobs: int, number of parallel jobs.
        - test_iterations: int, number of iterations to run for testing.

    Returns:
    - ndarray of shape (5733,) containing the t-values from the model fitting.
    """
    print('Starting parallel processing 15点35分')
    sigma = 0.0001
    test_iterations = 100000

    # Use Parallel to process DataFrames in parallel
    # Collect results as list of tuples (coef_estimate, std_error)
    # try:
    #     results = Parallel(n_jobs=1, temp_folder=r'D:\LYW\buff4parallel')(
    #         delayed(process_single_dataframe_hat)(array_of_dfs[0, j], model_formula)
    #         for j in tqdm(range(min(test_iterations, array_of_dfs.shape[1])), desc="Processing DataFrames")
    #     )
    # except KeyboardInterrupt:
    #     print("Process interrupted by user.")
    #     return None

    # results = Parallel(n_jobs=10, temp_folder=r'D:\LYW\buff4parallel')(
    #     delayed(process_single_dataframe_hat)(array_of_dfs[0, j], model_formula)
    #     for j in tqdm(range(min(test_iterations, array_of_dfs.shape[1])), desc="Processing DataFrames")
    # )

    results = []
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=7) as executor:
        futures = [
            executor.submit(process_single_dataframe_hat, array_of_dfs[0, j], model_formula)
            for j in tqdm(range(min(test_iterations, array_of_dfs.shape[1])), desc="Processing DataFrames")
        ]

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"An error occurred: {e}")
                results.append((np.nan, np.nan))

    # Initialize arrays to store the results
    coef_estimates = np.zeros(array_of_dfs.shape[1], dtype=float)
    std_errors = np.zeros(array_of_dfs.shape[1], dtype=float)

    # Unpack results into separate arrays
    for idx, (coef_estimate, std_error) in enumerate(results):
        if np.isnan(coef_estimate) or np.isnan(std_error):
            print(f"NaN value found at index {idx}")
            print(f"DataFrame at index {idx}:\n{array_of_dfs[0, idx]}")
            # Export the DataFrame to an Excel file for inspection
            df_with_nan = array_of_dfs[0, idx]
            df_with_nan.to_excel(f"DataFrame_with_NaN_at_index_{idx}.xlsx", index=False)
            print(f"DataFrame with NaN exported to DataFrame_with_NaN_at_index_{idx}.xlsx")
            coef_estimates[idx] = np.nan
            std_errors[idx] = np.nan
        else:
            coef_estimates[idx] = coef_estimate
            std_errors[idx] = std_error

    # Compute max_std_error while ignoring NaNs
    max_std_error = np.nanmax(std_errors)
    print(f"Max std_error (ignoring NaNs): {max_std_error}")

    # Compute t-values, adding smoothing to avoid division by zero
    denominator = std_errors + max_std_error * sigma
    results_t_original = coef_estimates / std_errors
    results_t_smooth = coef_estimates / denominator

    # Save the results
    np.save('results_t_original.npy', results_t_original)
    np.save('results_t_smooth.npy', results_t_smooth)
    print('Results saved to results_t_original.npy and results_t_smooth.npy')

    return results_t_smooth

# Example usage with a limited number of iterations for testing
# array_of_dfs = ...  # Your ndarray of DataFrames
# results = apply_lmer_models_to_array_parallel_hat(array_of_dfs, test_iterations=10)

'''
one channel all time points find and test cluster
'''
def clusterbased_permutation_1d_1samp_2sided_mixmodel_2007(
                                                            p_threshold=0.05, clusterp_threshold=0.05,
                                                           n_threshold=2,
                                                           iter=1000,
                                                           updated_dataframes_list=None,
                                                           model_formula=None,
                                                           x=450, allow_permutation = True
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

    here, ps_stats is dict:n_channel

    """
    ps = np.zeros([x])
    ps2 = np.zeros([x])
    ts = np.zeros([x])
    cluster_p_values_pos = [] # 用来存储每个cluster的p值 = index/iter
    cluster_p_values_neg = []
    print(len(updated_dataframes_list))

    for t in range(x):
        print(t)
        ts[t], p = fit_lmer_model_R(updated_dataframes_list[t], model_formula)
        # print(p)
        ps2[t] = p # ps2 record original p-values

        # 应该算单边的
        if p / 2 < p_threshold and ts[t] > 0:
            ps[t] = 1
        if p / 2 < p_threshold and ts[t] < 0:
            ps[t] = -1

    # cluster_n1 就是正的情况.
    cluster_index1, cluster_n1, cluster_index2, cluster_n2 = get_cluster_index_1d_2sided(ps)

    # Perform permutation test if allowed
    if allow_permutation:

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
                    i_ts[t], p = fit_lmer_model_R(shuffled_data, model_formula)
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
                    i_ts[t], p = fit_lmer_model_R(shuffled_data, model_formula)
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
'''


'''
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

for hotmap ploting

hotmap with static process (refine from the neurora function plot_hetmap_withstata)

I deleted the plot part and left only statistical part.

in fact, I think this function and plot_tbyt_diff_decoding_acc  

are same..

hotmap function include a loop over all fields and plot_tbyt_diff_decoding_acc
only process one field.

the function seems only prepare data for permutation test.

'''
def mixmodelMultiChannel(
                            chllabels=None, time_unit=[0, 0.1], lim=[-7, 7], p=0.05, cbpt=False,
                            clusterp=0.05, stats_time=[0, 1], smooth=False, xlabel='Time (s)', ylabel='Channel',
                            clabel='t', ticksize=18, figsize=None, cmap=None, title=None, title_fontsize=16,
                            updated_dfs_dict=None, nts=450, nsubs=28, model_formula = None, num_iter = 10):
    """
    plot the hotmap of statistical results for channels/regions by time sequence

    results : array
        The results.
        The shape of results must be [n_subs, n_chls, ts, 2] or [n_subs, n_chls, ts]. n_subs represents the number of
        subjects. n_chls represents the number of channels or regions. ts represents the number of time-points. If shape
        of corrs is [n_chls, ts 2], each time-point of each channel/region contains a r-value and a p-value. If shape is
        [n_chls, ts], only r-values.
    chllabels : string-array or string-list or None. Default is None.
        The label for channels/regions.
        If label=None, the labels will be '1st', '2nd', '3th', '4th', ... automatically.
    time_unit : array or list [start_t, t_step]. Default is [0, 0.1]
        The time information of corrs for plotting
        start_t represents the start time and t_step represents the time between two adjacent time-points. Default
        time_unit=[0, 0.1], which means the start time of corrs is 0 sec and the time step is 0.1 sec.
    lim : array or list [min, max]. Default is [0, 1].
        The corrs view lims.
    p: float. Default is 0.05.
        The p threshold for outline.
    cbpt : bool True or False. Default is True.
        Conduct cluster-based permutation test or not.
    clusterp : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    stats_time : array or list [stats_time1, stats_time2]. Default os [0, 1].
        The time period for statistical analysis.
    smooth : bool True or False. Default is False.
        Smooth the results or not.
    xlabel : string. Default is 'Time (s)'.
        The label of x-axis.
    ylabel : string. Default is 'Channel'.
        The label of y-axis.
    clabel : string. Default is 'Similarity'.
        The label of color-bar.
    ticksize : int or float. Default is 18.
        The size of the ticks.
    figsize : array or list, [size_X, size_Y]
        The size of the figure.
        If figsize=None, the size of the figure will be ajusted automatically.
    cmap : matplotlib colormap or None. Default is None.
        The colormap for the figure.
        If cmap=None, the ccolormap will be 'bwr'.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    """


    # if len(np.shape(results)) < 3 or len(np.shape(results)) > 4:
    #     return "Invalid input!"

    # get the number of channels
    # nchls = results.shape[1]
    nchls = len(chllabels)

    # get the number of time-points
    # nts = results.shape[2]
    nts = nts

    # get the start time and the time step
    start_time = time_unit[0]
    tstep = time_unit[1]

    # calculate the end time
    end_time = start_time + nts * tstep

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

    # # set labels of the channels
    # if chllabels == None:
    #
    #     chllabels = []
    #     for i in range(nchls):
    #
    #         if i % 10 == 0 and i != 10:
    #             newlabel = str(i + 1) + "st"
    #         elif i % 10 == 1 and i != 11:
    #             newlabel = str(i + 1) + "nd"
    #         elif i % 10 == 2 and i != 12:
    #             newlabel = str(i + 1) + "rd"
    #         else:
    #             newlabel = str(i + 1) + "th"
    #
    #         chllabels.append(newlabel)
    #
    # '''
    # 后面不用result 那就全都不要了
    # '''
    # # if len(results.shape) == 4:
    # #     rlts = results[:, :, :, 0]
    # # elif len(results.shape) == 3:
    # #     rlts = results
    # #
    #
    # '''
    # 比较有问题的地方是这里,
    #
    # 我的数据如果要进行 smooth,还是非常复杂的,需要重构了..
    # '''
    # # # smooth the results
    # # if smooth == True:
    # #
    # #     for chl in range(nchls):
    # #         rlts[:, chl] = smooth_1d(rlts[:, chl])
    #
    # fig = plt.gcf()
    # size = fig.get_size_inches()
    #
    # if figsize == None:
    #     size_x = nts * tstep * (size[0] - 2) + 2
    #     size_y = nchls * 0.2 * (size[1] - 1.5) + 1.5
    # else:
    #     size_x = figsize[0]
    #     size_y = figsize[1]
    #
    # fig.set_size_inches(size_x, size_y)
    #
    # delta = (size_y * 3) / (size_x * 4)
    #
    # '''
    # !!!! 统计不需要ts了,但是作图需要啊...所以还是得想办法把result还原出来,相当于数据的另一种形态.
    # 而且你的result不应该是这样子?
    # 应该是 2sample之间进行比较, 我应该把 array_massed_s, array_spaced_s 给传进来使用.
    #
    #
    # ts只在非permutation的那段代码中用到,所以可以直接不要了.
    #
    # ts将具有与rlts中的n_chls和ts相对应的结构，但不再有n_subs维度，因为沿该维度进行了统计测试。
    #
    # '''
    # # # ts = ttest_1samp(rlts, 0, axis=0)[:, :, 0]
    # # ts = np.zeros((nchls, nts))
    # #
    # # # 对每个通道和每个时间点进行t检验
    # # for chl in range(nchls):
    # #     for time_point in range(nts):
    # #         # 提取两组数据中对应的数据
    # #         massed_data = rlts_array_massed_s[:, chl, time_point]
    # #         spaced_data = rlts_array_spaced_s[:, chl, time_point]
    # #
    # #         # 执行t检验
    # #         # 这里可以考虑ttest_rel.
    # #         t_stat, p_val = ttest_ind(spaced_data, massed_data)
    # #
    # #         # 存储t统计量
    # #         ts[chl, time_point] = t_stat
    # #
    # #     ps = np.zeros([nchls, nts])

    if cbpt == True:

        # Initialize dictionaries to store results
        stats_results = {
            'ps_stats': {},
            'ts_stats': {},
            'pss_stats': {},
            'cluster_p_values_pos': {},
            'cluster_p_values_neg': {}
        }

        for chl in range(nchls):
            # 是和0进行比较 level = 0, 是否有t大于0显著的cluster.
            # updated_dataframes_list =updated_dataframes_list[stats_time1-1:stats_time2] 不能这么写了.

            channel = chl
            time_range = range(stats_time1, stats_time2)  # 时间范围

            # 初始化一个列表来存储这个时间范围内的所有DataFrame
            list_time_range_dfs = []

            # 遍历时间范围，获取每个时间点对应的DataFrame，并添加到列表中
            for time_point in time_range:
                if time_point in updated_dfs_dict[channel]:  # 确保这个时间点的数据是存在的
                    print('append success')
                    list_time_range_dfs.append(updated_dfs_dict[channel][time_point])

            # 此时，time_range_dfs包含了第1个channel在时间点0到199的所有DataFrame副本
            # rlts可以直接删除了.
            # 真正费时间的是 iter, 测试的时候感觉应该把这个参数调低一点.
            print(chl)

            allow_permutation = True
            if num_iter == 0:
                allow_permutation = False
            else:
                allow_permutation = True

            ps_stats, ts_stats, pss_stats, cluster_p_values_pos, cluster_p_values_neg = clusterbased_permutation_1d_1samp_2sided_mixmodel_2007(
                p_threshold=p,
                clusterp_threshold=clusterp, iter=num_iter,
                updated_dataframes_list=list_time_range_dfs[
                                        stats_time1:stats_time2],
                x=stats_time2 - stats_time1,
                model_formula= model_formula
                # 因为你就给了updated_dataframes_list的是统计区间数据 而不是所有数据.
                ,allow_permutation=allow_permutation
            )
            print(pss_stats)
            ps = np.zeros([nts])
            ps[stats_time1:stats_time2] = ps_stats

            # Store the results for this channel
            stats_results['ps_stats'][chl] = ps_stats
            stats_results['ts_stats'][chl] = ts_stats
            stats_results['pss_stats'][chl] = pss_stats

            stats_results['cluster_p_values_pos'][chl] = cluster_p_values_pos
            stats_results['cluster_p_values_neg'][chl] = cluster_p_values_neg

    else:
        print('请设置 cbpt=Ture,进行permutation')

    '''
    我一定会进行clusterbased permutation.
    
    but for test... I don't need.

    '''

    # else:
    #     for chl in range(nchls):
    #         for t in range(nts):
    #             if t >= stats_time1 and t < stats_time2:
    #                 ps[chl, t] = ttest_1samp(rlts[:, chl, t], 0)[1]
    #                 if ps[chl, t] < p and ts[chl, t] > 0:
    #                     ps[chl, t] = 1
    #                 elif ps[chl, t] < p and ts[chl, t] < 0:
    #                     ps[chl, t] = -1
    #                 else:
    #                     ps[chl, t] = 0



    return stats_results

'''
keep only plot part except statistical part.
'''
def plot_heatmap(ts_stats, chllabels, time_unit=[0, 0.1], lim=[-7, 7], xlabel='Time (s)', ylabel='Channel',
                 clabel='t', ticksize=18, figsize=(10, 5), cmap='bwr', title=None, title_fontsize=16):
    """
    Plots a heatmap of statistical results for channels/regions by time sequence.

    Parameters:
        ts_stats : array
            The array containing statistical values to be plotted.
            Expected shape is [n_chls, n_ts] where n_chls is the number of channels and n_ts is the number of time points.
        chllabels : list
            Labels for the channels.
        time_unit : list
            Time information for plotting: [start_time, time_step].
        lim : list
            Color scale limits: [min_value, max_value].
        xlabel, ylabel, clabel : str
            Labels for the x-axis, y-axis, and colorbar, respectively.
        ticksize : int
            Font size for tick labels.
        figsize : tuple
            Figure size.
        cmap : str or Colormap
            Matplotlib colormap.
        title : str
            Title of the plot.
        title_fontsize : int
            Font size for the plot title.
    """

    # Get the number of time points and start/end times
    nts = ts_stats.shape[1]
    start_time = time_unit[0]
    tstep = time_unit[1]
    end_time = start_time + nts * tstep

    # Setting the figure size
    plt.figure(figsize=figsize)

    # Create a meshgrid for plotting
    x = np.linspace(start_time, end_time, nts)
    y = np.arange(len(chllabels))
    X, Y = np.meshgrid(x, y)

    # Plotting the heatmap
    heatmap = plt.pcolormesh(X, Y, ts_stats, cmap=cmap, vmin=lim[0], vmax=lim[1])
    plt.colorbar(heatmap, label=clabel)

    # Setting labels and ticks
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.yticks(np.arange(len(chllabels)) + 0.5, chllabels, fontsize=ticksize)
    plt.xticks(fontsize=ticksize)

    # Setting the title
    if title:
        plt.title(title, fontsize=title_fontsize)

    # Display the plot
    plt.tight_layout()
    plt.show()

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
    for sigle subject? 应该是了, 否则会有 mix model

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
                    # keep original r, becasue r have a range of [-1, 1],
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
'''
update 2024年10月6日:
To handle unequal numbers of trials between conditions.

Explanation of Changes
Separate Data for Each Condition:
Extracted data_cond1 and data_cond2 from data.
Now we can handle different numbers of trials for each condition.
Adjusted Data Preprocessing:
Applied preprocessing separately for each condition:
Modified Nested Loops:
Looped over trials in both conditions:
Distance Computation:
Computed distances between all possible pairs of trials from condition 1 and condition 2.
Distance Metric Adjustment:
For the correlation method, adjusted the calculation of distance based on use_abs:
Return Value:
The function now returns distances correctly computed over all trial pairs between conditions.
Returned as a NumPy array for consistency.

'''
def condition_difference_score_permutations(data, time_opt='features',
                                            time_win=50, time_step=5,
                                            method='correlation', use_abs=False,
                                            return_type='all'):
    """
    Calculate distances representing the differences between two conditions in EEG data across time windows,
    considering all possible pairings of trials between the two conditions using specified distance metric.

    Parameters:
    - data (list or numpy.ndarray): The input data containing data for two conditions.
        Should be a list or array with elements:
        data[0]: Data for condition 1, shape (n_trials_cond1, n_channels, n_timepoints)
        data[1]: Data for condition 2, shape (n_trials_cond2, n_channels, n_timepoints)
    - time_opt (str): Option to preprocess the data; 'average' to average all time points,
        'features' to use the reshaped feature vectors. Default is 'features'.
    - time_win (int): Length of the time window over which to calculate the distance.
    - time_step (int): Step size to move the window across the time samples.
    - method (str): Method to calculate distance between trials ('euclidean' or 'correlation').
    - use_abs (bool): If using 'correlation', whether to use the absolute value of the correlation coefficient.
    - return_type (str): Determines the return value; 'mean' returns the average distance per window,
        'all' returns an array of all computed distances per window.

    Returns:
    - numpy.ndarray: Depending on return_type, either an array of mean distances per window or a list
      of arrays of distances for each window.
    """
    data_cond1 = data[0]  # Shape: (n_trials_cond1, n_channels, n_timepoints)
    data_cond2 = data[1]  # Shape: (n_trials_cond2, n_channels, n_timepoints)

    n_trials_cond1 = data_cond1.shape[0]
    n_trials_cond2 = data_cond2.shape[0]
    n_chls = data_cond1.shape[1]
    n_ts = data_cond1.shape[2]
    n_windows = (n_ts - time_win) // time_step + 1
    results = []

    for window_idx in range(n_windows):
        start_idx = window_idx * time_step
        end_idx = start_idx + time_win
        data_window_cond1 = data_cond1[:, :, start_idx:end_idx]
        data_window_cond2 = data_cond2[:, :, start_idx:end_idx]

        if time_opt == 'average':
            data_preprocessed_cond1 = data_window_cond1.mean(axis=2)  # Shape: (n_trials_cond1, n_chls)
            data_preprocessed_cond2 = data_window_cond2.mean(axis=2)  # Shape: (n_trials_cond2, n_chls)
        elif time_opt == 'features':
            data_preprocessed_cond1 = data_window_cond1.reshape(n_trials_cond1, n_chls * time_win)
            data_preprocessed_cond2 = data_window_cond2.reshape(n_trials_cond2, n_chls * time_win)

        distances = []
        for i in range(n_trials_cond1):
            for j in range(n_trials_cond2):
                if method == 'correlation':
                    r, _ = pearsonr(data_preprocessed_cond1[i], data_preprocessed_cond2[j])
                    if use_abs:
                        distance = 1 - abs(r)
                    else:
                        # distance = 1 - r  # Inverse correlation to get distance
                        distance = r  # the sign of r have meaning, 1-r will lose this information.
                elif method == 'euclidean':
                    distance = np.linalg.norm(data_preprocessed_cond1[i] - data_preprocessed_cond2[j])
                distances.append(distance)

        if return_type == 'mean':
            mean_distance = np.mean(distances)
            results.append(mean_distance)
        elif return_type == 'all':
            results.append(distances)

    return np.array(results)
'''
update 2024年10月6日:
Here's how we can proceed:
Modify condition_difference_score_permutations to store trial pair indices along with the distances.
Adjust normalize_across_comparisons_trial to handle the new output format.
Ensure that the rest of your code can process the new data structure.
---
1. Modify condition_difference_score_permutations to Store Trial Pair Information
We'll update the function to return not just the distances but also the indices of the trial pairs that were compared.
We'll achieve this by storing tuples of (distance, trial_index_cond1, trial_index_cond2).

'''
def condition_difference_score_permutations(data, time_opt='features',
                                            time_win=50, time_step=5,
                                            method='correlation', use_abs=False,
                                            return_type='all'):
    """
    Calculate distances representing the differences between two conditions in EEG data across time windows,
    considering all possible pairings of trials between the two conditions using specified distance metric,
    and keeping track of trial pairs.

    Parameters:
    - data (list or numpy.ndarray): The input data containing data for two conditions.
        data[0]: Data for condition 1, shape (n_trials_cond1, n_channels, n_timepoints)
        data[1]: Data for condition 2, shape (n_trials_cond2, n_channels, n_timepoints)
    - time_opt (str): Option to preprocess the data; 'average' to average all time points,
        'features' to use the reshaped feature vectors. Default is 'features'.
    - time_win (int): Length of the time window over which to calculate the distance.
    - time_step (int): Step size to move the window across the time samples.
    - method (str): Method to calculate distance between trials ('euclidean' or 'correlation').
    - use_abs (bool): If using 'correlation', whether to use the absolute value of the correlation coefficient.
    - return_type (str): Determines the return value; 'mean' returns the average distance per window,
        'all' returns a list of lists, each containing tuples with distances and trial indices.

    Returns:
    - If return_type is 'mean':
        - numpy.ndarray: Array of mean distances per window.
    - If return_type is 'all':
        - List of lists for each window, where each inner list contains tuples:
            (distance, trial_index_cond1, trial_index_cond2)
    """
    data_cond1 = data[0]  # Shape: (n_trials_cond1, n_channels, n_timepoints)
    data_cond2 = data[1]  # Shape: (n_trials_cond2, n_channels, n_timepoints)

    n_trials_cond1 = data_cond1.shape[0]
    n_trials_cond2 = data_cond2.shape[0]
    n_chls = data_cond1.shape[1]
    n_ts = data_cond1.shape[2]
    n_windows = (n_ts - time_win) // time_step + 1
    results = []

    for window_idx in range(n_windows):
        start_idx = window_idx * time_step
        end_idx = start_idx + time_win
        data_window_cond1 = data_cond1[:, :, start_idx:end_idx]
        data_window_cond2 = data_cond2[:, :, start_idx:end_idx]

        if time_opt == 'average':
            data_preprocessed_cond1 = data_window_cond1.mean(axis=2)  # Shape: (n_trials_cond1, n_chls)
            data_preprocessed_cond2 = data_window_cond2.mean(axis=2)  # Shape: (n_trials_cond2, n_chls)
        elif time_opt == 'features':
            data_preprocessed_cond1 = data_window_cond1.reshape(n_trials_cond1, n_chls * time_win)
            data_preprocessed_cond2 = data_window_cond2.reshape(n_trials_cond2, n_chls * time_win)

        distances = []
        for i in range(n_trials_cond1):
            for j in range(n_trials_cond2):
                if method == 'correlation':
                    r, _ = pearsonr(data_preprocessed_cond1[i], data_preprocessed_cond2[j])
                    if use_abs:
                        distance = 1 - abs(r)
                    else:
                        distance = 1 - r
                elif method == 'euclidean':
                    distance = np.linalg.norm(data_preprocessed_cond1[i] - data_preprocessed_cond2[j])
                # Store the distance along with the trial indices
                distances.append((distance, i, j))

        if distances:
            if return_type == 'mean':
                mean_distance = np.mean([d[0] for d in distances])  # Extract distances for averaging
                results.append(mean_distance)
            elif return_type == 'all':
                results.append(distances)
        else:
            # Handle cases where no distances were calculated
            results.append(np.nan)

    return results

'''
return including both distances and corresponding metadata for each trial pair.
'''

def condition_difference_score_permutations(data, metadata=None, time_opt='features',
                                            time_win=50, time_step=5,
                                            method='correlation', use_abs=False,
                                            return_type='all'):
    """
    Calculate distances representing the differences between two conditions in EEG data across time windows,
    considering all possible pairings of trials between the two conditions using specified distance metric,
    and keeping track of trial pairs and their metadata.

    Parameters:
    - data (list or numpy.ndarray): The input data containing data for two conditions.
        data[0]: Data for condition 1, shape (n_trials_cond1, n_channels, n_timepoints)
        data[1]: Data for condition 2, shape (n_trials_cond2, n_channels, n_timepoints)
    - metadata (list, optional): Metadata for the two conditions, each a DataFrame.
        metadata[0]: Metadata for condition 1, shape (n_trials_cond1, n_features)
        metadata[1]: Metadata for condition 2, shape (n_trials_cond2, n_features)
    - time_opt (str): Option to preprocess the data; 'average' to average all time points,
        'features' to use the reshaped feature vectors. Default is 'features'.
    - time_win (int): Length of the time window over which to calculate the distance.
    - time_step (int): Step size to move the window across the time samples.
    - method (str): Method to calculate distance between trials ('euclidean' or 'correlation').
    - use_abs (bool): If using 'correlation', whether to use the absolute value of the correlation coefficient.
    - return_type (str): Determines the return value; 'mean' returns the average distance per window,
        'all' returns a list of lists, each containing tuples with distances and trial indices.

    Returns:
    - If return_type is 'mean':
        - numpy.ndarray: Array of mean distances per window.
    - If return_type is 'all':
        - List of lists for each window, where each inner list contains tuples:
            (distance, trial_index_cond1, trial_index_cond2, metadata_cond1, metadata_cond2)
    """
    data_cond1 = data[0]  # Shape: (n_trials_cond1, n_channels, n_timepoints)
    data_cond2 = data[1]  # Shape: (n_trials_cond2, n_channels, n_timepoints)

    n_trials_cond1 = data_cond1.shape[0]
    n_trials_cond2 = data_cond2.shape[0]
    n_chls = data_cond1.shape[1]
    n_ts = data_cond1.shape[2]
    n_windows = (n_ts - time_win) // time_step + 1
    results = []

    for window_idx in range(n_windows):
        start_idx = window_idx * time_step
        end_idx = start_idx + time_win
        data_window_cond1 = data_cond1[:, :, start_idx:end_idx]
        data_window_cond2 = data_cond2[:, :, start_idx:end_idx]

        if time_opt == 'average':
            data_preprocessed_cond1 = data_window_cond1.mean(axis=2)
            data_preprocessed_cond2 = data_window_cond2.mean(axis=2)
        elif time_opt == 'features':
            data_preprocessed_cond1 = data_window_cond1.reshape(n_trials_cond1, n_chls * time_win)
            data_preprocessed_cond2 = data_window_cond2.reshape(n_trials_cond2, n_chls * time_win)

        distances = []
        for i in range(n_trials_cond1):
            for j in range(n_trials_cond2):
                if method == 'correlation':
                    r, _ = pearsonr(data_preprocessed_cond1[i], data_preprocessed_cond2[j])
                    distance = 1 - abs(r) if use_abs else 1 - r
                elif method == 'euclidean':
                    distance = np.linalg.norm(data_preprocessed_cond1[i] - data_preprocessed_cond2[j])

                # Retrieve metadata for the trial pairs
                metadata_cond1 = metadata[0].iloc[i] if metadata else None
                metadata_cond2 = metadata[1].iloc[j] if metadata else None

                # Store the distance along with the trial indices and metadata
                distances.append((distance, i, j, metadata_cond1, metadata_cond2))

        if distances:
            if return_type == 'mean':
                mean_distance = np.mean([d[0] for d in distances])
                results.append(mean_distance)
            elif return_type == 'all':
                results.append(distances)
        else:
            results.append(np.nan)

    return results


'''
the core function to calculate the distance between two conditions 
for 1 subject.
loop over time windows - trial pair 
'''


def condition_difference_score_permutations(data, metadata=None, time_opt='features',
                                            time_win=50, time_step=5,
                                            method='correlation', use_abs=False,
                                            return_type='all'):
    """
    Calculate distances representing the differences between two conditions in EEG data across time windows,
    considering all possible pairings of trials between the two conditions using specified distance metric,
    and keeping track of trial pairs and their metadata.

    Parameters:
    - data (list or numpy.ndarray): The input data containing data for two conditions.
        data[0]: Data for condition 1, shape (n_trials_cond1, n_channels, n_timepoints)
        data[1]: Data for condition 2, shape (n_trials_cond2, n_channels, n_timepoints)
    - metadata (list, optional): Metadata for the two conditions, each a DataFrame.
        metadata[0]: Metadata for condition 1, shape (n_trials_cond1, n_features)
        metadata[1]: Metadata for condition 2, shape (n_trials_cond2, n_features)
    - time_opt (str): Option to preprocess the data; 'average' to average all time points,
        'features' to use the reshaped feature vectors. Default is 'features'.
    - time_win (int): Length of the time window over which to calculate the distance.
    - time_step (int): Step size to move the window across the time samples.
    - method (str): Method to calculate distance between trials ('euclidean' or 'correlation').
    - use_abs (bool): If using 'correlation', whether to use the absolute value of the correlation coefficient.
    - return_type (str): Determines the return value; 'mean' returns the average distance per window,
        'all' returns a list of DataFrames for each window.

    Returns:
    - If return_type is 'mean':
        - numpy.ndarray: Array of mean distances per window.
    - If return_type is 'all':
        - List of DataFrames for each window, where each DataFrame contains:
            - 'distance': Distance value
            - Metadata columns from both conditions
    """
    data_cond1 = data[0]
    data_cond2 = data[1]

    n_trials_cond1 = data_cond1.shape[0]
    n_trials_cond2 = data_cond2.shape[0]
    n_chls = data_cond1.shape[1]
    n_ts = data_cond1.shape[2]
    n_windows = (n_ts - time_win) // time_step + 1
    results = []

    start_time = time.time()
    for window_idx in range(n_windows):
        start_idx = window_idx * time_step
        end_idx = start_idx + time_win
        data_window_cond1 = data_cond1[:, :, start_idx:end_idx]
        data_window_cond2 = data_cond2[:, :, start_idx:end_idx]

        if time_opt == 'average':
            data_preprocessed_cond1 = data_window_cond1.mean(axis=2)
            data_preprocessed_cond2 = data_window_cond2.mean(axis=2)
        elif time_opt == 'features':
            data_preprocessed_cond1 = data_window_cond1.reshape(n_trials_cond1, n_chls * time_win)
            data_preprocessed_cond2 = data_window_cond2.reshape(n_trials_cond2, n_chls * time_win)

        rows = []

        for i in range(n_trials_cond1):
            for j in range(n_trials_cond2):
                if method == 'correlation':
                    r, _ = pearsonr(data_preprocessed_cond1[i], data_preprocessed_cond2[j])
                    # distance = 1 - abs(r) if use_abs else 1 - r
                    distance = r # the original r have meaning in this case. the sign or r.
                elif method == 'euclidean':
                    distance = np.linalg.norm(data_preprocessed_cond1[i] - data_preprocessed_cond2[j])

                # Retrieve metadata for the trial pairs
                metadata_cond1 = metadata[0].iloc[i] if metadata else pd.Series()
                metadata_cond2 = metadata[1].iloc[j] if metadata else pd.Series()

                # Combine distance and metadata into a single row
                row = pd.concat([pd.Series({'distance': distance}), metadata_cond1.add_suffix('_cond1'), metadata_cond2.add_suffix('_cond2')])
                rows.append(row)
        end_time = time.time()

        if rows:
            df = pd.DataFrame(rows)
            results.append(df)
        else:
            results.append(pd.DataFrame())

    print(f"trial pairs number: {n_trials_cond1*n_trials_cond2}")
    print(f"loop over time windows : {end_time - start_time:.2f} seconds")

    return results


'''
Key Changes and Benefits
Vectorized Computations: Replaces the nested loops with vectorized operations, significantly reducing computation time.
Efficient Use of cdist: Utilizes scipy.spatial.distance.cdist to compute the pairwise distances between all trials in one function call.
Avoids Unnecessary Loops and Function Calls: Eliminates the per-trial overhead of calling pearsonr and using Pandas operations inside loops.
2. Optimize Data Structures and Memory Usage
Avoid Growing Lists in Loops: Preallocate lists or arrays when possible. This prevents the overhead of dynamically resizing data structures.
Efficient Metadata Handling: Instead of fetching and concatenating metadata inside loops, use indexing and vectorized operations to prepare the metadata for all combinations.
'''
from scipy.spatial.distance import cdist

def condition_difference_score_permutations(data, metadata=None, time_opt='features',
                                            time_win=50, time_step=5,
                                            method='correlation', use_abs=False,
                                            return_type='all'):
    """
    Optimized version using vectorized computations.
    """
    data_cond1 = data[0]  # Shape: (n_trials_cond1, n_channels, n_timepoints)
    data_cond2 = data[1]  # Shape: (n_trials_cond2, n_channels, n_timepoints)

    n_trials_cond1, n_chls, n_ts = data_cond1.shape
    n_trials_cond2 = data_cond2.shape[0]
    n_windows = (n_ts - time_win) // time_step + 1
    results = []

    for window_idx in range(n_windows):
        start_idx = window_idx * time_step
        end_idx = start_idx + time_win
        data_window_cond1 = data_cond1[:, :, start_idx:end_idx]
        data_window_cond2 = data_cond2[:, :, start_idx:end_idx]

        if time_opt == 'average':
            data_preprocessed_cond1 = data_window_cond1.mean(axis=2)
            data_preprocessed_cond2 = data_window_cond2.mean(axis=2)
        elif time_opt == 'features':
            data_preprocessed_cond1 = data_window_cond1.reshape(n_trials_cond1, -1)
            data_preprocessed_cond2 = data_window_cond2.reshape(n_trials_cond2, -1)

        if method == 'correlation':
            # Standardize data
            # data_preprocessed_cond1_z = (
            #     data_preprocessed_cond1 - data_preprocessed_cond1.mean(axis=1, keepdims=True)
            # ) / data_preprocessed_cond1.std(axis=1, ddof=1, keepdims=True)
            # data_preprocessed_cond2_z = (
            #     data_preprocessed_cond2 - data_preprocessed_cond2.mean(axis=1, keepdims=True)
            # ) / data_preprocessed_cond2.std(axis=1, ddof=1, keepdims=True)
            # Compute correlation distances
            # distance_matrix = cdist(data_preprocessed_cond1_z, data_preprocessed_cond2_z, metric='correlation')
            distance_matrix = cdist(data_preprocessed_cond1, data_preprocessed_cond2, metric='correlation')

            if not use_abs:
                # Convert correlation distance back to correlation coefficient
                distance_matrix = 1 - distance_matrix
            else:
                # Use absolute correlation coefficient
                # distance_matrix = 1 - np.abs(1 - distance_matrix)
                distance_matrix = distance_matrix

        elif method == 'euclidean':
            distance_matrix = cdist(data_preprocessed_cond1, data_preprocessed_cond2, metric='euclidean')

        # Flatten the distance matrix for DataFrame construction
        distances_flat = distance_matrix.flatten()

        # Create indices for trials
        idx_cond1 = np.repeat(np.arange(n_trials_cond1), n_trials_cond2)
        idx_cond2 = np.tile(np.arange(n_trials_cond2), n_trials_cond1)

        # Prepare metadata for all combinations
        if metadata:
            metadata_cond1 = metadata[0].iloc[idx_cond1].reset_index(drop=True)
            metadata_cond2 = metadata[1].iloc[idx_cond2].reset_index(drop=True)
        else:
            metadata_cond1 = pd.DataFrame()
            metadata_cond2 = pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame({'distance': distances_flat})
        if not metadata_cond1.empty:
            df = pd.concat([df, metadata_cond1.add_suffix('_cond1').reset_index(drop=True)], axis=1)
        if not metadata_cond2.empty:
            df = pd.concat([df, metadata_cond2.add_suffix('_cond2').reset_index(drop=True)], axis=1)

        results.append(df)

    return results

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

def normalize_across_comparisons_trial(subdata, normalize=False):
    """
    Normalize and compute distances between conditions, returning DataFrames with distance and metadata.

    Parameters:
    - subdata (dict): Dictionary containing:
        - 'data': List of data arrays for each condition.
        - 'metadata': List of metadata DataFrames for each condition.
    - normalize (bool): Whether to normalize distances.

    Returns:
    - df1, df2: DataFrames containing distances and metadata for the two comparisons.
    """
    data = subdata['data']
    metadata = subdata['metadata']

    # Extract data subsets for the two comparisons
    cond1_vs_cond2 = [data[0], data[1]]
    cond1_vs_cond3 = [data[0], data[2]]

    # Calculate distances for both comparisons
    df1 = condition_difference_score_permutations(cond1_vs_cond2, metadata=[metadata[0], metadata[1]], return_type='all')
    df2 = condition_difference_score_permutations(cond1_vs_cond3, metadata=[metadata[0], metadata[2]], return_type='all')

    if normalize:
        # Normalize distances in each DataFrame
        df1['distance'] = df1['distance'] / df1['distance'].max()
        df2['distance'] = df2['distance'] / df2['distance'].max()

    return df1, df2


# Function to process data for a given set of channels
'''

process_channels
then 
process_field



loop subjects
'''
import gc
def process_channels(
        list_epochs_One,list_epochs_M2,list_epochs_S2,
        field_channels,T1_sub_ids):

    '''
    # ## 数据整理 for distance
    # 每个被试一个df,hope to 包括了 行为数据列 和 各个通道 某个时间点的信号数据列
    # 
    # if I want to calculate based on different fields, I should only change the parameters of channels_all.
    '''
    # Start timing for the entire block
    overall_start_time = time.time()

    # data for specific field(4-5 channels)
    list_subdata = organize_eeg_data_by_subject_and_condition(T1_sub_ids, list_epochs_One, list_epochs_M2,
                                                              list_epochs_S2, field_channels)

    n_subs = len(list_subdata)
    # example_data = normalize_across_comparisons_trial(list_subdata[0], normalize=False)
    # n_windows = len(example_data[0])  # Assuming each entry in the example_data is a list for each window

    # Initialize lists to store lists of distances for all subjects
    normalized_distances_all_1 = [[] for _ in range(n_subs)]
    normalized_distances_all_2 = [[] for _ in range(n_subs)]

    # Loop over each subject's data to calculate distances
    for idx, data in enumerate(list_subdata):
        normalized_distances1, normalized_distances2 = normalize_across_comparisons_trial(data, normalize=False)
        normalized_distances_all_1[idx] = normalized_distances1
        normalized_distances_all_2[idx] = normalized_distances2

    # Example of accessing the first window's data for the first subject
    # print("Distances for the first window of the first subject (Cond1 vs Cond2):", normalized_distances_all_1[0][0])
    # print("Distances for the first window of the first subject (Cond1 vs Cond3):", normalized_distances_all_2[0][0])

    # Assuming 'normalized_distances_all_1' is loaded as described, with shape [n_subs, n_wins][n_distances]

    # sub_ids = T1_sub_ids

    # Initialize lists to hold the dataframes for each window for both comparison sets
    windows_dataframes_1 = []
    windows_dataframes_2 = []

    n_wins = 91  # Replace with dynamically determined value if necessary
    sub_ids = T1_sub_ids

    # to reform data structure, the distance calculation is finished before this step.
    # no meaning but to match the requirement of former version code.
    for window_index in range(n_wins):  # Replace 91 with n_wins if dynamically determined elsewhere
        # Initialize lists to collect all subjects' data for the current window for both sets
        all_subjects_data_1 = []
        all_subjects_data_2 = []

        # Gather data for each subject in this window for both sets
        for sub_index, sub_id in enumerate(sub_ids):
            # Get the distances for the current subject in the current window from both sets
            distances_1 = normalized_distances_all_1[sub_index][window_index]
            distances_2 = normalized_distances_all_2[sub_index][window_index]

            sub_df_1 = distances_1
            sub_df_2 = distances_2

            # Append the DataFrames to their respective lists
            all_subjects_data_1.append(sub_df_1)
            all_subjects_data_2.append(sub_df_2)

        # Concatenate all subjects' data into a single DataFrame for the current window for both sets
        window_df_1 = pd.concat(all_subjects_data_1, ignore_index=True)
        window_df_2 = pd.concat(all_subjects_data_2, ignore_index=True)

        # Append the concatenated DataFrames to the lists holding windows DataFrames
        windows_dataframes_1.append(window_df_1)
        windows_dataframes_2.append(window_df_2)
        '''
    ## integrate with additional data for mix model calculation.

    data strctrure: 

    a list of dataframes, each dataframe contains the data for one window, and each row represents one subject's data for that window.

    '''

    # additional_data = pd.read_csv('I:\pycharmProject\pre10\统计和建模\\filtered_pivoted_data_2024年03月14日.csv')
    #
    # Merge and combine windows_dataframes_1 and windows_dataframes_2 with additional_data
    # windows_dataframes_merged_1 = []
    # windows_dataframes_merged_2 = []
    #
    # for window_df_1, window_df_2 in zip(windows_dataframes_1, windows_dataframes_2):
    #     merged_df_1 = pd.merge(window_df_1, additional_data, on=['subId', 'category'], how='left')
    #     windows_dataframes_merged_1.append(merged_df_1)
    #
    #     merged_df_2 = pd.merge(window_df_2, additional_data, on=['subId', 'category'], how='left')
    #     windows_dataframes_merged_2.append(merged_df_2)
    #
    # Combine the corresponding DataFrames from merged lists vertically
    # final_merged_dataframes = []
    # for merged_df_1, merged_df_2 in zip(windows_dataframes_merged_1, windows_dataframes_merged_2):
    #     final_df = pd.concat([merged_df_1, merged_df_2], ignore_index=True)
    #     final_merged_dataframes.append(final_df)


    windows_dataframes_merged_1 = []
    windows_dataframes_merged_2 = []

    for window_df_1, window_df_2 in zip(windows_dataframes_1, windows_dataframes_2):
        # merged_df_1 = pd.merge(window_df_1, additional_data, on=['subId', 'category'], how='left')
        windows_dataframes_merged_1.append(window_df_1)

        # merged_df_2 = pd.merge(window_df_2, additional_data, on=['subId', 'category'], how='left')
        windows_dataframes_merged_2.append(window_df_2)

    # Combine the corresponding DataFrames from merged lists vertically
    final_merged_dataframes = []
    for merged_df_1, merged_df_2 in zip(windows_dataframes_merged_1, windows_dataframes_merged_2):
        final_df = pd.concat([merged_df_1, merged_df_2], ignore_index=True)
        final_merged_dataframes.append(final_df)

    '''
    ## calculate the mean just for plotting.

    plot_tbyt_diff_decoding_acc need these two arrays:
    '''

    all_distances_M = []
    all_distances_S = []
    # Initialize a new list to store the averaged DataFrames
    averaged_dataframes = []

    # for df in final_merged_dataframes:
    #     # Group by 'category' and 'subId' and calculate the mean
    #     averaged_df = df.groupby(['category_cond2', 'subId_cond2']).mean().reset_index()
    #     averaged_dataframes.append(averaged_df)
    for df in final_merged_dataframes:
        # Select only numeric columns for aggregation
        numeric_df = df.select_dtypes(include=[np.number])
        # Group by 'category_cond2' and 'subId_cond2' and calculate the mean
        averaged_df = numeric_df.groupby([df['category_cond2'], df['subId_cond2']]).mean().reset_index()
        # Add back non-numeric columns if needed
        non_numeric_df = df.select_dtypes(exclude=[np.number]).drop_duplicates(subset=['category_cond2', 'subId_cond2'])
        averaged_df = pd.merge(averaged_df, non_numeric_df, on=['category_cond2', 'subId_cond2'], how='left')
        averaged_dataframes.append(averaged_df)


    # Initialize empty lists to store distance values for both categories
    all_distances_M = []
    all_distances_S = []

    # Loop through each DataFrame in the list
    for df in averaged_dataframes:
        # Filter the DataFrame for category 'M' and extract distance values
        filtered_distances_M = df[df['category_cond2'] == 'M']['distance'].values
        # Filter the DataFrame for category 'S' and extract distance values
        filtered_distances_S = df[df['category_cond2'] == 'S']['distance'].values

        # Append the distances to the corresponding list
        all_distances_M.append(filtered_distances_M)
        all_distances_S.append(filtered_distances_S)

    # Horizontally stack all distance arrays from category 'M' to create a single ndarray
    # This requires each array to have the same length
    final_distances_array_M = np.column_stack(all_distances_M)

    # Horizontally stack all distance arrays from category 'S' to create a single ndarray
    # This requires each array to have the same length
    final_distances_array_S = np.column_stack(all_distances_S)

    # Print the resulting ndarrays
    # print("Distances for category M:", final_distances_array_M)
    # print("Distances for category S:", final_distances_array_S)

    # End timing for the entire block
    overall_end_time = time.time()
    print(f"****loop all subjects: {overall_end_time - overall_start_time:.2f} seconds")

    # save memory, delete the intermediate variables.
    del list_subdata, normalized_distances_all_1, normalized_distances_all_2
    gc.collect()

    return final_distances_array_M, final_distances_array_S, final_merged_dataframes


# Define the function to process each field
# for one field.. convenient to parallel..
'''
    list_epochs_One = list()
    list_epochs_M2 = list()
    list_epochs_S2 = list()
    
why we have to load data in process_field() ?
parallel problems
every subprcess have its own memory space, so we have to load data in each process.
'''
def process_field(
                  field_name,
                  field_channels,
                T1_sub_ids, T2M_sub_ids, T2S_sub_ids, num_epochs = 3):

    ''''''
    base_data_path = 'D:\\LYW\\pre10\\data\\6epoch_clean_allWords_fullMeta_sep\\'
    # base_data_path = 'D:\\LYW\\pre10\\data\\6epoch_clean_equal\\' # test equal version organize_eeg_data_by_subject_and_condition function.

    list_epochs_One = list()
    list_epochs_M2 = list()
    list_epochs_S2 = list()


    # 调用函数读取T1_sub_ids对应的数据
    load_epochs(T1_sub_ids, list_epochs_One, base_data_path, field_channels,
                num_epochs=num_epochs)
    # 调用函数读取T2M_sub_ids对应的数据，注意添加'M'作为文件名后缀
    load_epochs(T2M_sub_ids, list_epochs_M2, base_data_path, field_channels,
                num_epochs=num_epochs)
    # 调用函数读取T2S_sub_ids对应的数据，注意添加'S'作为文件名后缀
    load_epochs(T2S_sub_ids, list_epochs_S2, base_data_path, field_channels,
                num_epochs=num_epochs)

    print(f"Processing {field_name} with channels {field_channels}")
    result_M, result_S, merged_dataframes = process_channels(
        list_epochs_One,list_epochs_M2,list_epochs_S2,
        field_channels, T1_sub_ids,
        # T2M_sub_ids, T2S_sub_ids
    )

    del list_epochs_One, list_epochs_M2, list_epochs_S2
    gc.collect()

    return field_name, result_M, result_S, merged_dataframes

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


