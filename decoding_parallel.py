import numpy as np
from neurora.decoding import tbyt_decoding_kfold

'''
decoding for one subject data
which can be used for parallel processing
'''
def process_sub_data(sub_data, decoding_params):
    """
    处理单个被试的数据，执行解码并返回解码准确率。

    参数:
        sub_data (array): 包含特定被试的条件数据，结构为 [n_conditions, n_trials, n_channels, n_timepoints]
                          其中 n_conditions 为条件数，通常为 3 (e.g., ONE, M2, S2)
        decoding_params (dict): 解码函数的参数字典，包含所有解码配置

    返回:
        tuple: 包含两个元素，分别是 ONE vs M2 和 ONE vs S2 的解码准确率
    """
    print("Processing subject data...")
    one_data = sub_data[0, 0]  # ONE condition data
    m2_data = sub_data[1, 0]  # M2 condition data
    s2_data = sub_data[2, 0]  # S2 condition data

    def decode_conditions(condition1, condition2):
        """辅助函数用于解码两种条件的数据"""
        combined_data = np.expand_dims(np.concatenate((condition1, condition2), axis=0), axis=0)
        labels = np.expand_dims(np.concatenate((np.zeros(condition1.shape[0]), np.ones(condition2.shape[0])), axis=0), axis=0)
        return tbyt_decoding_kfold(data=combined_data, labels=labels, **decoding_params)

    acc_one_m2 = decode_conditions(one_data, m2_data)
    acc_one_s2 = decode_conditions(one_data, s2_data)

    return acc_one_m2, acc_one_s2
