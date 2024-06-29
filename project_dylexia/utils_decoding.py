"""
this part is for data preprocessing
"""
'''
according to different standards, 
we can divide the subjects into different groups, 
such as young and old, right and left, etc.


dd vs td maybe not work here..
'''
def get_sub_ids(group: str) -> list:
    """
    Returns the list of subject IDs based on the specified group.

    Parameters:
    group (str): The group for which to retrieve subject IDs.
                 Valid options are 'dd_young', 'dd_old', 'td_young', 'td_old', 'dd', 'td'.

    Returns:
    list: A list of subject IDs corresponding to the specified group.

    Example:
    >>> get_sub_ids('dd_young')
    ['pre4010_y', 'pre4011_y', 'pre4012_y', ..., 'pre4093_y']
    """
    dd_young_ids = [
        'pre4010_y', 'pre4011_y', 'pre4012_y', 'pre4013_y', 'pre4015_y', 'pre4017_y', 'pre4018_y', 'pre4026_y', 'pre4027_y',
        'pre4031_y', 'pre4033_y', 'pre4034_y', 'pre4042_y', 'pre4043_y', 'pre4045_y', 'pre4048_y', 'pre4052_y', 'pre4060_y',
        'pre4061_y', 'pre4062_y', 'pre4066_y', 'pre4067_y', 'pre4073_y', 'pre4078_y', 'pre4079_y', 'pre4081_y', 'pre4093_y'
    ]
    dd_old_ids = [
        'pre4021_y', 'pre4022_y', 'pre4023_y', 'pre4028_y', 'pre4041_y', 'pre4056_y', 'pre4065_y', 'pre4069_y', 'pre4070_y',
        'pre4072_y', 'pre4076_y'
    ]
    td_young_ids = [
        'pre4008_y', 'pre4009_y', 'pre4029_y', 'pre4030_y', 'pre4039_y', 'pre4040_y', 'pre4044_y', 'pre4046_y', 'pre4051_y',
        'pre4063_y', 'pre4064_y', 'pre4068_y', 'pre4071_y', 'pre4074_y', 'pre4075_y', 'pre4083_y', 'pre4084_y', 'pre4090_y',
        'pre4091_y'
    ]
    td_old_ids = [
        'pre4014_y', 'pre4037_y', 'pre4038_y', 'pre4047_y', 'pre4049_y', 'pre4053_y', 'pre4054_y', 'pre4055_y', 'pre4057_y',
        'pre4058_y', 'pre4077_y', 'pre4080_y', 'pre4082_y', 'pre4085_y', 'pre4086_y', 'pre4088_y', 'pre4089_y', 'pre4092_y'
    ]
    dd_ids = dd_young_ids + dd_old_ids
    td_ids = td_young_ids + td_old_ids

    group_dict = {
        'dd_young': dd_young_ids,
        'dd_old': dd_old_ids,
        'td_young': td_young_ids,
        'td_old': td_old_ids,
        'dd': dd_ids,
        'td': td_ids
    }

    return group_dict.get(group, "Invalid group name. Valid options are: 'dd_young', 'dd_old', 'td_young', 'td_old', 'dd', 'td'")
#
# # Example usage:
# selected_ids = get_sub_ids('dd_old')
# print(selected_ids)

'''
according to subids and channels of fields
get the data and labels
'''

import numpy as np
import mne
'''
    Prepare EEG data for decoding analysis, structuring it by conditions and subjects.

'''
def prepare_eeg_data(sub_ids, file_path, channels, conditions, epoch_file_suffix='_RSA-epo.fif', timesteps=1001):
    """
    Prepare EEG data for decoding analysis, structuring it by conditions and subjects.

    Args:
    sub_ids (list of str): List of subject identifiers.
    file_path (str): Path to the directory containing the EEG data files.
    channels (list of str): List of EEG channels to include.
    conditions (list of str): Event codes to extract specific trials.
    epoch_file_suffix (str): Suffix of the EEG file names. Defaults to '_RSA-epo.fif'.
    timesteps (int): Number of time steps in each epoch. Defaults to 1001.

    Returns:
    list: A list of EEG data arrays per subject, structured for decoding.
    list: A list of labels arrays per subject.
    """
    list_subdata = []
    list_sublabel = []
    list_epochs_all = []

    # Load and prepare epochs data for all subjects
    for sub_id in sub_ids:
        data_path = file_path + sub_id + epoch_file_suffix
        epochs_all = mne.read_epochs(fname=data_path, preload=True)
        epochs_all = epochs_all.pick(picks=channels)
        epochs_all.equalize_event_counts(method='mintime')
        list_epochs_all.append(epochs_all)

    # Process each subject's data
    for idx, epochs_all in enumerate(list_epochs_all):
        num_trials = len(epochs_all[conditions[0]].events)
        subdata = np.zeros([len(conditions), num_trials, len(channels), timesteps], dtype=np.float32)
        sublabel = np.zeros(0, dtype=int)

        for i, cond in enumerate(conditions):
            epochs = epochs_all[cond]
            data = epochs.get_data()
            label_cond = epochs.events[:, 2]
            sublabel = np.append(sublabel, label_cond)
            subdata[i] = data

        sublabel = np.reshape(sublabel, [1, len(sublabel)])
        subdata = np.reshape(subdata, [len(conditions), 1, num_trials, len(channels), timesteps])
        data_decode = np.reshape(subdata, [1, len(conditions) * num_trials, len(channels), timesteps])

        list_subdata.append(data_decode)
        list_sublabel.append(sublabel)

    del list_epochs_all  # Free memory

    return list_subdata, list_sublabel

