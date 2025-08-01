'''
calculate max and min of tfce for each permutation.

假设你已经用R算出来所有的permutation的t结果，需要转换成tfce 分布。。
'''

import os
import glob
import mne
import mne.stats.cluster_level_backup as cluster_level_backup
from importlib import reload
import utils_EEG.stuff as stuff
from concurrent.futures import ProcessPoolExecutor, as_completed
from mne.channels import find_ch_adjacency
import pandas as pd
import numpy as np


# Reload custom module to ensure latest changes
reload(stuff)

# Define the set of channels to use (excluding FT9, FT10, TP9, TP10)
channels_all = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
    'Fz', 'Cz', 'Pz', 'FC1', 'FC2', 'CP1', 'CP2',
    'FC5', 'FC6', 'CP5', 'CP6', 'F1', 'F2', 'C1',
    'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4',
    'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5',
    'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8',
    'TP7', 'TP8', 'PO7', 'PO8', 'Fpz', 'CPz', 'POz', 'Oz'
]

# Load an evoked file as an example
file_path = r"D:\LYW\pre10\data\7evoked_allWords\prex006-ave.fif"
evoked_example = mne.read_evokeds(file_path, proj=False, verbose=None)
evoked_example = evoked_example[0].resample(91)  # Resample to 91 Hz
evoked_example.pick_channels(channels_all)  # Select a subset of channels

# Compute adjacency for the chosen channels
info = evoked_example.info
adjacency, ch_names = find_ch_adjacency(info, ch_type="eeg")

# Set parameters related to cluster analysis
max_step = 1
t_power = 1
n_tests = 59 * 91
n_times = 91

# Setup adjacency structure required for TFCE calculations
adjacency2 = cluster_level_backup._setup_adjacency(adjacency, n_tests, n_times)

# Directory containing the CSV files for permutation summary
csv_dir = r"D:\LYW\pre10\data\permutation_summary_between"
# csv_files = glob.glob(os.path.join(csv_dir, "summary_results_permutation_*.csv"))
# exclude updated clusters
csv_files = glob.glob(os.path.join(csv_dir, "summary_results_permutation_[0-9]*.csv"))
# Filter out anything that contains "_updated"
csv_files = [f for f in csv_files
             if "_updated" not in os.path.basename(f)
             and "updated_clustersT" not in os.path.basename(f)]

# TFCE threshold parameters
threshold_tfce = {
    "start": 0,
    "step": 0.001,
    "h_power": 2,
    "e_power": 0.5
}

# Tail parameter (set here to 1, meaning massed < spaced)
tail = 1


# For debugging
print(f"adjacency2: {type(adjacency2)}")
print(f"n_tests: {type(n_tests)} - {n_tests}")
print(f"n_times: {type(n_times)} - {n_times}")
print(f"threshold_tfce: {type(threshold_tfce)} - {threshold_tfce}")

'''
read the csv file including coef and std
so you need to calculate and smooth the t values using the coef and std.
then correct the t to tfce.
'''
def process_csv_file(csv_file_path, adjacency2, test_iterations, sigma, threshold_tfce, max_step, t_power):
    """
    Process a single CSV file:
    1. Compute permutation-based t-values using apply_lmer_models_to_array_parallel_hat.
    2. Compute TFCE  for both t_obs and -t_obs.
    """
    print(f"Processing {csv_file_path}")
    print(f"csv_file_path: {type(csv_file_path)} - {csv_file_path}")
    print(f"n_tests: {type(n_tests)} - {n_tests}")
    print(f"n_times: {type(n_times)} - {n_times}")
    print(f"threshold_tfce: {type(threshold_tfce)} - {threshold_tfce}")

    # Compute permutation-based t-values
    # this version just load coef and std from csv file and don't fit model
    t_obs_orig_permu = stuff.apply_lmer_models_to_array_parallel_hat(
        array_of_dfs=None,
        test_iterations=test_iterations,
        max_workers=5,#useless
        sigma=sigma,
        csv_file_path=csv_file_path#
    )

    # Calculate max and min values of t_obs_orig_permu
    max_t_obs = np.max(t_obs_orig_permu)
    min_t_obs = np.min(t_obs_orig_permu)
    # print the max and min values of t_obs_orig_permu
    print(f"Max value of t_obs_orig_permu: {max_t_obs}")
    print(f"Min value of t_obs_orig_permu: {min_t_obs}")

    # Define number of steps
    # use same number of steps rather than same step size.
    num_steps = cluster_level_backup.num_steps

    # Calculate step for positive and negative threshold_tfce
    step_positive = max_t_obs / num_steps
    step_negative = abs(min_t_obs) / num_steps

    # Update threshold_tfce for positive and negative directions
    threshold_tfce_positive = {
        "start": 0,
        "step": step_positive,
        "h_power": 2,
        "e_power": 0.5
    }

    threshold_tfce_negative = {
        "start": 0,
        "step": step_negative,
        "h_power": 2,
        "e_power": 0.5
    }

    print(f"Step for positive threshold_tfce: {step_positive}")
    print(f"Step for negative threshold_tfce: {step_negative}")

    # Find clusters for the positive side
    out = cluster_level_backup._find_clusters(
        x=t_obs_orig_permu,
        threshold=threshold_tfce_positive,
        tail=1,
        adjacency=adjacency2,
        max_step=max_step,
        include=None,
        partitions=None,
        t_power=t_power,
        show_info=True,
    )

    # Find clusters for the negative side
    out2 = cluster_level_backup._find_clusters(
        x=-t_obs_orig_permu, # transfer the sign of t_obs_orig_permu
        threshold=threshold_tfce_negative,
        tail=1,
        adjacency=adjacency2,
        max_step=max_step,
        include=None,
        partitions=None,
        t_power=t_power,
        show_info=True,
    )

    return csv_file_path, {'out': out, 'out2': out2}

'''

Calculate H0 distribution of tfce for all permutation.

procedure:

1. load all csv files
2. calculate tfce for all csv files
3. find the max and min values of tfce


'''

from datetime import datetime
results = {}

if __name__ == "__main__":
    # Reload custom module once more if needed
    reload(stuff)

    # Parameters
    max_step = 1
    t_power = 1
    n_tests = 59 * 91
    n_times = 91
    # file_limit = 3 # Number of files to process 490
    file_limit = len(csv_files)
    max_workers = 10

    num_steps = cluster_level_backup.num_steps

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_csv_file,
                csv_file,
                adjacency2,
                5733,      # test_iterations
                0.001,     # sigma
                threshold_tfce,
                max_step,
                t_power
            ): csv_file for csv_file in csv_files[:file_limit]
        }

        # Collect results as they complete
        for future in as_completed(futures):
            csv_file_path, result = future.result()
            results[csv_file_path] = result
            print(f"Completed processing {csv_file_path}")

        #     # Save results to CSV
        # for csv_file_path, result in results.items():
        #     # Convert results to a DataFrame
        #     out = pd.DataFrame(result['out'])
        #     out2 = pd.DataFrame(result['out2'])
        #
        #     # Calculate max and min for the second row of 'out'
        #     second_row_out = out.iloc[1]
        #     max_value_out = second_row_out.max()
        #     min_value_out = second_row_out.min()
        #
        #     # Calculate max and min for the second row of 'out2'
        #     second_row_out2 = out2.iloc[1]
        #     max_value_out2 = second_row_out2.max()
        #     min_value_out2 = second_row_out2.min()
        #
        #     print(f"Max value in the second row of 'out': {max_value_out}")
        #     print(f"Min value in the second row of 'out': {min_value_out}")
        #     print(f"Max value in the second row of 'out2': {max_value_out2}")
        #     print(f"Min value in the second row of 'out2': {min_value_out2}")

        # Initialize lists to store min and max values
        max_values_out = []
        min_values_out = []
        max_values_out2 = []
        min_values_out2 = []

        # Collect results
        for csv_file_path, result in results.items():

            # Convert results to a DataFrame
            # out structure
            # out have two row, the second row is tfce values, so you just need to get the max and min of the second row.
            out = pd.DataFrame(result['out'])
            out2 = pd.DataFrame(result['out2'])

            # Calculate max and min for the second row of 'out'
            second_row_out = out.iloc[1]
            max_values_out.append(second_row_out.max())
            min_values_out.append(second_row_out.min())

            # Calculate max and min for the second row of 'out2'
            second_row_out2 = out2.iloc[1]
            max_values_out2.append(second_row_out2.max())
            min_values_out2.append(second_row_out2.min())

            print(f"Max value in the second row of 'out': {max_values_out[-1]}")
            print(f"Min value in the second row of 'out': {min_values_out[-1]}")
            print(f"Max value in the second row of 'out2': {max_values_out2[-1]}")
            print(f"Min value in the second row of 'out2': {min_values_out2[-1]}")

        # # Save accumulated min and max values to CSV files
        # pd.DataFrame(max_values_out, columns=['max_value_out']).to_csv(os.path.join(csv_dir, "max_values_out.csv"),
        #                                                                index=False)
        # pd.DataFrame(min_values_out, columns=['min_value_out']).to_csv(os.path.join(csv_dir, "min_values_out.csv"),
        #                                                                index=False)
        # pd.DataFrame(max_values_out2, columns=['max_value_out2']).to_csv(os.path.join(csv_dir, "max_values_out2.csv"),
        #                                                                  index=False)
        # pd.DataFrame(min_values_out2, columns=['min_value_out2']).to_csv(os.path.join(csv_dir, "min_values_out2.csv"),
        #                                                                  index=False)


        # # Save accumulated min and max values to CSV files with num_steps in the filename
        # max_out_filename = f"max_values_out_num_steps_{num_steps}.csv"
        # min_out_filename = f"min_values_out_num_steps_{num_steps}.csv"
        # max_out2_filename = f"max_values_out2_num_steps_{num_steps}.csv"
        # min_out2_filename = f"min_values_out2_num_steps_{num_steps}.csv"

        # Get the current date in the desired format
        current_date = datetime.now().strftime("%Y%m%d")

        # Define the filenames with date and file_limit information
        max_out_filename = f"max_values_out_{current_date}_num_steps_{num_steps}_file_limit_{file_limit}.csv"
        min_out_filename = f"min_values_out_{current_date}_num_steps_{num_steps}_file_limit_{file_limit}.csv"
        max_out2_filename = f"max_values_out2_{current_date}_num_steps_{num_steps}_file_limit_{file_limit}.csv"
        min_out2_filename = f"min_values_out2_{current_date}_num_steps_{num_steps}_file_limit_{file_limit}.csv"

        pd.DataFrame(max_values_out, columns=['max_value_out']).to_csv(os.path.join(csv_dir, max_out_filename), index=False)
        pd.DataFrame(min_values_out, columns=['min_value_out']).to_csv(os.path.join(csv_dir, min_out_filename), index=False)
        pd.DataFrame(max_values_out2, columns=['max_value_out2']).to_csv(os.path.join(csv_dir, max_out2_filename), index=False)
        pd.DataFrame(min_values_out2, columns=['min_value_out2']).to_csv(os.path.join(csv_dir, min_out2_filename), index=False)

print("Done.")
print(results)
