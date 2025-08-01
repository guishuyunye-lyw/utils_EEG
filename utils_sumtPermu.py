
'''
why use p value rather than t value to find clusters?

notice: you cannot use t-threshold to find clusters, because there is no constant t threshold for 
each temporal-spatial point.
For each temporal-spatial point, the t-threshold is different.
Because the degree of freedom is different for linear mixed model!

So you read the rds file to load summary results, and extract the p value for each temporal-spatial point.
we have to use the p value to find clusters.


----

calculate the permutation sum-t

prodedure:
1 calculate sum-t
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


# Modify the process_csv_file function to use sum-t clustering
# include
def calculate_sumT_from_csv_file(csv_file_path, adjacency2, test_iterations,
                     sigma, threshold_sum_t, max_step, t_power):

    print(f"Processing {csv_file_path}")

    '''
    why don't you use cluster_level_backup.spatio_temporal_cluster_test() directly?
    because you didn't add a new parameter to read csv file.
    '''

    # Compute t based on the csv file
    # using coef_estimate and std_error from csv file
    # and sigma to smooth the t values.
    # I should change the function name to calculate_t_values_from_csv_file。。。
    # smoothing..
    # if you don't need smoothing, then set sigma = 0 or load from csv file directly
    t_obs_orig_permu = stuff.apply_lmer_models_to_array_parallel_hat(
        array_of_dfs=None,
        test_iterations=test_iterations,#useless
        max_workers=5,# useless,no parallel process in this version function.
        sigma=sigma,
        csv_file_path=csv_file_path # default csv file path: summary_results_2024年11月14日.csv
    )

    # load p-values from csv file
    data = pd.read_csv(csv_file_path)
    # Extract the p-values column
    p_values = data['p_value'].values

    # Calculate max and min values of t_obs_orig_permu
    max_t_obs = np.max(t_obs_orig_permu)
    min_t_obs = np.min(t_obs_orig_permu)
    print(f"Max value of t_obs_orig_permu: {max_t_obs}")
    print(f"Min value of t_obs_orig_permu: {min_t_obs}")

    # Find clusters using sum-t method
    # I have to change the function to use p-values instead of t-values for the clustering.
    out = cluster_level_backup._find_clusters_TorP(
        x=t_obs_orig_permu,
        threshold=threshold_sum_t,
        tail=tail,
        adjacency=adjacency2,
        max_step=max_step,
        include=None,
        partitions=None,
        t_power=t_power,
        show_info=True,
        # additional
        threshold_type='p',
        p_values=p_values
    )
    clustersT, clusterT_sum = out

    return csv_file_path, {'out': out} # if you want all cluster information, you can return it.
    # # Calculate min and max of cluster sums
    # min_cluster_sum = np.min(clusterT_sum)
    # max_cluster_sum = np.max(clusterT_sum)
    #
    # return csv_file_path, min_cluster_sum, max_cluster_sum


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
csv_files = glob.glob(os.path.join(csv_dir, "summary_results_permutation_*.csv"))
print(len(csv_files))

# original value file
csv_files = ["D:\LYW\pre10\data\permutation_summary_between\summary_results_2024年11月14日_upaded.csv"]

# Define a fixed threshold for sum-t clustering
 # Example threshold value, not like tfce, we have to set it manually.
 # not threshold free!
threshold_sum_t = 2.58
threshold_sum_t = 0.01 # if you set threshold_type='p',

# Tail parameter (set here to 1, meaning massed < spaced)
tail = 0

# sigma = 0.001  # Set the smoothing parameter
sigma = 0.001  # Set the smoothing parameter
# sigma = 0 # I don't need smoothing for sum-t clustering

# For debugging
print(f"adjacency2: {type(adjacency2)}")
print(f"n_tests: {type(n_tests)} - {n_tests}")
print(f"n_times: {type(n_times)} - {n_times}")
print(f"threshold_tfce: {type(threshold_sum_t)} - {threshold_sum_t}")


test_iterations = 5733 # useless

# one shot test
#
out = calculate_sumT_from_csv_file(
    csv_files[0],
    adjacency2,
    test_iterations,
    sigma,
    threshold_sum_t,
    max_step,
    t_power,
)
# Assuming out is a tuple (clustersT, clusterT_sum)
clustersT, clusterT_sum = out

# Convert to DataFrame
df = pd.DataFrame({
    'Cluster Indices': clustersT,
    'Cluster Sums': clusterT_sum
})
# Specify the file path where you want to save the CSV
output_csv_path = "D:\LYW\pre10\data\permutation_summary_between\summary_results_2024年11月14日_upaded_clustersT.csv"
# Write the DataFrame to a CSV file
df.to_csv(output_csv_path, index=False)


# Initialize a list to store results
results_list = []

max_workers = 10
# Process files in parallel

# # process all permutation csv files
# # main function because I want to use the function in parallel.
# if __name__ == '__main__':
#     # Directory containing the CSV files for permutation summary
#     csv_dir = r"D:\LYW\pre10\data\permutation_summary_between"
#     csv_files = glob.glob(os.path.join(csv_dir, "summary_results_permutation_*.csv"))
#
#     # Parameters
#     max_step = 1
#     t_power = 1
#     threshold_sum_t = 0.01  # Example threshold value
#     sigma = 0.001
#     test_iterations = 5733  # Example value
#     max_workers = 10
#     file_limit = len(csv_files)  # Limit the number of files to process
#
#     # Limit the number of files to process
#     csv_files_to_process = csv_files[:file_limit]
#
#     # Initialize a list to store results
#     results_list = []
#
#     # Process files in parallel
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = {
#             executor.submit(
#                 calculate_sumT_from_csv_file,
#                 csv_file,
#                 adjacency2,
#                 test_iterations,
#                 sigma,
#                 threshold_sum_t,
#                 max_step,
#                 t_power
#             ): csv_file for csv_file in csv_files_to_process
#         }
#
#         # Collect results as they complete
#         for future in as_completed(futures):
#             csv_file_path, min_cluster_sum, max_cluster_sum = future.result()
#             # Append the result to the list
#             results_list.append({
#                 'csv_file_path': csv_file_path,
#                 'min_cluster_sum': min_cluster_sum,
#                 'max_cluster_sum': max_cluster_sum
#             })
#             print(f"Completed processing {csv_file_path}")
#
#     # Convert results to a DataFrame
#     df_results = pd.DataFrame(results_list)
#
#     # Save the DataFrame to a CSV file
#     output_csv_path = os.path.join(csv_dir, "sum_t_cluster_sumsMaxMin_p0.01.csv")
#     df_results.to_csv(output_csv_path, index=False)
#
#     print("Results saved to:", output_csv_path)