import numpy as np
import pandas as pd
import signac
import signac
import json

from bo_methods_lib.bo_methods_lib.analyze_data import General_Analysis, open_file_helper
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import Plotters

#Ignore warnings
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)

#Set Stuff
meth_name_val_list = [1,2,3,4,5,6,7]
save_csv = True #Set to False if you don't want to save/resave csvs
save_figs = True
modes = ["act"]
project = signac.get_project("GPBO_Fix")

# Initialize dictionaries to store the condition numbers and statistics
dict_k = {}  # Stores raw condition numbers for each case study and BO method
dict_stats = {}  # Stores the statistics (log averages, min, max, median)

# Iterate through the different case study values
for val in [11,14,2,1,12,13,3,10]:
    criteria_dict = {
        "cs_name_val": val,
        "ep_enum_val": 1,
        "gp_package": "gpflow",
        "meth_name_val": {"$in": meth_name_val_list}
    }
    
    # Iterate through each mode
    for mode in modes:
        analyzer = General_Analysis(criteria_dict, project, mode, save_csv)
        plotters = Plotters(analyzer, save_figs)

        # Get all data from experiments
        df_all_jobs, job_list, theta_true_data = analyzer.get_df_all_jobs(save_csv)
        
        # Get best data from ep experiment
        df_best, job_list_best = analyzer.get_best_data()

        # Load the best GP from each method
        for i in range(len(job_list_best)):
            # Load job
            job = job_list_best[i]
            cs_name = df_best["CS Name"].iloc[i]
            bo_method = df_best["BO Method"].iloc[i]
            run_num = df_best["Run Number"].iloc[i]
            bo_iter = df_best["BO Iter"].iloc[i]
            
            # Open BO_Results_GPs.gz
            loaded_results = open_file_helper(job.fn("BO_Results_GPs.gz"))
            with open(job.fn("signac_statepoint.json"), 'r') as json_file:
                # Load the JSON data
                sp_data = json.load(json_file)
            run_num -= sp_data["bo_run_num"]
            bo_iter -= 1

            # Get the GP emulator
            try:
                gp_emulator = loaded_results[run_num].list_gp_emulator_class[bo_iter]
            except:
                print(len(loaded_results), run_num, bo_iter)
                print(len(loaded_results[run_num].list_gp_emulator_class))

            # Compute the condition number of the kernel matrix
            k = np.linalg.cond(gp_emulator.fit_gp_model.kernel(gp_emulator.feature_train_data))

            # Store the condition number under both case study and BO Method
            case_study_key = f"{cs_name}"
            method_key = f"{bo_method}"

            if (case_study_key, method_key) in dict_k:
                dict_k[(case_study_key, method_key)].append(np.log10(k))
            else:
                dict_k[(case_study_key, method_key)] = [np.log10(k)]

# Initialize a dictionary to store the condition numbers aggregated by BO Method
from collections import defaultdict
bo_method_dict = defaultdict(list)  # This will store condition numbers for each BO Method

# Aggregate condition numbers across all case studies for each BO Method
for (case_study, bo_method), values in dict_k.items():
    bo_method_dict[bo_method].extend(values)  # Collect all condition numbers for each BO Method

# Prepare the data for DataFrame and for dict_stats
data = {
    'BO Method': [],
    'Log10 Avg a': [],
    'Log10 Min a': [],
    'Log10 Max a': [],
    'Log10 Median a': []
}

# Process the aggregated condition numbers and compute stats for each BO Method
for bo_method, values in bo_method_dict.items():
    log_avg = np.mean(values)  # Compute the average
    log_min = np.min(values)   # Compute the minimum
    log_max = np.max(values)   # Compute the maximum
    log_median = np.median(values)  # Compute the median

    # Save the statistics in dict_stats (with just the BO Method key)
    dict_stats[bo_method] = {
        'Log10 Avg a': log_avg,
        'Log10 Min a': log_min,
        'Log10 Max a': log_max,
        'Log10 Median a': log_median
    }

    # Add the aggregated data to the DataFrame
    data['BO Method'].append(bo_method)
    data['Log10 Avg a'].append(log_avg)
    data['Log10 Min a'].append(log_min)
    data['Log10 Max a'].append(log_max)
    data['Log10 Median a'].append(log_median)

# Create a DataFrame from the data
df_results = pd.DataFrame(data)

# Display or save the DataFrame as needed
print(df_results)

# Optional: Save to CSV
df_results.to_csv('gpflow_condition_numbers.csv', index=False)


# Prepare the data for DataFrame conversion from dict_k
data_k = {
    'Case Study': [],
    'BO Method': [],
    'Log10 Condition Number (a)': []
}

# Process the condition numbers and add them to the DataFrame
for (case_study, bo_method), values in dict_k.items():
    for value in values:
        data_k['Case Study'].append(case_study)
        data_k['BO Method'].append(bo_method)
        data_k['Log10 Condition Number (a)'].append(value)

# Convert the data into a DataFrame
df_k = pd.DataFrame(data_k)

print(df_k)

# Save the DataFrame as a CSV file
df_k.to_csv('gpflow_condition_numbers_raw.csv', index=False)