#Import Dependencies
import numpy as np
import pandas as pd
import copy
import signac
from ast import literal_eval

from .GPBO_Classes_New import *
from .GPBO_Class_fxns import * 
import pickle
import gzip
import json

def open_file_helper(file_path):
    """
    Opens a .gz or .pickle file based on the extension
    
    Parameters
    ----------
    file_path: str, The file path of the data
    
    Returns
    -------
    results: pickled object, The results stored in the .pickle or .gz file
    """
    
    if file_path.endswith('.pickle'):
        with open(file_path, 'rb') as fileObj:
            results = pickle.load(fileObj) 
    elif file_path.endswith('.gz'):
        with gzip.open(file_path, 'rb') as fileObj:
            results = pickle.load(fileObj)          
    else:
        raise Warning("File type must be .gz or .pickle!")
    fileObj.close()
    
    return results

def make_dir_name_from_criteria(criteria_dict):
    """
    Makes a directory string name from a criteria dictionary
    """
    
    #Organize Dictionary keys and values sorted from lowest to highest
    sorted_dict = dict(sorted(criteria_dict.items(), key=lambda item: (item[0], item[1])))
    
    #Make list of parts
    parts = []
    for key, value in criteria_dict.items():
        if isinstance(value, dict):
            # Recursively format nested dictionaries
            nested_path = make_dir_name_from_criteria(value)
            parts.append(f"{key.replace('$', '')}_{nested_path}")
        elif isinstance(value, list):
            # Format lists as a string without square brackets and commas
            list_str = "_".join(map(str, value))
            parts.append(f"{key.replace('$', '')}_{list_str}")
        else:
            parts.append(f"{key.replace('$', '')}_{value}")

    return "/".join(parts)

def get_df_all_jobs(criteria_dict, save_csv = False):
    """
    Creates a dataframe of all information for a given experiment
    
    Parameters
    ----------
    criteria_dict: dict, Signac statepoints to consider for the job. Should include minimum of cs_name_val and param_name_str
    save_csv: bool, whether to save csvs. Default False
    
    Returns
    -------
    df_all_jobs: A dataframe of the all of the data for the given dictionary
    job_list: list, a list of jobs from Signac that fit criteria dict for the methods in meth_name_val_list
    theta_true: np.ndarray, True values of the case study parameters
    
    """
    
    assert isinstance(criteria_dict, dict), "criteria_dict must be a dictionary"
    assert isinstance(save_csv, bool), "save_csv must be boolean"
    
    #Get project
    project = signac.get_project()
    
    #Initialize data_created as False
    data_created = False

    #Intialize dataframe and job list for all jobs in criteria_dict
    df_all_jobs = pd.DataFrame()
    job_list = []
    
    #Find all jobs of a certain cs and method type for the criteria in order of job id
    jobs = sorted(project.find_jobs(criteria_dict), key=lambda job: job._id)
    
    #Loop over each job
    for job in jobs:
        assert os.path.exists(job.fn("BO_Results.gz")), "File must exist!" 
        data_file = job.fn("BO_Results.gz")
        #Add job to job list
        job_list += [job]

        #create workspace directory for data files if it doesn't already exists
        data_direc = job.fn("analysis_data")
        os.makedirs(data_direc, exist_ok=True)
        tab_data_path = os.path.join(data_direc , "tabulated_data.csv")

        # # #See if result data exists, if so add it to df
        if os.path.exists(tab_data_path):
            df_job = pd.read_csv(tab_data_path, index_col=0)

        #Otherwise, create it
        else:
            data_created = True
            df_job, theta_true = get_study_data_signac(job, save_csv = save_csv)
            
        #Add job dataframe to dataframe of all jobs
        df_all_jobs = pd.concat([df_all_jobs, df_job], ignore_index=False)

    #Reset index on df_all_jobs after adding all rows 
    df_all_jobs = df_all_jobs.reset_index(drop=True)     
    
    #Open Datafile to get theta_true if necessary
    if data_created == False:
        with gzip.open(data_file, 'rb') as fileObj:
            results = pickle.load(fileObj)   
        fileObj.close()
        theta_true = results[0].simulator_class.theta_true
        
    return df_all_jobs, job_list, theta_true

def get_study_data_signac(job, save_csv = False):
    """
    Get best data from jobs and optionally save the csvs for the data
    
    Parameters
    ----------
    criteria_dict: dictionary, Criteria for jobs to analyze   
    save_csv: bool, Whether or not to save csv data from analysis
    
    Returns
    -------
    df: pd.DataFrame, Dataframe containing the results from the study given a case study and method name
    study_id: str "ep" or "sf", whether to analyze data for the 
    
    """
    #Initialize df for a single job
    df_job = pd.DataFrame()
    data_file = job.fn("BO_Results.gz")

    #Open the file and get the dataframe
    with gzip.open(data_file, 'rb') as fileObj:
        results = pickle.load(fileObj)   
    fileObj.close()

    #Back out number of workflow restarts
    tot_runs = results[0].configuration["Number of Workflow Restarts"]
    
    #get theta_true from 1st run since it never changes within a case study
    theta_true = results[0].simulator_class.theta_true

    #Loop over runs
    for run in range(tot_runs):
        #Read data as pd.df
        df_run = results[run].results_df
        #Add the EP enum value as a column
        col_vals = job.sp.ep_enum_val
        df_run['EP Method Val'] = Ep_enum(int(col_vals)).name
        #Use job's run number if it has one
        if "bo_run_num" in job.statepoint():
            df_run["index"] = job.sp.bo_run_num 
        #Or use the run number from tot_run when tot_runs > 0
        elif tot_runs > 1:
            #Number of runs is the (seed # of the run - initial seed )/2 if tot_runs > 1 
            df_run["index"] = int(run + 1)
        else:
            #Otherwise it is (seed # of the run - 1 )/2 if tot_runs = 1 (Old job initialization system)
            df_run["index"] = int((results[run].configuration["Seed"] - 1)/2 + 1 )
        #Add other important columns
        df_run["BO Method"] = Method_name_enum(job.sp.meth_name_val).name
        df_run["Job ID"] = job.id
        
        df_run["Max Evals"] = len(df_run)
        try:
            df_run["Termination"] = results[run].why_term
        except:
            pass
        df_run["Total Run Time"] = df_run["Time/Iter"]*df_run["Max Evals"]  

        #Set BO and run numbers as columns        
        df_run.rename(columns={'index': 'Run Number'}, inplace=True)   
        df_run.insert(1, "BO Iter", df_run.index + 1)
        
        #Add run dataframe to job dataframe after
        df_job = pd.concat([df_job, df_run], ignore_index=False)

    #Set number of runs in job in the table
    # df_job["Runs in Job"] = tot_runs
    #Reset index on job dataframe
    df_job = df_job.reset_index(drop=True)
    # print(df_job.head())

    # print(os.path.join(job.fn("analysis_data"), "tabulated_data.csv"))
    #Put in a csv file in a directory based on the job
    if save_csv: 
        file_name1 = os.path.join(job.fn("analysis_data"), "tabulated_data.csv")
        df_job.to_csv(file_name1) 

    return df_job, theta_true


def get_best_data(criteria_dict, df = None, jobs = None, theta_true = None, save_csv = False):
    """
    Given all data from a study, find the best value
    
    Parameters
    ----------
    criteria_dict: dict, Signac statepoints to consider for the job. Should include minimum of cs_name_val and param_name_str
    meth_name_val_list: list, list of method numbers to consider. See /bo_methods_lib/GPBO_Classes_New.py Method_name_enum class
    save_csv: bool, Whether or not to save csv data from analysis. Set to false to avoid regernating data
    
    Returns
    -------
    df_best: pd.DataFrame, Dataframe containing the best result from the study given a case study and method name
    
    """
    #Get project
    project = signac.get_project()
    
    assert isinstance(criteria_dict, dict), "criteria_dict must be dictionary"
    if not all(var is not None for var in [df, jobs, theta_true]) == True:
        #Get data from Criteria dict if you need it
        df, jobs, theta_true = get_df_all_jobs(criteria_dict, save_csv)
    
    #Create a directory name based on the search criteria
    dir_name = "Results/" + make_dir_name_from_criteria(criteria_dict) + "/"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    #Check if csv already exists
    if os.path.exists(dir_name + "best_results.csv"):
        #If so, load the file
        filepath = dir_name + "best_results.csv"
        df_best = pd.read_csv(filepath, index_col=0)
        # df_best = pd.DataFrame(df.iloc[df.index.isin(best_indecies)])
        # df_found = True

    else:
        #See if relavent files exist
        for job in jobs:
            assert os.path.exists(job.fn("BO_Results.gz")), "File must exist!"
            data_file = job.fn("BO_Results.gz")
                
            #Analyze for best data
            #Initialize best idcs
            best_indecies = np.zeros(len(df['BO Method'].unique()))
            count = 0
            #Loop over methods, SFs/EPs/, and runs (capable of doing all or just 1 method)
            for meth in df['BO Method'].unique():
                #Loop over EPs or SFs and runs
                sse_best_overall = np.inf
                for param in df['EP Method Val'].unique():
                    for run in df['Run Number'].unique():                    
                        #Find the best sse at the end of the run. This is guaraneteed to be the lowest sse value found for that run
                        sse_run_best_value = df["Min Obj Cum."][(df['BO Method'] == meth) & (df['EP Method Val'] == param) & 
                                                                (df['Run Number'] == run)  & (df['BO Iter']+1 == df["Max Evals"]) ]

                        if isinstance(df["Min Obj Act"].iloc[0], np.ndarray):
                            sse_run_best_value = sse_run_best_value.iloc[0][0]
                        else:
                            sse_run_best_value = sse_run_best_value.iloc[0]

                        if sse_run_best_value < sse_best_overall:
                            #Set value as new best
                            sse_best_overall = sse_run_best_value
                            #Find the first instance where the minimum sse is found
                            index = df.index[(df['BO Method'] == meth) & (df["Run Number"] == run) & (df['EP Method Val'] == param) & 
                                                (df["Min Obj Act"] == sse_run_best_value)]

                            best_indecies[count] = index[0]
                count += 1


        #Make new df of only best single iter over all runs and SFs
        df_best = pd.DataFrame(df.iloc[df.index.isin(best_indecies)])

        #Calculate the L2 norm of the best runs
        df_best = calc_L2_norm(df_best, theta_true)
        
    #Put in order of method
    row_order = sorted([Method_name_enum[meth].value for meth in df['BO Method'].unique()])
    order = [Method_name_enum(num).name for num in row_order]

    # Reindex the DataFrame with the specified row order
    df_best['BO Method'] = pd.Categorical(df_best['BO Method'], categories=order, ordered=True)

    # Sort the DataFrame based on the categorical order
    df_best = df_best.sort_values(by='BO Method')
    
    #Get list of best jobs
    job_list_best = []
    job_id_list_best = list(df_best["Job ID"])
    for job_id in job_id_list_best:
        job = project.open_job(id=job_id)
        if job:
            job_list_best.append(job)
    
    if save_csv:
        #Save this as a csv in the same directory as all data
        #Create directory based on criteria dict
        file_name = os.path.join(dir_name, "best_results.csv")

        #Add file to directory 
        df_best.to_csv(file_name)
        
    return df_best, job_list_best

def get_median_data(criteria_dict, df = None, jobs = None, theta_true = None, save_csv = False):
    """
    Given data from a study, find the median value(s)
    
    Parameters
    ----------
    df: pd.DataFrame, dataframe including study data
    col_name: str, column name in the pandas dataframe to find the best value w.r.t
    theta_true: true parameter values from case study. Important for calculating L2 Norms
    save_csv: bool, Whether or not to save csv data from analysis
    
    Returns
    -------
    df_median: pd.DataFrame, Dataframe containing the median result from the study given a case study and method name
    
    """
    #Get project
    project = signac.get_project()
    
    assert isinstance(criteria_dict, dict), "criteria_dict must be dictionary"
    if not all(var is not None for var in [df, jobs, theta_true]) == True:
        #Get data from Criteria dict if you need it
        df, jobs, theta_true = get_df_all_jobs(criteria_dict, save_csv)
    
    #Create a directory name based on the search criteria
    dir_name = "Results/" + make_dir_name_from_criteria(criteria_dict) + "/"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    #Check if csv already exists
    if os.path.exists(dir_name + "median_results.csv"):
        #If so, load the file
        filepath = dir_name + "median_results.csv"
        df_median = pd.read_csv(filepath, index_col=0)
    else:
        #See if relavent files exist
        for job in jobs:
            assert os.path.exists(job.fn("BO_Results.gz")), "File must exist!"
            data_file = job.fn("BO_Results.gz")
            col_name = 'EP Method Val'

            #Get median values from df_best
            # Create a list containing 1 dataframe for each method in df_best
            df_list = []
            for meth in df['BO Method'].unique():
                df_meth = df[df["BO Method"]==meth]   
                df_list.append(df_meth)

            #Create new df for median values
            df_median = pd.DataFrame()

            #Loop over all method dataframes
            for df_meth in df_list:
                #Add the row corresponding to the median value of SSE to the list
                if isinstance(df["Min Obj Act"].iloc[0], np.ndarray):
                    median_sse = df_meth['Min Obj Act'].quantile(interpolation='nearest')[0]
                else:
                    median_sse = df_meth['Min Obj Act'].quantile(interpolation='nearest')

                df_median = pd.concat([df_median,df_meth[df_meth['Min Obj Act'] == median_sse]])

        #Calculate the L2 Norm for the median values
        df_median = calc_L2_norm(df_median, theta_true)  

    #Put in order of method
    row_order = sorted([Method_name_enum[meth].value for meth in df['BO Method'].unique()])
    order = [Method_name_enum(num).name for num in row_order]

    # Reindex the DataFrame with the specified row order
    df_median['BO Method'] = pd.Categorical(df_median['BO Method'], categories=order, ordered=True)

    # Sort the DataFrame based on the categorical order
    df_median = df_median.sort_values(by='BO Method')
    
    #Get list of best jobs
    job_list_med = []

    job_id_list_med = list(df_median["Job ID"])
    for job_id in job_id_list_med:
        job = project.open_job(id=job_id)
        if job:
            job_list_med.append(job)

    # print(dir_name + "median_results.csv")
    if save_csv:
        #Save this as a csv in the same directory as all data
        #Create directory based on criteria dict
        file_name = os.path.join(dir_name, "median_results.csv")

        #Add file to directory 
        df_median.to_csv(file_name)
        
    return df_median, job_list_med

def get_mean_data(criteria_dict, df = None, jobs = None, theta_true = None, save_csv = False):
    """
    Given data from a study, find the mean value(s)
    
    Parameters
    ----------
    df: pd.DataFrame, dataframe including study data
    col_name: str, column name in the pandas dataframe to find the best value w.r.t
    theta_true: true parameter values from case study. Important for calculating L2 Norms
    save_csv: bool, Whether or not to save csv data from analysis
    
    Returns
    -------
    df_median: pd.DataFrame, Dataframe containing the median result from the study given a case study and method name
    
    """
    #Get project
    project = signac.get_project()
    
    assert isinstance(criteria_dict, dict), "criteria_dict must be dictionary"
    if not all(var is not None for var in [df, jobs, theta_true]) == True:
        #Get data from Criteria dict if you need it
        df, jobs, theta_true = get_df_all_jobs(criteria_dict, save_csv)
    
    #Create a directory name based on the search criteria
    dir_name = "Results/" + make_dir_name_from_criteria(criteria_dict) + "/"
    
    #Initialize data_created as False
    df_found = False
    
    #See if relavent files exist
   

    #Check if csv already exists
    if os.path.exists(dir_name + "mean_results.csv"):
        #If so, load the file
        filepath = dir_name + "mean_results.csv"
        df_mean = pd.read_csv(filepath, index_col=0)

    else:
        for job in jobs:
            assert os.path.exists(job.fn("BO_Results.gz")), "File must exist!"
            data_file = job.fn("BO_Results.gz")
            col_name = 'EP Method Val'
            #Get median values from df_best
            # Create a list containing 1 dataframe for each method in df_best
            df_list = []
            for meth in df['BO Method'].unique():
                df_meth = df[df["BO Method"]==meth]   
                df_list.append(df_meth)

            #Create new df for median values
            df_mean = pd.DataFrame()

            #Loop over all method dataframes
            for df_meth in df_list:
                #Add the row corresponding to the median value of SSE to the list
                if isinstance(df["Min Obj Act"].iloc[0], np.ndarray):
                    #Find true mean
                    df_true_mean = df_meth["Min Obj Act"].mean()[0]
                else:
                    #Find true mean
                    df_true_mean = df_meth["Min Obj Act"].mean()

                #Find point closest to true mean
                df_closest_to_mean = df_meth.iloc[(df_meth["Min Obj Act"]-df_true_mean).abs().argsort()[:1]]
                #Add mean min and max points to dfs
                df_mean = pd.concat([df_mean, df_closest_to_mean])

        #Calculate the L2 Norm for the median values
        df_mean = calc_L2_norm(df_mean, theta_true)  

    #Put in order of method
    row_order = sorted([Method_name_enum[meth].value for meth in df['BO Method'].unique()])
    order = [Method_name_enum(num).name for num in row_order]

    # Reindex the DataFrame with the specified row order
    df_mean['BO Method'] = pd.Categorical(df_mean['BO Method'], categories=order, ordered=True)

    # Sort the DataFrame based on the categorical order
    df_mean = df_mean.sort_values(by='BO Method')
    
    #Get list of best jobs
    job_list_mean = []

    job_id_list_mean = list(df_mean["Job ID"])
    for job_id in job_id_list_mean:
        job = project.open_job(id=job_id)
        if job:
            job_list_mean.append(job)

    # print(dir_name + "mean_results.csv")
    if save_csv:
        #Save this as a csv in the same directory as all data
        #Create directory based on criteria dict
        file_name = os.path.join(dir_name, "mean_results.csv")

        #Add file to directory 
        df_mean.to_csv(file_name)
        
    return df_mean, job_list_mean

def calc_L2_norm(df, theta_true):
    """
    Calculates the L2 norm of Theta Values in a Pandas DataFrame
    
    Parameters
    ----------
    df: pd.DataFrame, The original dataframe containing the parameters you want to calculate the L2 norm for
    theta_true: ndarray, The true values of the parameters
    
    Returns
    -------
    df: pd.DataFrame, The original dataframe containing the L2 norm values of the parameters
    """
    #Calculate the difference between the true values and the GP best values in the dataframe for each parameter    
    def string_to_array(s):
        try:
            return np.array(eval(s), dtype=np.float64)
        except (SyntaxError, NameError):
            return s
    
    # Apply the function to the DataFrame column   
    try:
        #If the values are not being read as strings this works
        theta_min_obj = np.array(list(df['Theta Min Obj'].to_numpy()[:]), dtype=np.float64)
    except:
        #Otherwise, turn the theta values into a list and manually format the strings to be arrays
        thetas_as_list = np.array(df['Theta Min Obj']).tolist()
        theta_min_obj = np.array([list(map(float, s.strip('[]').split())) for s in thetas_as_list])

    del_theta = theta_min_obj - theta_true
    theta_L2_norm = np.zeros(del_theta.shape[0])
    for i in range(del_theta.shape[0]):
        theta_L2_norm[i] = np.linalg.norm(del_theta[i,:], ord = 2)
        
    df["L2 Norm Theta"] = theta_L2_norm

    return df

def analyze_hypers(file_path, save_csv = False):
    """
    Collects and plots hyperparameter data
    
    Parameters
    ----------
    file_path: str, The file path of the data
    run_num: int, The run you want to analyze. Note, run_num 1 corresponds to index 0
    
    Returns
    -------
    hps: np.ndarray, The hyperparameter data for plotting
    hp_names: str, the names of the hyperparameters
    hp_true: np.ndarray or None, the true values of the hyperparameters
    """
    #Check to see if file already exists
    org_dir_name = os.path.dirname(file_path)
    hp_dir_name =  os.path.join(org_dir_name, "analysis_data")
    hp_data_file = os.path.join(hp_dir_name, "hp_data.npy")
    hp_data_name_file = os.path.join(hp_dir_name, "hp_names.npy")

    # print([hp_file_path for hp_file_path in [hp_data_file, hp_data_name_file]])
    
    #If data is saved, load it and use it
    if all(os.path.exists(hp_file_path) for hp_file_path in [hp_data_file, hp_data_name_file]):
        hps = np.load(hp_data_file)
        hp_names = np.load(hp_data_name_file)
        hp_true = None
        
    #Otherwise generate and save it  
    else:            
        loaded_results = open_file_helper(file_path)
        runs = loaded_results[0].configuration["Number of Workflow Restarts"]
        num_sets = loaded_results[0].configuration["Max BO Iters"]
        dim_hps = len(loaded_results[0].list_gp_emulator_class[0].trained_hyperparams[0]) + 2
        hps = np.zeros((runs, num_sets, dim_hps))
        hp_names = [f"\\ell_{i}" for i in range(1, dim_hps+1)]
        hp_names[-2] = "\sigma"
        hp_names[-1] = "\\tau"
        hp_true = None

        for j in range(runs):
            run = loaded_results[j]
            for i in range(len(run.list_gp_emulator_class)):
               # Extract the array and convert other elements to float
                array_part = run.list_gp_emulator_class[i].trained_hyperparams[0]
                rest_part = np.array(run.list_gp_emulator_class[i].trained_hyperparams[1:], dtype=float)
                hp = np.concatenate([array_part, rest_part])
                # Create the resulting array of shape (1, 10)
                hps[j,i,:] = hp

        if save_csv:
            #Make directory if it doesn't exist
            if not os.path.exists(hp_dir_name):
                os.makedirs(hp_dir_name)
            
            #Save data
            np.save(hp_data_file, hps)
            np.save(hp_data_name_file, hp_names) 

    return hps, hp_names, hp_true

def analyze_sse_min_sse_ei(file_path, z_choices, save_csv = False):
    """
    Gets the data into an array for any comination of sse, log_sse, and ei
    
    Parameters
    ----------
    file_path: str, The file path of the data
    value_names: list of str, the values to plot. In order, sse, min_sse, and ei
    
    Returns
    -------
    data: np.ndarray, The data for plotting
    data_true: np.ndarray or None, the true values of the data
    """
    if isinstance(z_choices, str):
        z_choices = [z_choices]
    assert isinstance(z_choices, list), "z_choices must be list of string. List must contain at least 'ei' or 'sse'"
    assert all(isinstance(item, str) for item in z_choices), "z_choices elements must be string"
    assert any(item in z_choices for item in ["ei", "min_sse", "sse"]), "z_choices must contain at least 'min_sse', 'ei', or 'sse'"
    
    strings_for_df = [] 
    data_names = []
    
    for z_choice in z_choices:
        if "sse" == z_choice:
            strings_for_df += ["Min Obj Act"]
            data_names += ["\mathbf{e(\\theta)}"]
        if "min_sse" == z_choice:
            strings_for_df += ["Min Obj Cum."]
            data_names += ["\mathbf{Min\,e(\\theta)}"]        
        if "ei" == z_choice:
            strings_for_df += ["Max EI"]
            data_names += ["\mathbf{Max\,EI(\\theta)}"]
            
    #Get method value from json file
    org_dir_name = os.path.dirname(file_path)
    with open(org_dir_name+ "/signac_statepoint.json", 'r') as json_file:
        # Load the JSON data
        enum_method = json.load(json_file)["meth_name_val"]
        
    #Check to see if files already exist  
    dir_name =  os.path.join(org_dir_name, "analysis_data", "z_choice")
    data_files = [os.path.join(dir_name, z_choice + "_data.npy") for z_choice in z_choices]
    
    # print(data_files)

    #If data is saved, load it and use it
    if all(os.path.exists(data_file) for data_file in data_files):
        data = np.array([np.load(data_file) for data_file in data_files])
        data_true = None
            
    else:
        loaded_results = open_file_helper(file_path)
        runs = loaded_results[0].configuration["Number of Workflow Restarts"]
        num_sets = loaded_results[0].configuration["Max BO Iters"]

        #Get Method from loaded results
        enum_method = loaded_results[0].configuration["Method Name Enum Value"]
        meth_name = Method_name_enum(enum_method)
        method = GPBO_Methods(meth_name)

        dim_data = len(z_choices) #obj, min obj, and ei
        data = np.zeros((runs, num_sets, dim_data))
        data_true = None

        #Loop over runs
        for j in range(runs):
            run = loaded_results[j]
            #Loop over iterations
            for i in range(len(run.results_df["Min Obj Act"])):
                #Loop over types of data to extract
                # Extract the array and convert other elements to float
                for k in range(len(strings_for_df)):
                    #For 2B and 1B, need exp values for sse
                    if method.obj.value == 2 and "Obj" in strings_for_df[k]:
                        data[j,i,k] = np.exp(run.results_df[strings_for_df[k]].to_numpy().astype(float)[i])
                    else:
                        data[j,i,k] = run.results_df[strings_for_df[k]].to_numpy().astype(float)[i]
        #Save each piece of the data matrix separately
        if save_csv:
            #Make directory if it doesn't exist
            os.makedirs(dir_name, exist_ok = True)
            for k in range(data.shape[-1]):
               #Save data
               np.save(data_files[k], data[:,:,k]) 
            
    return data, data_names, data_true, enum_method

def analyze_thetas(file_path, z_choice, save_csv = False):
    """
    Gets the data into an array for thetas corresponding to the minimum sse at each iteration
    
    Parameters
    ----------
    file_path: str, The file path of the data
    run_num: int, The run you want to analyze. Note, run_num 1 corresponds to index 0
    string_for_df_theta: str, String corresponding to a column name in a saved pandas dataframe
    
    Returns
    -------
    data: np.ndarray, The parameter values for plotting
    data_names_names: str, the names of the parameter values
    data_true: np.ndarray or None, the true parameter values
    
    """
    
    assert isinstance(z_choice, str), "z_choices must be string"
    assert any(item == z_choice for item in ["ei", "min_sse", "sse"]), "z_choice must be 'min_sse', 'ei', or 'sse'"

    if "min_sse" in z_choice:
        string_for_df_theta = "Theta Min Obj Cum."  
    if "sse" == z_choice:
        string_for_df_theta = "Theta Min Obj"
    if "ei" in z_choice:
        string_for_df_theta = "Theta Max EI"
        
    #Check to see if file already exists
    org_dir_name = os.path.dirname(file_path)
    dir_name =  os.path.join(org_dir_name, "analysis_data", "z_choice")
    data_file = os.path.join(dir_name, z_choice + "_param_data.npy")
    data_true_file = os.path.join(dir_name, "true_param.npy")
    data_name_file = os.path.join(dir_name, "param_names.npy")
    
    # print([data_file_path for data_file_path in [data_file, data_name_file, data_true_file]])
    
    #If data is saved, load it and use it
    if all(os.path.exists(data_file_path) for data_file_path in [data_file, data_name_file, data_true_file]):
        data = np.load(data_file)
        data_names = np.load(data_name_file)
        data_true = np.load(data_true_file)
            
    else:
        loaded_results = open_file_helper(file_path)
        runs = loaded_results[0].configuration["Number of Workflow Restarts"]
        num_sets = loaded_results[0].configuration["Max BO Iters"]
        dim_data = len(loaded_results[0].results_df[string_for_df_theta].to_numpy()[0]) #len theta best

        data = np.zeros((runs, num_sets, dim_data))
        data_true = loaded_results[0].simulator_class.theta_true
        data_names = loaded_results[0].simulator_class.theta_true_names

        for j in range(runs):
            run = loaded_results[j]
            num_iters = len(run.results_df[string_for_df_theta])
            for i in range(num_iters):
                # Extract the array and convert other elements to float
                theta_min_obj = run.results_df[string_for_df_theta].to_numpy()[i]
                # Create the resulting array of shape (1, 10)
                data[j,i,:] = theta_min_obj
                
        if save_csv:
            #Make directory if it doesn't exist
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            #Save data
            np.save(data_file, data)
            np.save(data_name_file, data_names)
            np.save(data_true_file, data_true)
            
    data_names = [element.replace('theta', '\\theta') for element in data_names]
            
    return data, data_names, data_true

def analyze_train_test(file_path, run_num, bo_iter):
    """
    Gets the data into an array for thetas corresponding to the minimum sse at each iteration
    
    Parameters
    ----------
    file_path: str, The file path of the data
    run_num: int, The run you want to analyze. Note, run_num 1 corresponds to index 0
    bo_iter: int, The BO iteration you want to analyze. Note, bo_iter 1 corresponds to index 0
    
    Returns
    -------
    train_data: np.ndarray, The training parameter values for plotting
    test_data: np.ndarray, The testing parameter values for plotting
    val_data: np.ndarray, The validation parameter values for plotting
    data_names_names: str, the names of the parameter values
    data_true: np.ndarray or None, the true parameter values
    
    """
    run_num -= 1
    bo_iter -= 1
    loaded_results = open_file_helper(file_path)
    
    x_exp = loaded_results[run_num].exp_data_class.x_vals
    dim_data = loaded_results[run_num].list_gp_emulator_class[bo_iter].get_dim_gp_data() #dim training data
    data_true = loaded_results[run_num].simulator_class.theta_true

    param_names = loaded_results[run_num].simulator_class.theta_true_names
    x_names = [f"Xexp_{i}" for i in range(1, x_exp.shape[1]+1)]
    data_names = param_names+x_names

    train_data = loaded_results[run_num].list_gp_emulator_class[bo_iter].feature_train_data
    test_data = loaded_results[run_num].list_gp_emulator_class[bo_iter].feature_test_data
    if loaded_results[run_num].list_gp_emulator_class[bo_iter].gp_val_data is not None:
        val_data = loaded_results[run_num].list_gp_emulator_class[bo_iter].feature_val_data
    else:
        val_data = None
    
    return train_data, test_data, val_data, x_exp, data_names, data_true

def analyze_heat_maps(file_path, run_num, bo_iter, pair_id, log_data, get_ei = False, save_csv =False):
    """
    Gets the heat map data necessary for plotting heat maps
    
    Parameters
    ----------
    file_path: str, The file path of the data
    run_num: int, The run you want to analyze. Note, run_num 1 corresponds to index 0
    bo_iter: int, The BO iteration you want to analyze. Note, bo_iter 1 corresponds to index 0
    pair_id: int or str, The pair of data parameters
    log_data: bool, Wheteher to log transform data
    get_ei: bool, default False: Determines whether to calculate EI
    
    Returns
    -------
    all_data: ndarray, containing sse_sim, sse_mean, sse_var, and ei data 
    test_mesh: The meshgrid over which all_data was evaluated
    theta_true: ndarray, The true parameter values
    theta_opt: ndarray, the parameter values at the lowest sse over all bo iters
    theta_next: ndarray, the parameter value at the maximum ei
    train_theta: ndarray, Training parameter values
    param_names: list, The names of the parameters
    """
    #Check if data already exists, if so, just use it
    org_dir_name = os.path.dirname(file_path)
    dir_name = os.path.join(org_dir_name, "analysis_data", "gp_evaluations", "run_" + str(run_num), "pair_" + str(pair_id))
    os.makedirs(dir_name, exist_ok=True)
    
    #save heat_map_data as a pickle file instead of sse_sim, sse_mean, sse_var, and ei. Skip Ei gen step if ei isn't none
    hm_data_file = os.path.join(dir_name, "hm_data.gz")
    hm_sse_data_file = os.path.join(dir_name, "hm_sse_data.gz")
    param_info_file = os.path.join(dir_name, "param_info.npy")

    data_file_list = [hm_data_file, hm_sse_data_file, param_info_file]
    
    # print([file_name for file_name in data_file_list])
    
    if not all(os.path.exists(data_file_path) for data_file_path in data_file_list):
        run_num -= 1
        bo_iter -= 1
        loaded_results = open_file_helper(file_path)

        #Create Heat Map Data for a run and iter

        #Regeneate simulator, gp_emulator, exerimental data, best error, true theta, lowest obj theta, and highest ei theta
        gp_emulator = loaded_results[run_num].list_gp_emulator_class[bo_iter]
        exp_data = loaded_results[run_num].exp_data_class
        simulator = loaded_results[run_num].simulator_class

        enum_method = loaded_results[run_num].configuration["Method Name Enum Value"]
        enum_ep = Ep_enum(loaded_results[run_num].configuration["Exploration Bias Method Value"])
        ep_at_iter = loaded_results[run_num].results_df["Exploration Bias"].iloc[bo_iter]
        ep_bias = Exploration_Bias(None, ep_at_iter, enum_ep, None, None, None, None, None, None, None)

        cs_params, method, gen_meth_theta = get_driver_dependencies_from_results(loaded_results, run_num)

        if loaded_results[run_num].heat_map_data_dict is not None:
            heat_map_data_dict = loaded_results[run_num].heat_map_data_dict
        else:
            driver = GPBO_Driver(cs_params, method, simulator, exp_data, gp_emulator.gp_sim_data, gp_emulator.gp_sim_data, gp_emulator.gp_val_data, gp_emulator.gp_val_data, gp_emulator, ep_bias, gen_meth_theta)
            loaded_results[run_num].heat_map_data_dict = driver.create_heat_map_param_data()
            heat_map_data_dict = loaded_results[run_num].heat_map_data_dict

        #Get pair ID
        if isinstance(pair_id, str):
            param_names = pair_id
        elif isinstance(pair_id, int):
            param_names = list(loaded_results[run_num].heat_map_data_dict.keys())[pair_id]
        else:
            raise Warning("Invalid pair_id!")

        heat_map_data = heat_map_data_dict[param_names]    

        #Get index of param set
        idcs_to_plot = [loaded_results[run_num].simulator_class.theta_true_names.index(name) for name in param_names]   
        best_error =  loaded_results[run_num].results_df["Best Error"].iloc[bo_iter]
        if method.emulator == False:
            #Type 1 best error is inferred from training data 
            best_error, be_theta = gp_emulator.calc_best_error()
            best_errors_x = None
        else:
            #Type 2 best error must be calculated given the experimental data
            best_error, be_theta, best_errors_x = gp_emulator.calc_best_error(method, exp_data)
        best_error_metrics = (best_error, be_theta, best_errors_x)
        theta_true = loaded_results[run_num].simulator_class.theta_true
        theta_opt =  loaded_results[run_num].results_df["Theta Min Obj Cum."].iloc[bo_iter]
        theta_next = loaded_results[run_num].results_df["Theta Max EI"].iloc[bo_iter]
        train_theta = loaded_results[run_num].list_gp_emulator_class[bo_iter].train_data.theta_vals
        sep_fact = loaded_results[run_num].configuration["Separation Factor"]
        seed = loaded_results[run_num].configuration["Seed"]    
        meth_name = Method_name_enum(enum_method)
        method = GPBO_Methods(meth_name)    

        #Calculate GP mean and var for heat map data
        featurized_hm_data = gp_emulator.featurize_data(heat_map_data)
        heat_map_data.gp_mean, heat_map_data.gp_var = gp_emulator.eval_gp_mean_var_misc(heat_map_data, featurized_hm_data)

        #If not in emulator form, rearrange the data such that y_sim can be calculated
        if method.emulator == False:
            #Rearrange the data such that it is in emulator form
            n_points = int(np.sqrt(heat_map_data.get_num_theta())) #Since meshgrid data is always in grid form this gets num_points/param
            repeat_x = n_points**2 #Square because only 2 values at a time change
            x_vals = np.vstack([exp_data.x_vals]*repeat_x) #Repeat x_vals n_points**2 number of times
            repeat_theta = exp_data.get_num_x_vals() #Repeat theta len(x) number of times
            theta_vals =  np.repeat(heat_map_data.theta_vals, repeat_theta , axis =0) #Create theta data repeated
            #Generate full data class
            heat_map_data = Data(theta_vals, x_vals, None,heat_map_data.gp_mean,heat_map_data.gp_var,None,None,None,
                                 simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)

        #Calculate y and sse values
        heat_map_data.y_vals = simulator.gen_y_data(heat_map_data, 0 , 0)
        heat_map_sse_data = simulator.sim_data_to_sse_sim_data(method, heat_map_data, exp_data, sep_fact, gen_val_data = False)

        #Calculate SSE, SSE var, and EI with GP
        if method.emulator == False:
            heat_map_data.sse, heat_map_data.sse_var = gp_emulator.eval_gp_sse_var_misc(heat_map_data)            
        else:
            heat_map_data.sse, heat_map_data.sse_var = gp_emulator.eval_gp_sse_var_misc(heat_map_data, method, exp_data)

    else:
        heat_map_data = open_file_helper(hm_data_file)
        heat_map_sse_data = open_file_helper(hm_sse_data_file)
        param_info_dict = np.load(param_info_file)
        theta_true = param_info_dict["true"]
        theta_opt = param_info_dict["min_sse"]
        theta_next = param_info_dict["max_ei"]
        train_theta = param_info_dict["train"]
        param_names = param_info_dict["names"]
        idcs_to_plot = param_info_dict["idcs"]
           
    if get_ei and heat_map_data.ei is None:
        if method.emulator == False:
            heat_map_data.ei = gp_emulator.eval_ei_misc(heat_map_data, exp_data, ep_bias, best_error_metrics)[0]
        else:
            try:
                sg_depth = loaded_results[run_num].configuration["Sparse Grid Depth"]
                heat_map_data.ei = gp_emulator.eval_ei_misc(heat_map_data, exp_data, ep_bias, best_error_metrics, method, sg_depth)[0]
            except:
                heat_map_data.ei = gp_emulator.eval_ei_misc(heat_map_data,exp_data,ep_bias,best_error_metrics,method,sg_depth =10)[0]  
    elif not get_ei:
        ei = None
    
    #Define original theta_vals (for restoration later)
    org_theta = heat_map_data.theta_vals
    #Redefine the theta_vals in the given Data class to be only the 2D (varying) parts you want to plot
    heat_map_data.theta_vals = heat_map_data.theta_vals[:,idcs_to_plot]
    #Create a meshgrid with x and y values fron the uniwue theta values of that array
    unique_theta = heat_map_data.get_unique_theta()
    theta_pts = int(np.sqrt(len(unique_theta)))
    test_mesh = unique_theta.reshape(theta_pts,theta_pts,-1).T
    heat_map_data.theta_vals = org_theta
    
    param_info_dict = {"true":theta_true, "min_sse":theta_opt, "max_ei":theta_next, "train":train_theta, "names":param_names, "idcs":idcs_to_plot}
    
    if save_csv:
        fileObj = gzip.open(hm_data_file, 'wb', compresslevel  = 1)
        pickled_results = pickle.dump(heat_map_data, fileObj)
        fileObj.close()
        fileObj = gzip.open(hm_sse_data_file, 'wb', compresslevel  = 1)
        pickled_results = pickle.dump(heat_map_data, fileObj)
        fileObj.close()
        np.save(param_info_file, param_info_dict)
        
    #Define sse_sim, sse_gp_mean, and sse_gp_var, and ei based on whether to report log scaled data
    sse_sim = heat_map_sse_data.y_vals
    sse_var = heat_map_data.sse_var
    sse_mean = heat_map_data.sse

    #Get log or unlogged data values        
    if log_data == False:
        #Change sse sim, mean, and stdev to not log for 1B and 2B
        if method.obj.value == 2:
            #SSE variance is var*(e^((log(sse)))^2
            sse_mean = np.exp(sse_mean)
            sse_var = (sse_var*sse_mean**2)      
            sse_sim = np.exp(sse_sim)
        if get_ei:
            ei = heat_map_data.ei.reshape(theta_pts,theta_pts).T

    #If getting log values
    else:
        #Get log data from 1A, 2A, and 2C
        if method.obj.value == 1:            
            #SSE Variance is var/sse**2
            sse_var = sse_var/sse_mean**2
            sse_mean = np.log(sse_mean)
            sse_sim = np.log(sse_sim)
        if get_ei:
            ei = np.log(heat_map_data.ei).reshape(theta_pts,theta_pts).T

    #Reshape data to correct shape and add to list to return
    reshape_list = [sse_sim, sse_mean, sse_var]     
#     all_data = [var.reshape(theta_pts,theta_pts,-1).T for var in reshape_list] + [ei]
    all_data = [var.reshape(theta_pts,theta_pts).T for var in reshape_list] + [ei]
    
    return all_data, test_mesh, param_info_dict

def analyze_xy_plot(file_path, run_num, bo_iter, x_lin_pts):
    """
    Gets the data necessary for plotting x vs y. Type 2 GP Only
    
    Parameters
    ----------
    file_path: str, The file path of the data
    run_num: int, The run you want to analyze. Note, run_num 1 corresponds to index 0
    bo_iter: int, The BO iteration you want to analyze. Note, bo_iter 1 corresponds to index 0
    pair_id: int or str, The pair of data parameters
    
    Returns
    -------
    theta_opt_data: Instance of Data, class containing data relavent to theta_opt, the parameter value at the lowest sse over all bo iters
    exp_data: Instance of Data, Class containing experimental data
    train_data: Instance of Data, Class containing GP training data
    test_data Instance of Data, Class containing GP testing data
    """ 
    run_num -= 1
    bo_iter -= 1
    loaded_results = open_file_helper(file_path)
    
    #get exp_data and theta_opt
    exp_data = loaded_results[run_num].exp_data_class
    gp_emulator = loaded_results[run_num].list_gp_emulator_class[bo_iter]
    simulator = loaded_results[run_num].simulator_class
    sep_fact = loaded_results[run_num].configuration["Separation Factor"]
    seed = loaded_results[run_num].configuration["Seed"]
    theta_opt =  loaded_results[run_num].results_df["Theta Min Obj Cum."].iloc[bo_iter]
    
    #Make a Data class instance with just the values you want to plot
    #Generate exp_data that is pretty
    gen_meth_x = Gen_meth_enum(2)
    exp_data_lin = simulator.gen_exp_data(x_lin_pts, gen_meth_x)

    #Repeat the theta best array once for each x value
    #Need to repeat theta_best such that it can be evaluated at every x value in exp_data using simulator.gen_y_data
    theta_opt_repeated = np.repeat(theta_opt.reshape(1,-1), exp_data_lin.get_num_x_vals() , axis =0)
    #Add instance of Data class to theta_best
    theta_opt_data = Data(theta_opt_repeated, exp_data_lin.x_vals, None, None, None, None, None, None, 
                          simulator.bounds_theta_reg, simulator.bounds_x, sep_fact, seed)
    #Calculate y values and sse for theta_best with noise
    theta_opt_data.y_vals = simulator.gen_y_data(theta_opt_data, simulator.noise_mean, simulator.noise_std)  
    #Calculate GP mean and var for heat map data
    featurized_to_data = gp_emulator.featurize_data(theta_opt_data)
    theta_opt_data.gp_mean, theta_opt_data.gp_var = gp_emulator.eval_gp_mean_var_misc(theta_opt_data, featurized_to_data)
    train_data = loaded_results[run_num].list_gp_emulator_class[bo_iter].train_data
    test_data = loaded_results[run_num].list_gp_emulator_class[bo_iter].test_data
    
    return theta_opt_data, exp_data, train_data, test_data

def get_driver_dependencies_from_results(loaded_results, run_num):
    """
    builds instance of CaseStudyParameters from saved file data
    """
    simulator = loaded_results[run_num].simulator_class
    configuration = loaded_results[run_num].configuration
    method = GPBO_Methods(Method_name_enum(configuration["Method Name Enum Value"]))
    cs_name = configuration["Case Study Name"]
    ep0 = loaded_results[run_num].results_df["Exploration Bias"].iloc[0]
    sep_fact = configuration["Separation Factor"]
    normalize = configuration["Normalize"]
    kernel = configuration["Initial Kernel"]
    lenscl = configuration["Initial Lengthscale"]
    outputscl = configuration["Initial Outputscale"]
    retrain_GP = configuration["Retrain GP"]
    reoptimize_obj = configuration["Reoptimize Obj"]
    gen_heat_map_data = configuration["Heat Map Points Generated"]
    bo_iter_tot = configuration["Max BO Iters"]
    bo_run_tot = configuration["Number of Workflow Restarts"]
    save_data = False
    DateTime = configuration["DateTime String"]
    seed = configuration["Seed"]
    obj_tol = configuration["Obj Improvement Tolerance"]
    ei_tol = configuration["EI Tolerance"]
    if "Theta Generation Enum Value" in configuration.keys():
        gen_meth_theta = Gen_meth_enum(configuration["Theta Generation Enum Value"])
    else:
        gen_meth_theta = Gen_meth_enum(1)
    
    cs_params = CaseStudyParameters(cs_name, ep0, sep_fact, normalize, kernel, lenscl, outputscl, retrain_GP, reoptimize_obj, gen_heat_map_data, bo_iter_tot, bo_run_tot, save_data, DateTime, seed, obj_tol, ei_tol)
    
    return cs_params, method, gen_meth_theta
    
def compare_muller_heat_map(file_path, run_num, bo_iter, x_val_num, theta_choice, seed, gen_meth_theta = Gen_meth_enum(1)):
    """
    Compares simulation and GP data for the Muller potential over a heat map
    
    Parameters
    ----------
    file_path: str, The file path of the data
    run_num: int, The run you want to analyze. Note, run_num 1 corresponds to index 0
    bo_iter: int, The BO iteration you want to analyze. Note, bo_iter 1 corresponds to index 0
    x_val_num: int, The number of x values to make heat maps over in each dimension of x data
    theta_choice: 1D ndarray, or None, the theta_value to evaluate the heat map at. If none, chosen based off seed
    seed: int, the seed for theta_choice if applicable
    
    Returns
    -------
    test_mesh: ndarray, meshgrid of x values to generate the heat map 
    y_sim: ndarray, The simulated values for test_mesh
    gp_mean: ndarray, The gp mean values for test_mesh
    gp_var: ndarray, The gp variance values for test_mesh
    theta_value: ndarray, the parameter set evaluated
    exp_data.x_vals: ndarray, experimental x data
    idcs_to_plot: list of str, all parameter names
    
    """
    run_num -= 1
    bo_iter -= 1
    loaded_results = open_file_helper(file_path)
    #get exp_data and theta_opt
    exp_data = loaded_results[run_num].exp_data_class
    gp_emulator = loaded_results[run_num].list_gp_emulator_class[bo_iter]
    simulator = loaded_results[run_num].simulator_class
    sep_fact = loaded_results[run_num].configuration["Separation Factor"]
    method = GPBO_Methods(Method_name_enum(loaded_results[run_num].configuration["Method Name Enum Value"]))
    
    enum_ep = Ep_enum(loaded_results[run_num].configuration["Exploration Bias Method Value"])
    ep_at_iter = loaded_results[run_num].results_df["Exploration Bias"].iloc[bo_iter]
    ep_bias = Exploration_Bias(None, ep_at_iter, enum_ep, None, None, None, None, None, None, None)
    
    theta_obj_min =  loaded_results[run_num].results_df["Theta Min Obj Cum."].iloc[bo_iter]
    theta_ei_max = loaded_results[run_num].results_df["Theta Max EI"].iloc[bo_iter]
    train_theta = loaded_results[run_num].list_gp_emulator_class[bo_iter].train_data.theta_vals
    
    if loaded_results[run_num].heat_map_data_dict is not None:
        param_names = list(loaded_results[run_num].heat_map_data_dict.keys())[0]
    else:
        cs_params, method, gen_meth_theta = get_driver_dependencies_from_results(loaded_results, run_num)
        driver = GPBO_Driver(cs_params, method, simulator, exp_data, gp_emulator.gp_sim_data, gp_emulator.gp_sim_data, gp_emulator.gp_val_data, gp_emulator.gp_val_data, gp_emulator, ep_bias, gen_meth_theta)
        loaded_results[run_num].heat_map_data_dict = driver.create_heat_map_param_data()
        param_names = list(loaded_results[run_num].heat_map_data_dict.keys())[0]
        
    idcs_to_plot = [loaded_results[run_num].simulator_class.theta_true_names.index(name) for name in param_names]
    idcs_to_plot = [loaded_results[run_num].simulator_class.theta_true_names.index(name) for name in param_names]
    
    #Generate simulation data for x given 1 theta
    simulator.seed = seed
    sim_data_x = simulator.gen_sim_data(1, x_val_num, Gen_meth_enum(1), Gen_meth_enum(2), sep_fact, False)
    if theta_choice is not None:
        sim_data_x.theta_vals[:] = theta_choice
        sim_data_x.y_vals = simulator.gen_y_data(sim_data_x, 0, 0)
    
    theta_value = sim_data_x.theta_vals[0]        
    featurized_sim_x_data = gp_emulator.featurize_data(sim_data_x)
    
    sim_data_x.gp_mean, sim_data_x.gp_var = gp_emulator.eval_gp_mean_var_misc(sim_data_x, featurized_sim_x_data)
    
    #Create a meshgrid with x and y values fron the uniwue theta values of that array
    test_mesh = sim_data_x.x_vals.reshape(x_val_num, x_val_num,-1).T

    #Calculate valus
    y_sim = sim_data_x.y_vals.reshape(x_val_num, x_val_num).T
    gp_mean = sim_data_x.gp_mean.reshape(x_val_num, x_val_num).T
    gp_var = sim_data_x.gp_var.reshape(x_val_num, x_val_num).T
    
    if method.emulator == False and method.obj.value ==2:
        gp_mean = np.exp(sim_data_x.gp_mean.reshape(x_val_num, x_val_num).T)
        gp_var  =  np.exp(sim_data_x.gp_var.reshape(x_val_num, x_val_num).T)
    
    return test_mesh, y_sim, gp_mean, gp_var, theta_value, exp_data.x_vals, idcs_to_plot

def analyze_parity_plot_data(file_path, run_num, bo_iter):
    """
    Generates parity plot for testing data
    """
    run_num -= 1
    bo_iter -= 1
    loaded_results = open_file_helper(file_path)
    
    #get exp_data and theta_opt
    exp_data = loaded_results[run_num].exp_data_class
    gp_emulator = loaded_results[run_num].list_gp_emulator_class[bo_iter]
    simulator = loaded_results[run_num].simulator_class
    train_data = loaded_results[run_num].list_gp_emulator_class[bo_iter].train_data
    test_data = loaded_results[run_num].list_gp_emulator_class[bo_iter].test_data
    enum_method = loaded_results[run_num].configuration["Method Name Enum Value"]
    meth_name = Method_name_enum(enum_method)
    method = GPBO_Methods(meth_name)
    sep_fact = loaded_results[run_num].configuration["Separation Factor"]

    test_data.gp_mean, test_data.gp_var = gp_emulator.eval_gp_mean_var_test()

    if method.emulator == False:
        test_data.sse, test_data.sse_var = gp_emulator.eval_gp_sse_var_test()
        sse_data = copy.copy(test_data) 
        test_data_sse_data = None
    else:
        test_data.sse, test_data.sse_var = gp_emulator.eval_gp_sse_var_test(method, exp_data)
        test_data_sse_data = simulator.sim_data_to_sse_sim_data(method, test_data, exp_data, sep_fact, False)
        sse_data = None                           
                   
    return test_data, test_data_sse_data, sse_data, method

#NOTE: DO NOT USE THIS FXN UNTIL THE NORMALIZATION ISSUE IS FIXED IN IT
def analyze_param_sens(file_path, run_num, bo_iter, param_id, n_points):
    """
    Analyzes Parameter Sensitivity
    
    Parameters
    ----------
    file_path: str, The file path of the data
    run_num: int, The run you want to analyze. Note, run_num 1 corresponds to index 0
    bo_iter: int, The BO iteration you want to analyze. Note, bo_iter 1 corresponds to index 0
    param_id: int or str, The parameter name/index
    n_points: int, The number of points to evaluate the parameter over
    
    Returns
    -------
    param_data: Instance of Data, Class containing the data for the parameter sensitivity analysis
    param_idx: int, The index of the parameter being analyzed w.r.t theta_true
    param_name: str, The name of the parameter being analyzed
    data_name: str, The name contiaing the marker of how many training data were used for this analysis
    exp_data: Instance of Data, Class containing experimental data
    train_data: Instance of Data, Class containing GP training data
    test_data Instance of Data, Class containing GP testing data
    """
    run_num -= 1
    bo_iter -= 1
    loaded_results = open_file_helper(file_path)
        
    #get exp_data and theta_opt
    exp_data = loaded_results[run_num].exp_data_class
    gp_emulator = loaded_results[run_num].list_gp_emulator_class[bo_iter]
    simulator = loaded_results[run_num].simulator_class
    theta_true = loaded_results[run_num].simulator_class.theta_true
    train_data = loaded_results[run_num].list_gp_emulator_class[bo_iter].train_data
    test_data = loaded_results[run_num].list_gp_emulator_class[bo_iter].test_data
    enum_method = loaded_results[run_num].configuration["Method Name Enum Value"]
    meth_name = Method_name_enum(enum_method)
    method = GPBO_Methods(meth_name)
    sep_fact = loaded_results[run_num].configuration["Separation Factor"]
    seed = loaded_results[run_num].configuration["Seed"]

    #Find this paramete's index in theta_true
    if isinstance(param_id, str):
        param_idx = simulator.theta_true_names.index(param_id)
    elif isinstance(param_id, int):
        param_idx = param_id
    else:
        raise Warning("Invalid param_id!")
    bounds_param = simulator.bounds_theta_reg[:,param_idx]
    param_name = simulator.theta_true_names[param_idx]

    #Create new theta true array of n_points
    array_of_vals = np.tile(np.array(simulator.theta_true), (n_points, 1))
    #Change the value of the parameter you want to vary to a linspace of n_points between the parameter bounds
    varying_theta = np.linspace(bounds_param[0], bounds_param[1], n_points)
    array_of_vals[:,param_idx] = varying_theta
    #Give an x_value to evaluate at
    x_val = exp_data.x_vals[0] 
    x_data = np.vstack([x_val]*n_points)
    bounds_x = simulator.bounds_x
    #Create instance of Data Class
    param_data = Data(array_of_vals, x_data, None, None, None, None, None, None, bounds_param, bounds_x, sep_fact, seed)
    feat_param_data = gp_emulator.featurize_data(param_data)
    param_data.y_vals = simulator.gen_y_data(param_data, simulator.noise_mean, simulator.noise_std)  
    param_data.gp_mean, param_data.gp_var = gp_emulator.eval_gp_mean_var_misc(param_data, feat_param_data)
    data_name = "TP_" + str(train_data.get_num_theta())
    
    return param_data, param_idx, param_name, data_name, exp_data, train_data, test_data