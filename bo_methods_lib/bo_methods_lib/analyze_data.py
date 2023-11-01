#Import Dependencies
import numpy as np
import pandas as pd
import copy
import signac

from .GPBO_Classes_New import *
from .GPBO_Class_fxns import * 
import pickle
import gzip

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

def get_study_data_signac(criteria_dict, study_id, save_csv = False):
    """
    Get best ep or sf data from jobs and optionally save the csvs for the data
    
    Parameters
    ----------
    criteria_dict: dictionary, Criteria for jobs to analyze   
    study_id: str "ep" or "sf", whether to analyze data for the 
    save_csv: bool, Whether or not to save csv data from analysis
    
    Returns
    -------
    df: pd.DataFrame, Dataframe containing the results from the study given a case study and method name
    study_id: str "ep" or "sf", whether to analyze data for the 
    
    """
    project = signac.get_project()
    
    #Get method name and CS name
    cs_name_val = criteria_dict["cs_name_val"]
    param_name_str = criteria_dict["param_name_str"]
    meth_name_val = criteria_dict["meth_name_val"]
    
    
    meth_name = Method_name_enum(meth_name_val)
    cs_name_enum = CS_name_enum(cs_name_val)
    
    #For the ep study
    if study_id == "ep":
        col_name = 'EP Method Val'
        criteria_dict_ep = criteria_dict.copy()
        criteria_dict_ep["sep_fact"] = 1.0
        #Find all jobs of a certain cs and method type for the ep studies w/ SF = 1 in order of job id
        jobs = sorted(project.find_jobs(criteria_dict_ep), key=lambda job: job._id)
        
    elif study_id == "sf":
        col_name = 'Sep Fact'
    
        #Get best ep data from previous results if possible
        criteria_dict_ep = criteria_dict.copy()
        criteria_dict_ep["sep_fact"] = 1.0
        jobs_ep = project.find_jobs(criteria_dict_ep)
        
        #Note, this will only ever be 1 job
        for job in jobs_ep:
            path_name = job.fn("ep_study_best_all.csv")
        if os.path.exists(path_name):
            df_ep_best = pd.read_csv(path_name, index_col = 0, header = 0)
            
        #If there is no results path infer it directly from the jobs
        else:
            df_ep, jobs_ep_out, cs_name, theta_true = get_study_data_signac(criteria_dict, "ep", save_csv = False) 
            df_ep_best = get_best_data(df_ep, "ep", cs_name, theta_true, param_name_str, date_time_str = None, save_csv = False)
            
        #Set ep enum val to the best one for that cs and method
        best_ep_enum_val = int(df_ep_best["EP Method Val"][(df_ep_best['BO Method'] == meth_name.name)])
#         best_ep_enum_val = int(df_ep_best["EP Method Val"].iloc[0])
        criteria_dict_sf = criteria_dict.copy()
        criteria_dict_sf["ep_enum_val"] = best_ep_enum_val
        
        #Get all jobs with that ep enum val
        jobs = project.find_jobs(criteria_dict_sf)
    
    else:
        raise Warning("study_id must be ep or sf!")
    
    #Do analysis for study
    #Initialize df for all sf/ep method data for each case study and method
    df = pd.DataFrame()
    #Loop over all jobs of this category
    for job in jobs:
        assert os.path.exists(job.fn("BO_Results.gz")), "File must exist!"
        data_file = job.fn("BO_Results.gz")
        #Open the file and get the dataframe
        with gzip.open(data_file, 'rb') as fileObj:
            results = pickle.load(fileObj)   
        fileObj.close()
        tot_runs = results[0].configuration["Number of Workflow Restarts"]
        #Loop over runs
        for run in range(tot_runs):
            #Read data
            df_job = results[run].results_df
            #Add the value of the SF/EP enum as a column
            if study_id == "ep":
                col_vals = job.sp.ep_enum_val
            else:
                col_vals = job.sp.sep_fact
            df_job[col_name] = col_vals
            #Add other important columns
            df_job["index"] = run
            df_job["BO Method"] = meth_name.name
            df_job["Max Evals"] = len(df_job)
            df_job["Total Run Time"] = df_job["Time/Iter"]*df_job["Max Evals"]  
            #Add data to the dataframe with all the data
            df = pd.concat([df, df_job], ignore_index=False)

    #Set BO and run numbers as columns        
    df.rename(columns={'index': 'Run Number'}, inplace=True)   
    df.insert(1, "BO Iter", df.index)
    df = df.reset_index(drop=True)

    #get theta_true from 1st run since it never changes
    theta_true = results[0].simulator_class.theta_true
    #Put it in a csv file in a directory based on the method and case study
    if save_csv:
        #Make directory name
#         dir_name = "Results/" + study_id + "_study/" + cs_name_enum.name + "/" + param_name_str + "/" + meth_name.name
#         if not os.path.isdir(dir_name):
#             os.makedirs(dir_name)
#         file_name1 = dir_name + "/" + study_id + "_study_analysis.csv"
        file_name1 = job.fn(study_id + "_study_analysis.csv")
        df.to_csv(file_name1) 

    return df, jobs, cs_name_enum.name, theta_true

def get_study_data_org(date_time_str, name_cs_str, meth_name_str_list, study_id, study_param_list, save_csv = False):
    """
    Saves all results for Exploration Bias or Separation factor Study
    
    Parameters:
    -----------
    date_time_str: str, The DateTime string in format year/month/day/ (numbers only)
    name_cs_str: str, The case study name. Ex CS1
    meth_name_str_list: list of str, the BO method names to consider
    study_id: str, "EP" or "SF", Whether to get data for ep exp or sf exp
    study_param_list: list of str: Parmeters to consider
    save_csv: bool, Determines whether to print results or save them as a csv
    """
    
    #Get theta dimensions from any file
    if study_id == "ep":
        path_study_name = "_ep_method_"
        col_name = 'EP Method Val'
        csv_name = date_time_str + name_cs_str + "/Exploration_Bias_Data.csv"
    elif study_id == "sf":
        path_study_name = "_sep_fact_"
        col_name = 'Sep Fact'
        csv_name = date_time_str + name_cs_str + "/Separation_Factor_Data.csv"
    else:
        raise Warning("study_id must be 'ep' or 'sf'!")
        
    path = date_time_str + "Data_Files/" + name_cs_str + "_BO_method_" + meth_name_str_list[0] + path_study_name + study_param_list[0]
    try:
        with open(path + ".pickle", 'rb') as fileObj:
            results = pickle.load(fileObj) 
    except:
        with gzip.open(path + ".gz", 'rb') as fileObj:
            results = pickle.load(fileObj) 
    fileObj.close()

    # Create an empty target DataFrame
    all_result_df = pd.DataFrame()

    #Loop over methods
    for i in range(len(meth_name_str_list)):
        #Loop over ep_methods
        for j in range(len(study_param_list)):
            #Pull out file
            path = date_time_str+"Data_Files/"+name_cs_str + "_BO_method_" + meth_name_str_list[i] + path_study_name + study_param_list[j]
            try:
                with open(path + ".pickle", 'rb') as fileObj:
                    results = pickle.load(fileObj) 
            except:
                with gzip.open(path + ".gz", 'rb') as fileObj:
                    results = pickle.load(fileObj)   
            fileObj.close()
            theta_true = results[0].simulator_class.theta_true
            tot_runs = results[0].configuration["Number of Workflow Restarts"]
            #Loop over runs
            for k in range(tot_runs):
                #Get Results from pandas df and add more useful columns
                run_results = results[k].results_df
                run_results["index"] = k
                run_results[col_name] = study_param_list[j]
                run_results["BO Method"] = meth_name_str_list[i]
                run_results["Max Evals"] = len(run_results)
                run_results["Total Run Time"] = run_results["Time/Iter"]*run_results["Max Evals"]
                all_result_df = pd.concat([all_result_df, run_results], ignore_index=False)
                
    all_result_df.rename(columns={'index': 'Run Number'}, inplace=True)
    all_result_df.insert(1, "BO Iter", all_result_df.index)
    all_result_df = all_result_df.reset_index(drop=True)
                
    if save_csv == True:
        path_to_save_df = date_time_str + csv_name
        all_result_df.to_csv(path_to_save_df, index=True)
    
    return all_result_df, theta_true

def get_best_data(df, study_id, cs_name, theta_true, jobs = None, date_time_str = None, save_csv = False):
    """
    Given all data from a study, find the best value
    
    Parameters
    ----------
    df: pd.DataFrame, dataframe including study data
    col_name: str, column name in the pandas dataframe to find the best value w.r.t
    theta_true: true parameter values from case study. Important for calculating L2 Norms
    date_time_str: None or str, Saves to a datetime location instead of Results/ if not None
    save_csv: bool, Whether or not to save csv data from analysis
    
    Returns
    -------
    df_best: pd.DataFrame, Dataframe containing the best result from the study given a case study and method name
    
    """
    if study_id == "ep":
        col_name = 'EP Method Val'
    elif study_id == "sf":
        col_name = 'Sep Fact'
    else:
        raise Warning("study_id must be EP or SF!")
    
    #Analyze for best data
    #Initialize best idcs
    best_indecies = np.zeros(len(df['BO Method'].unique()))
    count = 0
    #Loop over methods, SFs/EPs/, and runs (capable of doing all or just 1 method)
    for meth in df['BO Method'].unique():
        #Loop over EPs or SFs and runs
        sse_best_overall = np.inf
        for param in df[col_name].unique():
            for run in df['Run Number'].unique():
                #Find the best sse at the end of the run. This is guaraneteed to be the lowest sse value found for that run
                sse_run_best_value = df["Min Obj Cum."][(df['BO Method'] == meth) & (df[col_name] == param) & 
                                                        (df['Run Number'] == run)  & (df['BO Iter']+1 == df["Max Evals"]) ]

                if isinstance(df["Min Obj Act"].iloc[0], np.ndarray):
                    sse_run_best_value = sse_run_best_value.iloc[0][0]
                else:
                    sse_run_best_value = sse_run_best_value.iloc[0]
                
                if sse_run_best_value < sse_best_overall:
                    #Set value as new best
                    sse_best_overall = sse_run_best_value
                    #Find the first instance where the minimum sse is found
                    index = df.index[(df['BO Method'] == meth) & (df["Run Number"] == run) & (df[col_name] == param) & 
                                     (df["Min Obj Act"] == sse_run_best_value)]

                    best_indecies[count] = index[0]
        count += 1


    #Make new df of only best single iter over all runs and SFs
    df_best = pd.DataFrame(df.iloc[df.index.isin(best_indecies)])
    
    #Calculate the L2 norm of the best runs
    df_best = calc_L2_norm(df_best, theta_true)
    
    if save_csv:
        #Save this as a csv in the same directory as all data
        #Make directory if it doesn't already exist
        if date_time_str is None:
            if len(jobs) > 1:
                file_name2 = [job.fn(study_id + "_study_best_all.csv") for job in jobs]
            else:
                file_name2 = [job.fn(study_id + "_study_best.csv") for job in jobs]
        else:
            dir_name =  date_time_str + study_id + "_study/" + cs_name
            
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
                
            file_name2 = [dir_name + "/" + study_id + "_study_best.csv"]
        #Add file to directory 
        for file in file_name2:
            df_best.to_csv(file)
        
    return df_best

def get_median_data(df, study_id, cs_name, theta_true, jobs = None, date_time_str = None, save_csv = False):
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
    if study_id == "ep":
        col_name = 'EP Method Val'
    elif study_id == "sf":
        col_name = 'Sep Fact'
    else:
        raise Warning("study_id must be EP or SF!")
        
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
    
    #Save or show df
    if save_csv:
        #Save this as a csv in the same directory as all data
        #Make directory if it doesn't already exist
        if date_time_str is None:
            file_name2 = [job.fn(study_id + "_study_median.csv") for job in jobs]
        else:
            dir_name =  date_time_str + study_id + "_study/" + cs_name
            file_name2 = dir_name + "/" + study_id + "_study_median.csv"
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        #Add file to directory        
        df_median.to_csv(file_name2)
#     if save_csv:
#         #Save this as a csv in the same directory as all data
#         #Make directory if it doesn't already exist
#         if date_time_str is None:
#             dir_name = "Results/" + study_id + "_study/" + cs_name + "/" + param_name_str + "/" + df['BO Method'].iloc[0]
#         else:
#             dir_name =  date_time_str + study_id + "_study/" + cs_name
#         if not os.path.isdir(dir_name):
#             os.makedirs(dir_name)
#         #Add file to directory
#         file_name2 = dir_name + "/" + study_id + "_study_median.csv"
#         df_median.to_csv(file_name2) 
        
    return df_median

def get_mean_data(df, study_id, cs_name, theta_true, jobs = None, date_time_str = None, save_csv = False):
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
    if study_id == "ep":
        col_name = 'EP Method Val'
    elif study_id == "sf":
        col_name = 'Sep Fact'
    else:
        raise Warning("study_id must be EP or SF!")
        
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
    
    #Save or show df
    if save_csv:
        #Save this as a csv in the same directory as all data
        #Make directory if it doesn't already exist
        if date_time_str is None:
            file_name2 = [job.fn(study_id + "_study_mean.csv") for job in jobs]
        else:
            dir_name =  date_time_str + study_id + "_study/" + cs_name
            file_name2 = dir_name + "/" + study_id + "_study_mean.csv"
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        #Add file to directory        
        df_mean.to_csv(file_name2)
        
    return df_mean

def get_mean_med_best_over_sf(df, cs_name, theta_true, job_list):
    df_list = []
    choices = ["mean", "median", "median_best", "best"]
    for choice in choices:        
        #Get df containing the best value at each sf for each method
        df_meth_sf = pd.DataFrame()
        names = df['BO Method'].unique()
        sep_fact_list = df['Sep Fact'].unique()
        #Loop over sfs
        for sf in range(len(sep_fact_list)):
            #Loop over names
            for name in names:
                df_meth = df[ (df["BO Method"]==name) & (df["Sep Fact"] == sep_fact_list[sf]) ]   
        #                 df_meth = df[(df["BO Method"]==name) & (df["Sep Fact"] == sep_fact_list[sf])]                  
                if choice == "mean":
                    df_piece = get_mean_data(df_meth, "sf", cs_name, theta_true, job_list, date_time_str = None, save_csv = False)
                elif choice == "median":
                    df_piece = get_median_data(df_meth, "sf", cs_name, theta_true, job_list, date_time_str = None, save_csv = False)
                    if len(df_piece) > 0:
                        df_piece = df_piece.iloc[0:1].reset_index(drop=True)
                elif choice == "median_best":
                    df_best = get_best_data(df_meth, "sf", cs_name, theta_true, job_list, date_time_str = None, save_csv = False)
                    df_piece = get_median_data(df_best, "sf", cs_name, theta_true, job_list, date_time_str = None, save_csv = False)
                elif choice == "best":
                    df_piece = get_best_data(df_meth, "sf", cs_name, theta_true, job_list, date_time_str = None, save_csv = False)
                df_meth_sf = pd.concat([df_meth_sf, df_piece])
        df_list.append(df_meth_sf)
        
    return df_list
                
        
    
def converter(instr):
    """
    Converts strings to arrays when loading pandas dataframes
    
    Parameters
    ----------
    instr: str, The string form of the array
    
    Returns 
    -------
    outstr: ndarry (object type), array of the string
    
    """
    outstr = np.fromstring(instr[1:-1],sep=' ')
    return outstr

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
    theta_min_obj = np.array(list(df['Theta Min Obj'].to_numpy()[:]), dtype=np.float64)
    del_theta = theta_min_obj - theta_true
    theta_L2_norm = np.zeros(del_theta.shape[0])
    for i in range(del_theta.shape[0]):
        theta_L2_norm[i] = np.linalg.norm(del_theta[i,:], ord = 2)
        
    df["L2 Norm Theta"] = theta_L2_norm
        
    return df
       
def analyze_SF_data_for_plot(df, meth_name_str, sep_fact_list):
    """
    Returns the results of the Separation Factor Study SSE data for plotting
    
    Parameters
    ----------
    df: pd.DataFrame, Dataframe of all SF data
    meth_name_str: str, String of the method name
    sep_fact_list: list of str, the Separation Factors to consider
    
    Returns
    -------
    y_data: ndarray (n_sfs x 2), The array of the gp predicted minimum sse and actual sse data
    data_names: list of str, the names of the gp predicted minimum sse and actual sse data
    """
    
    y_data = np.zeros((len(sep_fact_list) , 2))
    sse_min_all = np.ones(len(sep_fact_list))*np.inf
    sse_min_act_all = np.ones(len(sep_fact_list))*np.inf

    #Loop over sfs
    for i in range(len(sep_fact_list)):
        #Loop over runs
        for j in range(max(df["Run Number"].unique())):
            df_meth = df[(df["BO Method"]==meth_name_str) & (df["Sep Fact"] == sep_fact_list[i])]  
            #Find lowest sse and corresponding theta
            min_sse_index = np.argmin(df_meth['Min Obj Act']) #Should use Actual or GP min?
            min_sse = df_meth['Min Obj'].iloc[min_sse_index]
            min_sse_act = df_meth['Min Obj Act'].iloc[min_sse_index]

            if min_sse_act < sse_min_all[i]:
                sse_min_all[i] = min_sse
                sse_min_act_all[i] = min_sse_act

    if "B" in meth_name_str:
        sse_min_all = np.exp(sse_min_all)
        sse_min_act_all = np.exp(sse_min_act_all)
        
    y_data[:,0] = sse_min_all
    y_data[:,1] = sse_min_act_all
    data_names = ['Min Obj', 'Min Obj Act']
    
    return y_data, data_names

def analyze_hypers(file_path, run_num):
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
    run_num -= 1
    loaded_results = open_file_helper(file_path)
    runs = loaded_results[run_num].configuration["Number of Workflow Restarts"]
    num_sets = loaded_results[run_num].configuration["Max BO Iters"]
    dim_hps = len(loaded_results[run_num].list_gp_emulator_class[0].trained_hyperparams[0]) + 2
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
    
    return hps, hp_names, hp_true

def analyze_sse_min_sse_ei(file_path, run_num, strings_for_df):
    """
    Gets the data into an array for any comination of sse, log_sse, and ei
    
    Parameters
    ----------
    file_path: str, The file path of the data
    run_num: int, The run you want to analyze. Note, run_num 1 corresponds to index 0
    value_names: list of str, the values to plot. In order, sse, min_sse, and ei
    
    Returns
    -------
    data: np.ndarray, The data for plotting
    data_true: np.ndarray or None, the true values of the data
    """
    run_num -= 1
    loaded_results = open_file_helper(file_path)
    runs = loaded_results[run_num].configuration["Number of Workflow Restarts"]
    num_sets = loaded_results[run_num].configuration["Max BO Iters"]
    
    #Get Method from loaded results
    enum_method = loaded_results[run_num].configuration["Method Name Enum Value"]
    meth_name = Method_name_enum(enum_method)
    method = GPBO_Methods(meth_name)
    
    dim_data = len(strings_for_df) #obj, min obj, and ei
    data = np.zeros((runs, num_sets, dim_data))
    data_true = None

    for j in range(runs):
        run = loaded_results[j]
        # Extract the array and convert other elements to float
        for i in range(len(run.results_df["Min Obj Act"])):
            for k in range(len(strings_for_df)):
                #For 2B and 1B, need exp values for sse
                if method.obj.value == 2 and "Obj" in strings_for_df[k]:
                    data[j,i,k] = np.exp(run.results_df[strings_for_df[k]].to_numpy().astype(float)[i])
                else:
                    data[j,i,k] = run.results_df[strings_for_df[k]].to_numpy().astype(float)[i]
            
    return data, data_true

def analyze_thetas(file_path, run_num, string_for_df_theta):
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
    run_num -= 1
    loaded_results = open_file_helper(file_path)
    runs = loaded_results[run_num].configuration["Number of Workflow Restarts"]
    num_sets = loaded_results[run_num].configuration["Max BO Iters"]
    dim_data = len(loaded_results[run_num].results_df[string_for_df_theta].to_numpy()[0]) #len theta best

    data = np.zeros((runs, num_sets, dim_data))
    data_true = loaded_results[run_num].simulator_class.theta_true
    data_names = loaded_results[run_num].simulator_class.theta_true_names
    
    for j in range(runs):
        run = loaded_results[j]
        num_iters = len(run.results_df[string_for_df_theta])
        for i in range(num_iters):
            # Extract the array and convert other elements to float
            theta_min_obj = run.results_df[string_for_df_theta].to_numpy()[i]
            # Create the resulting array of shape (1, 10)
            data[j,i,:] = theta_min_obj
            
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

def analyze_heat_maps(file_path, run_num, bo_iter, pair_id, log_data, get_ei = False):
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
    
    if loaded_results[run_num].heat_map_data_dict is not None:
        heat_map_data_dict = loaded_results[run_num].heat_map_data_dict
    else:
        cs_params, method, gen_meth_theta = get_driver_dependencies_from_results(loaded_results, run_num)
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
    featurized_hm_data = gp_emulator.featurize_data(heat_map_data)
    
    #Get index of param set
    idcs_to_plot = [loaded_results[run_num].simulator_class.theta_true_names.index(name) for name in param_names]   
    best_error =  loaded_results[run_num].results_df["Best Error"].iloc[bo_iter]
    theta_true = loaded_results[run_num].simulator_class.theta_true
    theta_opt =  loaded_results[run_num].results_df["Theta Min Obj Cum."].iloc[bo_iter]
    theta_next = loaded_results[run_num].results_df["Theta Max EI"].iloc[bo_iter]
    train_theta = loaded_results[run_num].list_gp_emulator_class[bo_iter].train_data.theta_vals
    sep_fact = loaded_results[run_num].configuration["Separation Factor"]
    seed = loaded_results[run_num].configuration["Seed"]    
    meth_name = Method_name_enum(enum_method)
    method = GPBO_Methods(meth_name)    
    
    #Calculate GP mean and var for heat map data
    heat_map_data.gp_mean, heat_map_data.gp_var = gp_emulator.eval_gp_mean_var_misc(heat_map_data, featurized_hm_data)
    
    #If not in emulator form, rearrange the data such that y_sim can be calculated
    if method.emulator == False:
        #Rearrange the data such that it is in emulator form
        n_points = int(np.sqrt(heat_map_data.get_num_theta())) #Since meshgrid data is always in meshgrid form this gets num_points/param
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

    #Get log or unlogged data values        
    if log_data == False:
        #Change sse sim, mean, and stdev to not log for 1B and 2B
        if method.obj.value == 2:
            #SSE variance is var*(e^((log(sse)))^2
            heat_map_data.sse = np.exp(heat_map_data.sse)
            heat_map_data.sse_var = heat_map_data.sse_var*heat_map_data.sse**2            
            heat_map_sse_data.y_vals = np.exp(heat_map_sse_data.y_vals)
            
    #If getting log values
    else:
        #Get log data from 1A, 2A, and 2C
        if method.obj.value == 1:            
            #SSE Variance is var/sse**2
            heat_map_data.sse_var = heat_map_data.sse_var/heat_map_data.sse**2
            heat_map_data.sse = np.log(heat_map_data.sse)
            heat_map_sse_data.y_vals = np.log(heat_map_sse_data.y_vals)
            
    if get_ei:
        if method.emulator == False:
            heat_map_data.ei = gp_emulator.eval_ei_misc(heat_map_data, exp_data, ep_bias, best_error)
        else:
            heat_map_data.ei = gp_emulator.eval_ei_misc(heat_map_data, exp_data, ep_bias, best_error, method)   
        
    #Create test mesh
    #Define original theta_vals (for restoration later)
    org_theta = heat_map_data.theta_vals
    #Redefine the theta_vals in the given Data class to be only the 2D (varying) parts you want to plot
    heat_map_data.theta_vals = heat_map_data.theta_vals[:,idcs_to_plot]
    #Create a meshgrid with x and y values fron the uniwue theta values of that array
    unique_theta = heat_map_data.get_unique_theta()
    theta_pts = int(np.sqrt(len(unique_theta)))
    test_mesh = unique_theta.reshape(theta_pts,theta_pts,-1).T
    heat_map_data.theta_vals = org_theta

    sse_sim = heat_map_sse_data.y_vals.reshape(theta_pts,theta_pts).T
    sse_mean = heat_map_data.sse.reshape(theta_pts,theta_pts).T
    sse_var = heat_map_data.sse_var.reshape(theta_pts,theta_pts).T
    if get_ei:
        ei = heat_map_data.ei.reshape(theta_pts,theta_pts).T
    else:
        ei = None
    
    all_data = [sse_sim, sse_mean, sse_var, ei]
    
    return all_data, test_mesh, theta_true, theta_opt, theta_next, train_theta, param_names, idcs_to_plot

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
    normalize = simulator.normalize
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
    
    theta_true = loaded_results[run_num].simulator_class.theta_true
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
    theta_true = loaded_results[run_num].simulator_class.theta_true
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