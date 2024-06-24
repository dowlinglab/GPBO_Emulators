#Import Dependencies
import numpy as np
import pandas as pd
import copy
import signac
from ast import literal_eval
from collections.abc import Iterable

from .GPBO_Class_fxns import *
from .GPBO_Classes_New import *

import pickle
import gzip
import json
import ast
import re

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
    if file_path.endswith('.pickle') or file_path.endswith('.pkl'):
        with open(file_path, 'rb') as fileObj:
            results = pickle.load(fileObj) 
    elif file_path.endswith('.gz'):
        with gzip.open(file_path, 'rb') as fileObj:
            results = pickle.load(fileObj)        
    else:
        raise Warning("File type must be .gz or .pickle!")
    
    return results
class General_Analysis:
    """
    The base class for Gaussian Processes
    Parameters
    
    Methods
    --------------
    __init__
    make_dir_name_from_criteria
    get_jobs_from_criteria
    get_df_all_jobs
    get_study_data_signac
    str_to_array_df_col(self, str_arr)
    get_best
    get_median
    get_mean
    __get_job_list
    sort_by_meth
    __calc_L2_norm()
    __get_data
    load_data
    save_data
    __z_choice_helper
    __preprocess_analyze
    analyze_obj_vals
    analyze_thetas
    analyze_hypers
    __rebuild_cs
    analyze_parity_plot_data
    analyze_heat_maps
    """
    # Class variables and attributes
    
    def __init__(self, criteria_dict, project, mode, save_csv):
        """
        Parameters
        ----------
        criteria_dict: dict, Signac statepoints to consider for the job. Should include minimum of cs_name_val
        project: signac.project.Project, The signac project to analyze
        mode: str, the mode to analyze the data in (act, acq, or gp)
        save_csv: bool, whether to save csvs.
        """
        #Asserts
        assert isinstance(criteria_dict, dict), "criteria_dict must be a dictionary"
        assert isinstance(project, signac.project.Project), "project must be a signac.Project object"
        assert isinstance(save_csv, bool), "save_csv must be a boolean"
        assert mode in ["act", "acq", "gp"], "mode must be 'act', 'acq', or 'gp'"
        # Collect unique statepoints of all jobs
        statepoint_names = set()
        for job in project:
            statepoint_names.update(job.statepoint().keys())
        key_list = list(statepoint_names)
        self.sp_keys_valid = key_list
        # print(self.sp_keys_valid)
        # value = all(key in key_list for key in list(criteria_dict.keys()))
        # print(value)
        # print(all(key in key_list for key in list(criteria_dict.keys())))
        # print(list(criteria_dict.keys()))
        assert all(key in key_list for key in list(criteria_dict.keys())) == True, "All keys in criteria_dict must be in project statepoints"

        # Constructor method
        self.mode = mode
        self.criteria_dict = criteria_dict
        self.project = project
        self.study_results_dir = os.path.join(self.make_dir_name_from_criteria(self.criteria_dict))
        self.save_csv = save_csv
    
    def make_dir_name_from_criteria(self, dict_to_use, is_nested = False):
        """
        Makes a directory string name from a criteria dictionary

        Parameters
        ----------
        dict_to_use: dict, dictionary to use to make directory name
        is_nested: bool, whether the dictionary is nested or not

        Returns
        -------
        result_dir: str, the directory name from the dictionary

        Notes
        -----
        For proper use, ALWAYS use this function with is_nested = False
        """
        assert isinstance(dict_to_use, dict), "dict_to_use must be a dictionary"
        assert isinstance(is_nested, bool), "is_nested must be a boolean"
        #Note, criteria dict is only checked when is_nested = False
        if not is_nested:
            assert all(key in self.sp_keys_valid for key in list(dict_to_use.keys())) == True, "All keys in criteria_dict must be in project statepoints"
        #Organize Dictionary keys and values sorted from lowest to highest
        sorted_dict = dict(sorted(dict_to_use.items(), key=lambda item: (item[0], item[1])))
        
        #Make list of parts
        parts = []
        for key, value in dict_to_use.items():
            if isinstance(value, dict):
                # Recursively format nested dictionaries
                nested_path = self.make_dir_name_from_criteria(value, True)
                parts.append(f"{key.replace('$', '')}_{nested_path}")
            elif isinstance(value, list):
                # Format lists as a string without square brackets and commas
                list_str = "_".join(map(str, value))
                parts.append(f"{key.replace('$', '')}_{list_str}")
            else:
                parts.append(f"{key.replace('$', '')}_{value}")

        # result_dir = "/".join(parts) if is_nested else os.path.join("Results_acq", "/".join(parts))
        result_dir = "/".join(parts) if is_nested else os.path.join("Results2_" + self.mode, "/".join(parts))
        return result_dir

    def get_jobs_from_criteria(self, criteria_dict):
        """
        Gets a pointer of all jobs

        Returns
        -------
        jobs: list, a list of jobs from Signac that fit criteria dict
        """
        #Find all jobs of a certain cs and method type for the criteria in order of job id
        jobs = sorted(self.project.find_jobs(criteria_dict), key=lambda job: job._id)
        jobs = [job for job in jobs if os.path.exists(job.fn("BO_Results.gz"))]

        return jobs
    
    def str_to_array_df_col(self, str_arr):
        """
        Used to turn arrays from csvs loaded to pd dataframes from strings into arrays

        Parameters
        ----------
        str_arr: str, the string to turn into an array

        Returns
        -------
        array_from_str: np.ndarray, the array from the string
        """
        # Find the index of the first space
        first_space_index = str_arr.index(' ')
        # Remove the first space if its the 2nd character (the first will be [)
        if first_space_index == 1:
            str_no_space1 = str_arr[:first_space_index] + str_arr[first_space_index+1:]
        else:
            str_no_space1 = str_arr
        # Turn the string into an array be subbing spaces with a ,
        array_from_str = np.array(ast.literal_eval(re.sub(r'\s+', ',',str_no_space1)))
        return array_from_str

    def get_df_all_jobs(self, criteria_dict = None, save_csv = False):
        """
        Creates a dataframe of all information for a given experiment
        
        Parameters
        ----------
        save_csv: bool, whether to save csvs. Default False
        
        Returns
        -------
        df_all_jobs: A dataframe of the all of the data for the given dictionary
        job_list: list, a list of jobs from Signac that fit criteria dict for the methods in meth_name_val_list
        theta_true_data: dict, the true parameter values for the given case study
        
        """
        #Intialize dataframe and job list for all jobs in criteria_dict
        df_all_jobs = pd.DataFrame()
        job_list = []
        
        #Find all jobs of a certain cs and method type for the criteria in order of job id
        if criteria_dict == None:
            jobs = self.get_jobs_from_criteria(self.criteria_dict)
        else:
            jobs = self.get_jobs_from_criteria(criteria_dict)
        theta_true_data = None
        
        #Loop over each job
        for job in jobs:
            assert os.path.exists(job.fn("BO_Results.gz")), "File must exist!" 
            #Add job to job list and set data_file
            job_list += [job]
            data_file = job.fn("BO_Results.gz")
        
            # # #See if result data exists, if so add it to df
            tab_data_path = os.path.join(job.fn("analysis_data") , "tabulated_data.csv")
            tab_param_path = os.path.join(job.fn("analysis_data") , "true_param_data.json")
            found_data1, df_job = self.load_data(tab_data_path)
            found_data2, theta_true_data = self.load_data(tab_param_path)
            #If results don't exist or we are overwriting our csvs, create them
            if save_csv or not found_data1 or not found_data2:
                df_job, theta_true_data = self.get_study_data_signac(job, save_csv)
            elif found_data1:
                try:
                    df_job["Theta Opt Acq"] = df_job["Theta Opt Acq"].apply(self.str_to_array_df_col)
                    df_job["Theta Min Obj"] = df_job["Theta Min Obj"].apply(self.str_to_array_df_col)
                    df_job['Theta Obj GP Cum'] = df_job['Theta Obj GP Cum'].apply(self.str_to_array_df_col) 
                    df_job['Theta Obj Act Cum'] = df_job['Theta Obj Act Cum'].apply(self.str_to_array_df_col) 
                    df_job['Theta Acq Act Cum'] = df_job['Theta Acq Act Cum'].apply(self.str_to_array_df_col)
                except:
                    df_job.head()

                #Add cs name data to the dataframe
                df_job["CS Name Val"] = job.sp.cs_name_val
                df_job["CS Name"] = get_cs_class_from_val(job.sp.cs_name_val).name
                
            #Add job dataframe to dataframe of all jobs
            df_all_jobs = pd.concat([df_all_jobs, df_job], ignore_index=False)

        #Reset index on df_all_jobs after adding all rows 
        df_all_jobs = df_all_jobs.reset_index(drop=True)     
        
        # #Open Datafile to get theta_true if necessary
        if theta_true_data is not None:
            results = open_file_helper(data_file)
            theta_true = results[0].simulator_class.theta_true
            theta_true_names = results[0].simulator_class.theta_true_names
            theta_true_data = dict(zip(theta_true_names, theta_true))
            
        return df_all_jobs, job_list, theta_true_data
    
    def get_study_data_signac(self, job, save_csv = None):
        """
        Get best data from jobs and optionally save the csvs for the data
        
        Parameters
        ----------
        job: job, The job to get data from
        save_csv: bool, whether to save csvs. Default None
        
        Returns
        -------
        df_job: pd.DataFrame, The dataframe of the data for the given job
        theta_true_data: dict, the true parameter values for the given case study

        Notes:
        ------
        save_csv is set to the class default when not specified
        
        """
        assert isinstance(job, signac.job.Job), "job must be a signac.job.Job object"
        save_csv = self.save_csv if save_csv == None else save_csv
        #Initialize df for a single job
        df_job = pd.DataFrame()
        data_file = job.fn("BO_Results.gz")

        #Open the file and get the dataframe
        results = open_file_helper(data_file)

        #Get statepoint info
        with open(job.fn("signac_statepoint.json"), 'r') as json_file:
            # Load the JSON data
            sp_data = json.load(json_file)
        
        #Find number of workflow restarts in that job
        tot_runs = results[0].configuration["Number of Workflow Restarts"]
        num_x_exp = results[0].exp_data_class.get_num_x_vals()
        #get theta_true from 1st run since it never changes within a case study
        theta_true = results[0].simulator_class.theta_true
        theta_true_names = results[0].simulator_class.theta_true_names
        theta_true_data = dict(zip(theta_true_names, theta_true))

        #Loop over runs in each job
        for run in range(tot_runs):
            #Read data as pd.df
            df_run = results[run].results_df
            #Add the EP enum value as a column
            col_vals = job.sp.ep_enum_val
            df_run['EP Method Val'] = Ep_enum(int(col_vals)).name
            #Set index as the first run in the job's run number + the run we're at in the job
            df_run["index"] = int(job.sp.bo_run_num + run)
            df_run["Job ID"] = job.id

            #Set Run numbers as columns        
            df_run.rename(columns={'index': 'Run Number'}, inplace=True)   
           
            #Add run dataframe to job dataframe after
            df_job = pd.concat([df_job, df_run], ignore_index=False)

        #Reset index on job dataframe
        df_job = df_job.reset_index(drop=True)
        #Add cs name data to the dataframe
        df_job["CS Name Val"] = job.sp.cs_name_val
        df_job["CS Name"] = get_cs_class_from_val(job.sp.cs_name_val).name

        if save_csv:
            all_data_path = os.path.join(job.fn("analysis_data"), "tabulated_data.csv")
            theta_data_path = os.path.join(job.fn("analysis_data"), "true_param_data.json")
            self.save_data(df_job, all_data_path)
            self.save_data(theta_true_data, theta_data_path)

        return df_job, theta_true_data

    def get_best_data(self):
        """
        Gets the best (as described from self.mode) performing data for each method in the criteria dict
    
        Returns
        -------
        df_best: pd.DataFrame, The best data for each method
        job_list_best: list, a list of jobs from Signac corresponding to the ones in df_best
        """
        if self.mode == "act":
            obj_col = "Min Obj Act Cum"
        elif self.mode == "acq":
            obj_col = "Acq Obj Act Cum"
        elif self.mode == "gp":
            obj_col = "Min Obj GP Cum"
        #Get data from Criteria dict if you need it
        df, jobs, theta_true_data = self.get_df_all_jobs()
        data_best_path = os.path.join(self.study_results_dir, "best_results.csv")
        data_exists, df_best = self.load_data(data_best_path)
        if not data_exists or self.save_csv:
            #Start by sorting pd dataframe by lowest obj func value overall
            df_sorted = df.sort_values(by=[obj_col, 'BO Iter'], ascending=True)
            # df_sorted = df.sort_values(by=["Acq Obj Act Cum", 'BO Iter'], ascending=True)
            # df_sorted = df.sort_values(by=["Min Obj GP Cum", 'BO Iter'], ascending=True)
            #Then take only the 1st instance for each method
            df_best = df_sorted.drop_duplicates(subset='BO Method', keep='first').copy()
            #Calculate the L2 norm of the best runs
            df_best = self.__calc_l2_norm(df_best, np.array(list(theta_true_data.values())))
            #Sort df_best
            df_best = self.sort_by_meth(df_best)

        #Get list of best jobs
        job_list_best = self.__get_job_list(df_best)

        #Put in a csv file in a directory based on the job
        if self.save_csv:
            self.save_data(df_best, data_best_path)

        return df_best, job_list_best

    def get_best_all_runs(self, criteria_dict = None):
        """
        Gets the best (as described from self.mode) performing data for each method in the criteria dict
    
        Returns
        -------
        df_best: pd.DataFrame, The best data for each method
        job_list_best: list, a list of jobs from Signac corresponding to the ones in df_best
        """
        if self.mode == "act":
            obj_col = "Min Obj Act Cum"
        elif self.mode == "acq":
            obj_col = "Acq Obj Act Cum"
        elif self.mode == "gp":
            obj_col = "Min Obj GP Cum"
        #Get data from Criteria dict if you need it
        df, jobs, theta_true_data = self.get_df_all_jobs(criteria_dict)
        # print(df.head())
        data_best_path = os.path.join(self.study_results_dir, "best_run_results.csv")
        data_exists, df_best = self.load_data(data_best_path)
        if not data_exists or self.save_csv:
            #Start by sorting pd dataframe by lowest obj func value overall, then by BO Iter and Run Number
            df_sorted = df.sort_values(by=[obj_col, 'BO Iter', 'Run Number'], ascending=True)
            #Then take only the 1st instance for each method and run
            df_best = df_sorted.drop_duplicates(subset=['BO Method', 'Run Number'], keep='first').copy()
            #Calculate the L2 norm of the best runs
            df_best = self.__calc_l2_norm(df_best, np.array(list(theta_true_data.values())))
            #Sort df_best
            df_best = self.sort_by_meth(df_best)

        #Get list of best jobs
        job_list_best = self.__get_job_list(df_best)

        #Put in a csv file in a directory based on the job
        if self.save_csv:
            self.save_data(df_best, data_best_path)

        return df_best, job_list_best
    
    def get_median_data(self):
        """
        Gets the median performing data for each method in the criteria dict

        Returns
        -------
        df_median: pd.DataFrame, The median data for each method
        job_list_med: list, a list of jobs from Signac corresponding to the ones in df_median
        """
        if self.mode == "act":
            obj_col = "Min Obj Act"
        elif self.mode == "acq":
            obj_col = "Acq Obj Act"
        elif self.mode == "gp":
            obj_col = "Min Obj GP"
        #Get data from Criteria dict if you need it
        df, jobs, theta_true_data = self.get_df_all_jobs()
        data_path = os.path.join(self.study_results_dir, "median_results.csv")
        data_exists, df_median = self.load_data(data_path)
        if not data_exists or self.save_csv:
            #Initialize df for median values
            df_median = pd.DataFrame()
            #Loop over all methods
            for meth in df['BO Method'].unique():
                #Create a new dataframe w/ just the data for one method in it
                df_meth = df[df["BO Method"]==meth]
                #Add the row corresponding to the median value of SSE to the list
                if isinstance(df_meth[obj_col].iloc[0], np.ndarray):
                    median_sse = df_meth[obj_col].quantile(interpolation='nearest')[0]
                else:
                    median_sse = df_meth[obj_col].quantile(interpolation='nearest')
                #Ensure that only one values is used if there are multiple
                med_df = pd.DataFrame([df_meth[df_meth[obj_col] == median_sse].iloc[0]], columns=df_meth.columns) 
                #Add df to median
                df_median = pd.concat([df_median, med_df])
            #Calculate the L2 Norm for the median values
            df_median = self.__calc_l2_norm(df_median, np.array(list(theta_true_data.values())))
            #Sort df
            df_median = self.sort_by_meth(df_median)

        #Get list of best jobs
        job_list_med = self.__get_job_list(df_median)

        #Put in a csv file in a directory based on the job
        if self.save_csv:
            self.save_data(df_median, data_path)

        return df_median, job_list_med
    
    def get_mean_data(self):
        """
        Gets the mean performing data for each method in the criteria dict

        Returns
        -------
        df_mean: pd.DataFrame, The mean data for each method
        job_list_mean: list, a list of jobs from Signac corresponding to the ones in df_mean
        """
        if self.mode == "act":
            obj_col = "Min Obj Act"
        elif self.mode == "acq":
            obj_col = "Acq Obj Act"
        elif self.mode == "gp":
            obj_col = "Min Obj GP"
        #Get data from Criteria dict if you need it
        df, jobs, theta_true_data = self.get_df_all_jobs()
        data_path = os.path.join(self.study_results_dir, "mean_results.csv")
        data_exists, df_mean = self.load_data(data_path)
        if not data_exists or self.save_csv:
            #Initialize df for median values
            df_mean = pd.DataFrame()
            #Loop over all methods
            for meth in df['BO Method'].unique():
                #Get dataframe of data for just one method
                df_meth = df[df["BO Method"]==meth]  
                #Add find the true mean of the data
                if isinstance(df_meth[obj_col].iloc[0], np.ndarray):
                    df_true_mean = df_meth[obj_col].mean()[0]
                else:
                    df_true_mean = df_meth[obj_col].mean()
                #Find point closest to true mean
                df_closest_to_mean = df_meth.iloc[(df_meth[obj_col]-df_true_mean).abs().argsort()[:1]]
                #Add closest point to mean to df
                df_mean = pd.concat([df_mean, df_closest_to_mean])
            #Calculate the L2 Norm for the mean values
            df_mean = self.__calc_l2_norm(df_mean, np.array(list(theta_true_data.values())))
            #Sort df
            df_mean = self.sort_by_meth(df_mean)

        #Get list of best jobs
        job_list_mean = self.__get_job_list(df_mean)

        #Put in a csv file in a directory based on the job
        if self.save_csv:
            self.save_data(df_mean, data_path)

        return df_mean, job_list_mean
    
    
    def __get_job_list(self, df_data):
        """
        Helper function to pull best jobs from a dataframe

        Parameters
        ----------
        df_data: pd.DataFrame, The dataframe to pull the best jobs from

        Returns
        -------
        job_list: list, a list of jobs from Signac corresponding to the ones in df_data 
        """
        #Get list of best jobs
        job_list = []
        job_id_list = list(df_data["Job ID"])
        for job_id in job_id_list:
            job = self.project.open_job(id=job_id)
            if job:
                job_list.append(job)
        return job_list


    def sort_by_meth(self, df_data):
        """
        Sorts a dataframe by the method used

        Parameters
        ----------
        df_data: pd.DataFrame, The dataframe to sort

        Returns
        -------
        df_data: pd.DataFrame, The sorted dataframe
        """
        #Put rows in order of method
        # row_order = sorted([Method_name_enum[meth].value for meth in df_data['BO Method'].unique()])
        # order = [Method_name_enum(num).name for num in row_order]
        order = ["Conventional", "Log Conventional", "Independence", "Log Independence", "Sparse Grid", "Monte Carlo", "E[SSE]"]
        # Reindex the DataFrame with the specified row order
        df_data['BO Method'] = pd.Categorical(df_data['BO Method'], categories=order, ordered=True)
        # Sort the DataFrame based on the categorical order
        df_data = df_data.sort_values(by='BO Method')
        return df_data
    
    def __calc_l2_norm(self, df_data, theta_true):
        """
        Calculates the L2 norm of the theta values in a dataframe

        Parameters
        ----------
        df_data: pd.DataFrame, The dataframe to calculate the L2 norm for
        theta_true: np.array, The true theta values to compare to

        Returns
        -------
        df_data: pd.DataFrame, The dataframe with the L2 norm values added
        """
        #Calculate the difference between the true values and the GP best values in the dataframe for each parameter    
        theta_min_obj = np.array(list(df_data['Theta Min Obj'].to_numpy()[:]), dtype=np.float64)

        del_theta = theta_min_obj - theta_true
        theta_L2_norm = np.zeros(del_theta.shape[0])
        for i in range(del_theta.shape[0]):
            theta_L2_norm[i] = np.linalg.norm(del_theta[i,:], ord = 2)
            
        df_data["L2 Norm Theta"] = theta_L2_norm

        return df_data
    
    def load_data(self, path):
        """
        Loads data from a file based on the file extension

        Parameters
        ----------
        path: str, The file path of the data

        Returns
        -------
        found_data: bool, whether the data was found
        data: np.ndarray or pd.DataFrame or None, the data from the file or None
        """
        assert isinstance(path, str), "path_end must be str"
        #Split path into parts
        ext = os.path.splitext(path)[-1]
        #Extract directory name
        dirname = os.path.dirname(path)
        # #Make directory if it doesn't already exist
        # os.makedirs(dirname, exist_ok=True)
        #Based on extension, save in different ways
        #Check if csv already exists
        if os.path.exists(path):
            #If so, load the file
            if ext == ".csv":
                data = pd.read_csv(path, index_col=0)
            elif ext == ".npy":
                data = np.load(path, allow_pickle=True)
            elif ext == ".pkl" or ext == ".gz":
                data = open_file_helper(path)
            elif ext == ".json":
                with open(path, 'r') as file:
                    data = json.load(file)
            else:
                raise ValueError("NOT a csv, json, npy, pkl, or gz file")
            return True, data
        else:
            return False, None
    
    def save_data(self, data, save_path):
        """
        Saves data to a file based on the file extension

        Parameters
        ----------
        data: Object, the data to save
        save_path: str, The file path to save the data

        Returns
        -------
        None
        """
        #Split path into parts
        ext = os.path.splitext(save_path)[-1]
        #Extract directory name
        dirname = os.path.dirname(save_path)
        #Make directory if it doesn't already exist
        os.makedirs(dirname, exist_ok=True)
        #Based on extension, save in different ways
        if ext == '.csv':
            data.to_csv(save_path)
        elif ext == '.npy':
            np.save(save_path, data)
        elif ext == ".json":
            with open(save_path, 'w') as file:
                json.dump(data, file)
        elif ext == ".gz":
            with gzip.open(save_path, 'wb', compresslevel=1) as file:
                data = pickle.dump(data, file)
        elif ext == ".pkl":
            with open(save_path, 'wb') as file:
                data = pickle.dump(data, file)
        else:
            raise ValueError("NOT a csv, json, npy, pkl, or gz file")
        return

    def __z_choice_helper(self, z_choices, theta_true_data, data_type):
        """
        creates column and data names based on data type

        Parameters
        ----------
        z_choices: list, the choices of data to analyze
        theta_true_data: dict, the true parameter values for the given case study
        data_type: str, the type of data to analyze (parameter or objective data)

        Returns
        -------
        col_name: list, the column names for the data
        data_names: list, the names of the data

        """
        if self.mode == "act":
            obj_col_sse = "Min Obj Act"
            obj_col_sse_min = "Min Obj Act Cum"
            param_sse = 'Theta Min Obj'
            param_sse_min = 'Theta Obj Act Cum'
        elif self.mode == "acq":
            obj_col_sse = "Acq Obj Act"
            obj_col_sse_min = "Acq Obj Act Cum"
            param_sse = 'Theta Opt Acq'
            param_sse_min = 'Theta Opt Acq Cum'
        elif self.mode == "gp":
            obj_col_sse = "Min Obj GP"
            obj_col_sse_min = "Min Obj GP Cum"
            param_sse = 'Theta Min Obj'
            param_sse_min = 'Theta Obj GP Cum'

        if data_type == "objs":
            assert isinstance(z_choices, list), "z_choices must be list of string."
            assert all(isinstance(item, str) for item in z_choices), "z_choices elements must be string"
            assert any(item in z_choices for item in ["acq", "min_sse", "sse"]), "z_choices must contain at least 'min_sse', 'acq', or 'sse'"
            col_name = [] 
            data_names = []

            if self.mode == "gp":
                label_g = "\mathcal{M}_g(\mathbf{"
            else:
                label_g = "g(\mathbf{"

            for z_choice in z_choices:
                if "sse" == z_choice:
                    theta = "\\theta}^o" if self.mode != "acq" else "\hat{\\theta}}"
                    col_name += [obj_col_sse] #["Acq Obj Act"] #["Min Obj GP"], 
                    data_names += [label_g + theta + ")"]
                if "min_sse" == z_choice:
                    theta = "\\theta}^{\prime}"
                    col_name += [obj_col_sse_min] #["Acq Obj Act Cum"] #["Min Obj GP Cum"], 
                    data_names += [label_g + theta + ")"]        
                if "acq" == z_choice:
                    col_name += ["Opt Acq"]
                    data_names += ["\Xi(\mathbf{\hat{\\theta}})"]

        elif data_type == "params":
            assert isinstance(z_choices, str), "z_choices must be a string"
            assert any(item == z_choices for item in ["acq", "min_sse", "sse"]), "z_choices must be one of 'min_sse', 'acq', or 'sse'"
            data_names = list(theta_true_data.keys())
            if "min_sse" in z_choices:
                col_name = param_sse_min #['Theta Acq Act Cum'], ['Theta Obj GP Cum'], ['Theta Obj Act Cum']
            elif "sse" == z_choices:
                col_name = param_sse #['Theta Min Obj'], ['Theta Opt Acq']
            elif "acq" in z_choices:
                col_name = 'Theta Opt Acq'
            else:
                warnings.warn("z_choices must be 'acq', 'sse', or 'min_sse'.")
        return col_name, data_names

    def __preprocess_analyze(self, job, z_choice, data_type):
        """
        Preprocesses data for analysis based on data type

        Parameters
        ----------
        job: signac.job.Job, the job to analyze
        z_choice: list or str, the choices of data to analyze
        data_type: str, the type of data to analyze (parameter or objective data)

        Returns
        -------
        df_job: pd.DataFrame, the dataframe of the data for the given job
        data: np.ndarray, the data for plotting
        data_true: np.ndarray or None, the true values of the data
        sp_data: dict, the statepoint data for the job
        tot_runs: int, the total number of runs in the job
        """
        #Look for data if it already exists, if not create it
        #Check if we have theta data and create it if not
        tab_data_path = os.path.join(job.fn("analysis_data") , "tabulated_data.csv")
        true_param_data_path = os.path.join(job.fn("analysis_data") , "true_param_data.json")
        # print([data_file_path for data_file_path in [data_file, data_name_file, data_true_file]])
        found_data1, df_job = self.load_data(tab_data_path)
        found_data2, theta_true_data = self.load_data(true_param_data_path)

        if not found_data1 or not found_data2:
            df_job, theta_true_data = self.get_study_data_signac(job, save_csv = False)
        elif found_data1:
            df_job["Theta Opt Acq"] = df_job["Theta Opt Acq"].apply(self.str_to_array_df_col)
            df_job["Theta Min Obj"] = df_job["Theta Min Obj"].apply(self.str_to_array_df_col)
            df_job['Theta Obj GP Cum'] = df_job['Theta Obj GP Cum'].apply(self.str_to_array_df_col) 
            df_job['Theta Obj Act Cum'] = df_job['Theta Obj Act Cum'].apply(self.str_to_array_df_col) 
            df_job['Theta Acq Act Cum'] = df_job['Theta Acq Act Cum'].apply(self.str_to_array_df_col)

        #Get statepoint info
        with open(job.fn("signac_statepoint.json"), 'r') as json_file:
            # Load the JSON data
            sp_data = json.load(json_file)
            tot_runs = sp_data["bo_runs_in_job"]
            max_iters = sp_data["bo_iter_tot"]

        if data_type == "objs":
            #Get SSE data from least squares. This is the "True" value
            ls_analyzer = LS_Analysis(self.criteria_dict, self.project, self.save_csv)
            ls_results = ls_analyzer.least_squares_analysis()
            #Make a df that is only the iters of the best run
            df_sorted = ls_results.sort_values(by=['Min Obj Cum.', 'Iter'], ascending=True)
            best_run = df_sorted["Run"].iloc[0]
            data_true = ls_results[ls_results['Run'] == best_run].copy()
            # data_true = min(ls_results["Min Obj Cum."])
            data = np.zeros((tot_runs, max_iters, len(z_choice)))
        elif data_type == "params":
            data_true = theta_true_data
            data = np.zeros((tot_runs, max_iters, len(list(theta_true_data.keys()))))

        #Sort df_job by run and iter
        df_job = df_job.sort_values(by=['Run Number', 'BO Iter'], ascending=True)

        return df_job, data, data_true, sp_data, tot_runs
    
    def analyze_obj_vals(self, job, z_choices):
        """
        Gets the data into an array for for plotting any comination of sse, log_sse, and ei
        
        Parameters
        ----------
        job: signac.job.Job, the job to analyze
        z_choices: list or str, the choices of data to analyze
        
        Returns
        -------
        data: np.ndarray, The data for plotting
        data_names: list, the names of the data
        data_true: dict or None, the true values of the data
        sp_data: dict, the statepoint data for the job
        """
        assert isinstance(job, signac.job.Job), "job must be a signac.job.Job object"
        assert isinstance(z_choices, (Iterable, str)), "z_choices must be Iterable or str"
        if isinstance(z_choices, str):
            z_choices = [z_choices]
        assert all(isinstance(item, str) for item in z_choices), "z_choices elements must be str"
        for i in range(len(z_choices)):
            assert z_choices[i] in ['min_sse','sse','acq'],"z_choices items must be 'min_sse', 'sse', or 'acq'"

        df_job, data, data_true_val, sp_data, tot_runs = self.__preprocess_analyze(job, z_choices, "objs")
        data_true = {}
        col_name, data_names = self.__z_choice_helper(z_choices, data_true, "objs")

        unique_run_nums = pd.unique(df_job["Run Number"])
        #Loop over each choice
        for z in range(len(z_choices)):
            #Loop over runs 
            for i, run in enumerate(unique_run_nums):
                #Make a df of only the data which meets that run criteria
                df_run = df_job[df_job["Run Number"] == run]  
                z_data = df_run[col_name[z]]
                #If sse in log choices, the "true data" is sse data from least squares
                if "sse" in z_choices[z]:
                    data_true[z_choices[z]] = data_true_val
                    #If the z_choice is sse and the method has a log objective function value, un logscale data
                    if sp_data["meth_name_val"] in [2,4]:
                        z_data = np.exp(z_data.values.astype(float))
                else:
                    data_true[z_choices[z]] = None
                #Set data to be where it needs to go in the above data matrix
                data[i,:len(z_data),z] = z_data

        return data, data_names, data_true, sp_data
        
    def analyze_thetas(self, job, z_choice):
        """
        Gets the data into an array for for plotting parameter values

        Parameters
        ----------
        job: signac.job.Job, the job to analyze
        z_choice: str, the choice of data to analyze

        Returns
        -------
        data: np.ndarray, The data for plotting
        data_names: list, the names of the data
        data_true: dict or None, the true values of the data
        sp_data: dict, the statepoint data for the job
        """
        assert isinstance(job, signac.job.Job), "job must be a signac.job.Job object"
        assert isinstance(z_choice, (str)), "z_choice must be a str"
        assert z_choice in ['min_sse','sse','acq'],"z_choice must be 'min_sse', 'sse', or 'acq'"

        df_job, data, data_true, sp_data, tot_runs = self.__preprocess_analyze(job, z_choice, "params")
        col_name, data_names = self.__z_choice_helper(z_choice, data_true, "params")
        #Loop over runs
        unique_run_nums = pd.unique(df_job["Run Number"])
        for i, run in enumerate(unique_run_nums):
            #Make a df of only the data which meets that run criteria
            df_run = df_job[df_job["Run Number"]==run]  
            df_run_arry = np.array([arr.tolist() for arr in df_run[col_name].to_numpy()])
            for param in range(data.shape[-1]):
                z_data = df_run_arry[:,param]
                #Set data to be where it needs to go in the above data matrix
                data[i,:len(z_data),param] = z_data
                
        data_names = [element.replace('theta', '\\theta') for element in data_names]

        return data, data_names, data_true, sp_data
    
    def analyze_hypers(self, job):
        """
        Gets the data into an array for for plotting hyperparameters

        Parameters
        ----------
        job: signac.job.Job, the job to analyze

        Returns
        -------
        data: np.ndarray, The data for plotting
        data_names: list, the names of the data
        data_true: None, the true values of the data (None for hyperaprameters)
        sp_data: dict, the statepoint data for the job
        """
        data_true = None
        #Check for prexisting data
        hp_data_path = os.path.join(job.fn("analysis_data") , "hyperparam_data.npy")
        hp_name_path = os.path.join(job.fn("analysis_data") , "hp_name_data.json")
        found_data1, data = self.load_data(hp_data_path)
        found_data2, data_names = self.load_data(hp_name_path)

        #Get statepoint info
        with open(job.fn("signac_statepoint.json"), 'r') as json_file:
            # Load the JSON data
            sp_data = json.load(json_file)
            tot_runs = sp_data["bo_runs_in_job"]
            max_iters = sp_data["bo_iter_tot"]

        if self.save_csv or (not found_data1 and not found_data2):
            loaded_results = open_file_helper(job.fn("BO_Results_GPs.gz"))
            dim_hps = len(loaded_results[0].list_gp_emulator_class[0].trained_hyperparams[0]) + 2
            data = np.zeros((tot_runs, max_iters, dim_hps))
            data_names = [f"\\ell_{i}" for i in range(1, dim_hps+1)]
            data_names[-2] = "\sigma"
            data_names[-1] = "\\tau"
            
            for j in range(tot_runs):
                run = loaded_results[j]
                for i in range(len(run.list_gp_emulator_class)):
                # Extract the array and convert other elements to float
                    array_part = run.list_gp_emulator_class[i].trained_hyperparams[0]
                    rest_part = np.array(run.list_gp_emulator_class[i].trained_hyperparams[1:], dtype=float)
                    hp = np.concatenate([array_part, rest_part])
                    # Create the resulting array of shape (1, 10)
                    data[j,i,:] = hp

            if self.save_csv:
                self.save_data(data, hp_data_path)
                self.save_data(data_names, hp_name_path)

        return data, data_names, data_true, sp_data
    
    def __rebuild_cs(self, sp_data):
        """
        builds instance of CaseStudyParameters from saved file data

        Parameters
        ----------
        sp_data: dict, the statepoint data for the job

        Returns
        -------
        cs_params: CaseStudyParameters, the case study parameters
        method: GPBO_Methods, the method used
        gen_meth_theta: Gen_meth_enum, the method used to generate theta values
        ep_enum: Ep_enum, the method used to generate exploration bias values
        """
        method = GPBO_Methods(Method_name_enum(sp_data["meth_name_val"]))
        cs_name = CS_name_enum(sp_data["cs_name_val"]) if "cs_name_val" in sp_data else "New_CS"
        ep0 = sp_data["ep0"]
        sep_fact = sp_data["sep_fact"]
        normalize = sp_data["normalize"]
        kernel = Kernel_enum(sp_data["kernel_enum_val"])
        lenscl = sp_data["lenscl"]
        outputscl = sp_data["outputscl"]
        retrain_GP = sp_data["retrain_GP"]
        reoptimize_obj = sp_data["reoptimize_obj"]
        gen_heat_map_data = sp_data["gen_heat_map_data"]
        bo_iter_tot = sp_data["bo_iter_tot"]
        bo_run_tot = sp_data["bo_run_tot"]
        save_data = False
        DateTime = None
        seed = sp_data["seed"]
        obj_tol = sp_data["obj_tol"]
        ei_tol = sp_data["ei_tol"]
        gen_meth_theta = Gen_meth_enum(sp_data["gen_meth_theta"])
        ep_enum = Ep_enum(sp_data["ep_enum_val"])
        
        cs_params = CaseStudyParameters(cs_name, ep0, sep_fact, normalize, kernel, lenscl, outputscl, 
                                        retrain_GP, reoptimize_obj, gen_heat_map_data, bo_iter_tot, 
                                        bo_run_tot, save_data, DateTime, seed, obj_tol, ei_tol)
        
        return cs_params, method, gen_meth_theta, ep_enum        

    def analyze_parity_plot_data(self, job, run_num, bo_iter):
        """
        Generates parity plot for testing data

        Parameters
        ----------
        job: signac.job.Job, the job to analyze
        run_num: int, the run number to analyze
        bo_iter: int, the bo iteration to analyze

        Returns
        -------
        test_data: Data, the evaluated testing data for the given run and iteration
        """
        assert isinstance(job, signac.job.Job), "job must be a signac.job.Job object"
        assert isinstance(run_num, (np.int64, int)), "run_num must be an int"
        assert isinstance(bo_iter, (np.int64,int)), "bo_iter must be an int"
        #Get Best Data
        #Check if data exists, if so, load it
        #Assert that heat map data does not aleady exist
        dir_name = os.path.join(job.fn(""), "analysis_data", "gp_evaluations", 
                                "run_" + str(run_num), "iter_" + str(bo_iter))
        data_name = os.path.join(dir_name, "test_data.pkl")
        found_data1, test_data = self.load_data(data_name)

        #Get statepoint_info
        #Get statepoint info
        with open(job.fn("signac_statepoint.json"), 'r') as json_file:
            # Load the JSON data
            sp_data = json.load(json_file)
        bo_runs_in_job = sp_data["bo_runs_in_job"]
        bo_run_num_int = sp_data["bo_run_num"]
        run_idx = run_num - bo_run_num_int
        meth_name_val = sp_data["meth_name_val"]
        meth_name = Method_name_enum(meth_name_val)
        method = GPBO_Methods(meth_name)
        
        #Otherwise Generate it
        if self.save_csv or not found_data1:
            #Open file
            results = open_file_helper(job.fn("BO_Results.gz"))
            results_GP = open_file_helper(job.fn("BO_Results_GPs.gz"))
            assert len(results) > run_idx, "run_num is out of bounds"
            assert len(results_GP[run_idx].list_gp_emulator_class) > bo_iter-1, "bo_iter is out of bounds"
            gp_object = copy.copy(results_GP[run_idx].list_gp_emulator_class[bo_iter-1])
            simulator = copy.copy(results[run_idx].simulator_class)
            exp_data = copy.copy(results[0].exp_data_class) #Experimental data won't change

            #Get testing data if it doesn't exist
            if gp_object.test_data is None or len(gp_object.test_data.theta_vals) == 0:
                #Generate testing data if it doesn't exist
                #Get 10 num_theta points for testing
                num_x = exp_data.get_num_x_vals()
                dim_x = exp_data.get_dim_x_vals()
                use_x = int(num_x**(1/dim_x))
                #Make Data (For multi vs 1D X data)
                # For conventional methods, must use same x for testing data as exp values to get same results
                test_data_sim = simulator.gen_sim_data(10, use_x, Gen_meth_enum(1), Gen_meth_enum(2), 1.0, simulator.seed, False)
                if method.emulator == False:
                    test_data_sim = simulator.sim_data_to_sse_sim_data(method, test_data_sim, exp_data, 1.0, False)

                gp_object.test_data = test_data_sim
                gp_object.feature_test_data = gp_object.featurize_data(gp_object.test_data)
                gp_object.test_data.gp_mean, gp_object.test_data.gp_var = gp_object.eval_gp_mean_var_test()  

            test_data = gp_object.test_data
            
            if self.save_csv:
                self.save_data(test_data, data_name)
        
        return test_data
    
    def analyze_heat_maps(self, job, run_num, bo_iter, pair_id, get_ei = False):
        """
        Generates/analyzes heat map data for the given run and iteration

        Parameters
        ----------
        job: signac.job.Job, the job to analyze
        run_num: int, the run number to analyze
        bo_iter: int, the bo iteration to analyze
        pair_id: int or str, the pair of parameters to analyze
        get_ei: bool, whether to calculate the acquisition function. Default is False

        Returns
        -------
        all_data: np.ndarray, the data for plotting
        test_mesh: np.ndarray, the meshgrid for the testing data
        param_info_dict: dict, the parameter information for the given pair
        sp_data: dict, the statepoint data for the job
        """
        assert isinstance(job, signac.job.Job), "job must be a signac.job.Job object"
        assert isinstance(run_num, (np.int64, int)), "run_num must be an int"
        assert isinstance(bo_iter, (np.int64,int)), "bo_iter must be an int"
        assert isinstance(pair_id, (np.int64, int, str)), "pair_id must be an int or str"
        assert isinstance(get_ei, bool), "get_ei must be a bool"

        #Assert that heat map data does not aleady exist
        dir_name = os.path.join(job.fn(""), "analysis_data", "gp_evaluations", 
                                "run_" + str(run_num), "iter_" + str(bo_iter),  "pair_" + str(pair_id))
        hm_path_name = os.path.join(dir_name, "hm_data.gz")
        hm_sse_path_name = os.path.join(dir_name, "hm_sse_data.gz")
        param_info_path = os.path.join(dir_name, "notable_param_info.pkl")
        try:
            found_data1, heat_map_data = self.load_data(hm_path_name)
        except:
            print(job.id, hm_path_name)
            found_data1, heat_map_data = False, None
        found_data2, heat_map_sse_data = self.load_data(hm_sse_path_name)
        found_data3, param_info_dict = self.load_data(param_info_path)

        #Get statepoint info
        with open(job.fn("signac_statepoint.json"), 'r') as json_file:
            # Load the JSON data
            sp_data = json.load(json_file)

        #Set run number to the index of the run number in the job by subtracting
        #The number of the first run in the job
        run_num -= sp_data["bo_run_num"]
        bo_iter -= 1
        cs_params, method, gen_meth_theta, ep_method = self.__rebuild_cs(sp_data)

        data_not_found = not found_data1 or not found_data2 or not found_data3
        #Initialize data_needs_ei as true
        data_needs_ei = True
        #If we don't need acq data, set data_needs_ei to False
        if get_ei == False:
            data_needs_ei = False
        #If we have all the data and we need to calculate acq, check if we have acq data
        elif not data_not_found and not self.save_csv and get_ei:
            #If we have all the data, we won't need to calculate ei
            if heat_map_sse_data.acq is not None:
                data_needs_ei = False

        #Generate driver class/ emulator class if data doesn't exist or we need to calculate acq
        if self.save_csv or data_not_found or data_needs_ei:
            loaded_results = open_file_helper(job.fn("BO_Results.gz"))
            loaded_results_GPs = open_file_helper(job.fn("BO_Results_GPs.gz"))
            assert len(loaded_results_GPs) > run_num, "run_num is out of bounds"
            assert len(loaded_results_GPs[run_num].list_gp_emulator_class) > bo_iter, "bo_iter is out of bounds"

            #Create Heat Map Data for a run and iter
            #Regeneate class objects 
            gp_emulator = loaded_results_GPs[run_num].list_gp_emulator_class[bo_iter]
            exp_data = loaded_results[run_num].exp_data_class
            simulator = loaded_results[run_num].simulator_class
            ep_at_iter = loaded_results[run_num].results_df["Exploration Bias"].iloc[bo_iter]
            ep_bias = Exploration_Bias(None, ep_at_iter, ep_method, None, None, None, None, None, None, None)
            driver = GPBO_Driver(cs_params, method, simulator, exp_data, gp_emulator.gp_sim_data, 
                                     gp_emulator.gp_sim_data, gp_emulator.gp_val_data, gp_emulator.gp_val_data, 
                                     gp_emulator, ep_bias, gen_meth_theta)
            
            #Get best error metrics
            be_data, best_error_metrics = driver._GPBO_Driver__get_best_error()

        #Create heat map data if it doesn't exists
        if self.save_csv or data_not_found:
            if self.mode == "act":
                param_sse_min = 'Theta Obj Act Cum'
            elif self.mode == "acq":
                param_sse_min = 'Theta Acq Act Cum'
            elif self.mode == "gp":
                param_sse_min = 'Theta Obj GP Cum'

            #Get important theta values
            theta_true = loaded_results[run_num].simulator_class.theta_true
            theta_opt =  loaded_results[run_num].results_df[param_sse_min].iloc[bo_iter]
            theta_next = loaded_results[run_num].results_df["Theta Opt Acq"].iloc[bo_iter]
            train_theta = loaded_results_GPs[run_num].list_gp_emulator_class[bo_iter].train_data.theta_vals
            
            #Get specific heat map data or generate it
            # if loaded_results[0].heat_map_data_dict is not None:
            #     heat_map_data_dict = loaded_results[0].heat_map_data_dict
            # else:
            num_x = exp_data.get_num_x_vals()
            n_points_set = len(driver.gp_emulator.gp_sim_data.get_unique_theta())
            if num_x*n_points_set**2 >= 5000:
                n_points_set = int(np.sqrt(5000/num_x))
            loaded_results_GPs[0].heat_map_data_dict = driver.create_heat_map_param_data(n_points_set)
            heat_map_data_dict = loaded_results_GPs[0].heat_map_data_dict

            #Get pair ID
            if isinstance(pair_id, str):
                assert pair_id in loaded_results_GPs[0].heat_map_data_dict.keys(), "pair_id is an invalid string"
                param_names = pair_id
            elif isinstance(pair_id, int):
                assert pair_id < len(loaded_results_GPs[0].heat_map_data_dict.keys()), "pair_id is out of bounds"
                param_names = list(loaded_results_GPs[0].heat_map_data_dict.keys())[pair_id]
            else:
                raise Warning("Invalid pair_id!")

            #Initialize heat map data class
            heat_map_data_org = heat_map_data_dict[param_names] 

            #Calculate GP mean and var for heat map data
            featurized_hm_data = gp_emulator.featurize_data(heat_map_data_org)
            try:
                hm_org_mean, hm_org_var = gp_emulator.eval_gp_mean_var_misc(heat_map_data_org, featurized_hm_data)
            except:
                print(n_points_set)

            #Get index of param set and best error
            idcs_to_plot = [loaded_results[run_num].simulator_class.theta_true_names.index(name) for 
                            name in param_names]  

            #Set param info
            param_info_dict = {"true":theta_true, "min_sse":theta_opt, "opt_acq":theta_next, "train":train_theta,
                                "names":param_names, "idcs":idcs_to_plot} 
                
            #If the emulator is a conventional method, create heat map data in emulator form to calculate y_vals
            if not method.emulator:
                #Make surrogate heat map data for full theta and x grid to calculate y_vals
                n_points = int(np.sqrt(heat_map_data_org.get_num_theta()))
                repeat_x = n_points**2 #Square because only 2 values at a time change
                x_vals = np.vstack([exp_data.x_vals]*repeat_x) #Repeat x_vals n_points**2 number of times
                repeat_theta = exp_data.get_num_x_vals() #Repeat theta len(x) number of times
                theta_vals =  np.repeat(heat_map_data_org.theta_vals, repeat_theta , axis =0) #Create theta data repeated
                heat_map_data = Data(theta_vals, x_vals, None, None, None, None, None, None, 
                                     simulator.bounds_theta_reg, simulator.bounds_x, cs_params.sep_fact, 
                                     cs_params.seed) 
            else:
                heat_map_data = heat_map_data_org
                
            #Generate heat map data and sse heat map data sim y values
            heat_map_data.y_vals = simulator.gen_y_data(heat_map_data, 0 , 0)

            #Create sse data from regular y data
            heat_map_sse_data = simulator.sim_data_to_sse_sim_data(method, heat_map_data, exp_data, 
                                                                    cs_params.sep_fact, gen_val_data = False)
            #Set the mean and variance to the correct heat map data object
            if not method.emulator:
                heat_map_sse_data.gp_mean = hm_org_mean
                heat_map_sse_data.gp_var = hm_org_var
            else:
                heat_map_data.gp_mean = hm_org_mean
                heat_map_data.gp_var = hm_org_var

            #Calculate SSE and SSE var
            if method.emulator == False:
                heat_map_sse_data.sse, heat_map_sse_data.sse_var = gp_emulator.eval_gp_sse_var_misc(heat_map_sse_data)            
            else:
                heat_map_sse_data.sse, heat_map_sse_data.sse_var = gp_emulator.eval_gp_sse_var_misc(heat_map_data, 
                                                                                                    method, exp_data)

        #Get EI if needed. This operation can be expensive which is why it's optional
        if data_needs_ei:
            if method.method_name.value == 7:
                heat_map_sse_data.acq = heat_map_sse_data.sse + np.sum(heat_map_sse_data.sse_var)
            elif method.emulator == False:
                heat_map_sse_data.acq = gp_emulator.eval_ei_misc(heat_map_sse_data, exp_data, ep_bias, 
                                                                best_error_metrics)[0]
            #In older data, sparse grid depth is not a set parameter. Therefore, we it's either 10, or a set value
            else:
                try:
                    sg_depth = loaded_results[run_num].configuration["Sparse Grid Depth"]
                    heat_map_sse_data.acq = gp_emulator.eval_ei_misc(heat_map_data, exp_data, ep_bias, 
                                                                    best_error_metrics, method, sg_depth)[0]
                except:
                    heat_map_sse_data.acq = gp_emulator.eval_ei_misc(heat_map_data, exp_data,ep_bias,
                                                                    best_error_metrics, method,sg_depth =10)[0]  

        #Save data if necessary
        if self.save_csv:
            self.save_data(heat_map_data, hm_path_name)
            self.save_data(heat_map_sse_data, hm_sse_path_name)
            self.save_data(param_info_dict, param_info_path)

        #Find the theta_vals in the given Data class to be only the 2D (varying) parts you want to plot
        theta_mesh_vals = heat_map_sse_data.theta_vals[:,param_info_dict["idcs"]]
        #Back out the number of theta points from the hm_sse_data
        theta_pts = int(np.sqrt(len(theta_mesh_vals)))
        #Create test mesh for that specific pair and set it as the new sse data theta vals.
        test_mesh = theta_mesh_vals.reshape(theta_pts,theta_pts,-1).T
        
        #Define sse_sim, sse_gp_mean, and sse_gp_var, and ei based on whether to report log scaled data
        sse_sim = heat_map_sse_data.y_vals
        sse_var = heat_map_sse_data.sse_var
        sse_mean = heat_map_sse_data.sse
        
        #Reshape data to correct shape and add to list to return
        reshape_list = [sse_sim, sse_mean, sse_var]     
        all_data = [var.reshape(theta_pts,theta_pts).T for var in reshape_list]
        if get_ei: #and heat_map_sse_data.acq is not None
            try:
                acq_new = copy.deepcopy(heat_map_sse_data.acq)
                all_data += [acq_new.reshape(theta_pts,theta_pts).T]
            except:
                print(heat_map_sse_data.sse, heat_map_sse_data.sse_var, get_ei, data_needs_ei, heat_map_sse_data.acq)
                all_data += [heat_map_sse_data.acq.reshape(theta_pts,theta_pts).T]
        else:
            all_data += [None]
        
        return all_data, test_mesh, param_info_dict, sp_data

class All_CS_Analysis(General_Analysis):
    """
    Class for analyzing results from all case studies
    """
    def __init__(self, cs_list, meth_val_list, project, mode, save_csv):
        """
        Parameters
        ----------
        cs_list: list, the list of case studies to analyze
        meth_val_list: list, the list of methods to analyze
        project: signac.project.Project, The signac project to analyze
        mode: str, the mode to analyze the data in (act, acq, or gp)
        save_csv: bool, whether to save csvs.
        """
        if len(cs_list) == 1:
            cs_name_val = cs_list[0]
        else: 
            cs_name_val = {"$in": cs_list}

        if len(meth_val_list) == 1:
            meth_name_val_list = meth_val_list[0]
        else:
            meth_name_val_list = {"$in": meth_val_list}

        criteria_dict = {"cs_name_val" : cs_name_val,
                        "meth_name_val": meth_name_val_list}
        
        super().__init__(criteria_dict, project, mode, save_csv)

        self.cs_list = cs_list
        self.meth_val_list = meth_val_list

    def __sort_by_cs_meth(self, df_data):
        """
        Sorts a dataframe by the method used

        Parameters
        ----------
        df_data: pd.DataFrame, The dataframe to sort

        Returns
        -------
        df_data: pd.DataFrame, The sorted dataframe
        """
        #Put rows in order of method
        # row_order = sorted([Method_name_enum[meth].value for meth in df_data['BO Method'].unique()])
        # order = [Method_name_enum(num).name for num in row_order]
        order = ["Conventional", "Log Conventional", "Independence", "Log Independence", "Sparse Grid", "Monte Carlo", "E[SSE]"]
        # Reindex the DataFrame with the specified row order
        df_data['BO Method'] = pd.Categorical(df_data['BO Method'], categories=order, ordered=True)
        # Sort the DataFrame based on the categorical order
        df_data = df_data.sort_values(by=['CS Name Val','BO Method'])
        return df_data

    def get_all_data(self):
        """
        Gets all data for all case studies
        
        Returns
        -------
        df_all_jobs: pd.DataFrame, The dataframe of all of the data for the given dictionary
        job_list: list, a list of jobs from Signac that fit criteria dict for the methods in meth_name_val_list
        theta_true_data: dict, the true parameter values for the given case study
        """
        #Sort by method and CS Name
        df_all_jobs, job_list, __ = self.get_df_all_jobs()
        df_all_jobs = self.__sort_by_cs_meth(df_all_jobs)
        return df_all_jobs
    
    def get_acq_last10_avg(self):
        """
        Get the average acquisition function value for the last 10 iterations of each run
        """
        df_all_jobs = self.get_all_data()
        #Get the last 10 of iterations of each run
        df_last_10 = df_all_jobs[df_all_jobs['BO Iter'] > (df_all_jobs['Max Evals'] - 10)]
        #Group by CS Name and BO Method and get the mean of the acquisition function value
        grouped_stats = df_last_10.groupby(["CS Name", "BO Method"]).agg({
        'Opt Acq': ['mean', 'std']}).reset_index()
        # Flatten the MultiIndex columns
        grouped_stats.columns = ['CS Name', 'BO Method', 'Avg Opt Acq', 'Std Opt Acq']
        df_acq_10_avg = grouped_stats[['CS Name', 'BO Method', 'Avg Opt Acq', 'Std Opt Acq']]
        return df_acq_10_avg
    
    def __log_scale_mean(self, series):
        log_values = np.log(series)
        mean_log = np.mean(log_values)
        mean = np.exp(mean_log)
        return mean_log

    # Custom function to calculate log-scale std
    def __log_scale_std(self, series):
        log_values = np.log(series)
        std_log = np.std(log_values)
        # std = np.exp(std_log)
        return std_log
    
    def __get_iqr(self, series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return iqr

    
    def get_averages(self):
        """
        Get average computational time, max function evaltuations, and sse values for all case studies
        """

        df_all_jobs = self.get_all_data()

        if self.mode == "act":
            obj_col_sse_min = "Min Obj Act Cum"
        elif self.mode == "acq":
            obj_col_sse_min = "Acq Obj Act Cum"
        elif self.mode == "gp":
            obj_col_sse_min = "Min Obj GP Cum"

        #Scale the objective function values for log conv and log indep
        condition = df_all_jobs['BO Method'].isin(["Log Conventional", "Log Independence"])
        # Multiply values in column B by 3 where the condition is true
        df_all_jobs.loc[condition, obj_col_sse_min] = np.exp(df_all_jobs.loc[condition, obj_col_sse_min])

        grouped_stats = df_all_jobs.groupby(["CS Name", "BO Method"]).agg({
        obj_col_sse_min: ['median', self.__get_iqr],
        'Total Run Time': ['mean', 'std'],
        'Max Evals': ['mean', 'std']}
        ).reset_index()
        # for (cs_name, meth_name), group in df_all_jobs.groupby(["CS Name", "BO Method"]):
        #     if cs_name == "Complex Linear":
        #         print(meth_name)
        #         print(group[obj_col_sse_min])

        # Flatten the MultiIndex columns
        grouped_stats.columns = ['CS Name', 'BO Method', 'Median Loss', 'IQR Loss', 'Avg Time', 'Std Time', 'Avg Evals', 'Std Evals']

        # Create a new DataFrame with results
        df_averages = grouped_stats[['CS Name', 'BO Method', 'Median Loss', 'IQR Loss',  'Avg Time', 'Std Time', 'Avg Evals', 'Std Evals']]
        return df_averages
    
    def get_averages_best(self):
        """
        Get average computational time, max function evaltuations, and sse values for all case studies
        """
        if self.mode == "act":
            obj_col_sse_min = "Min Obj Act Cum"
        elif self.mode == "acq":
            obj_col_sse_min = "Acq Obj Act Cum"
        elif self.mode == "gp":
            obj_col_sse_min = "Min Obj GP Cum"
        
        #Loop over each case study
        for i, cs_name in enumerate(self.cs_list):
            for j, meth_val in enumerate(self.meth_val_list):
                #Create a criteria dictionary for the case study
                criteria_dict = {"cs_name_val" : cs_name,
                                "meth_name_val": meth_val}
                #Evaluate the best run for the case study for each method
                try:
                    df_best_runs, job_list_best_runs = self.get_best_all_runs(criteria_dict)
                    #On iter 1, create the DataFrame df_all_best
                    if i==0 and j == 0 and len(df_best_runs) > 0:
                        df_all_best = df_best_runs
                    #Otherwise, concatenate the DataFrame to df_all_best
                    else:
                        df_all_best = pd.concat([df_all_best, df_best_runs], axis = 0)
                except:
                    pass

        #Scale the objective function values for log conv and log indep
        condition = df_all_best['BO Method'].isin(["Log Conventional", "Log Independence"])
        # Multiply values in column B by 3 where the condition is true
        df_all_best.loc[condition, obj_col_sse_min] = np.exp(df_all_best.loc[condition, obj_col_sse_min])

        #Group the data by CS Name and BO Method, and get the mean and std for each group over all runs
        grouped_stats = df_all_best.groupby(["CS Name", "BO Method"]).agg({
        obj_col_sse_min: ['median', self.__get_iqr],
        'BO Iter': ['mean', 'std']}).reset_index()

        # Flatten the MultiIndex columns
        grouped_stats.columns = ['CS Name', 'BO Method', 'Median Loss', 'IQR Loss', 'Avg Evals', 'Std Evals']

        # Create a new DataFrame with results
        df_acq_opt = self.get_acq_last10_avg()
        df_avg_best = grouped_stats[['CS Name', 'BO Method', 'Median Loss', 'IQR Loss', 'Avg Evals', 'Std Evals']]
        df_avg_best_w_acq = pd.merge(df_acq_opt, df_avg_best, on=['CS Name', 'BO Method'])
        return df_avg_best_w_acq
    
class LS_Analysis(General_Analysis):
    """
    The class for Least Squares regression analysis. Child class of General_Analysis

    Methods:
    --------
    __init__(self, criteria_dict, project, save_csv, exp_data=None, simulator=None)
        Initializes the class
    __ls_scipy_func(self, theta_guess, exp_data, simulator)
        Function to define regression function for least-squares fitting
    __get_simulator_exp_data(self)
        Gets the simulator and experimental data from the job
    least_squares_analysis(self, tot_runs = None)
        Performs least squares regression on the problem equal to what was done with BO
    categ_min(self, tot_runs = None)
        Categorizes the number of unique minima through multiple restarts of nonlinear least squares
    """
    #Inherit objects from General_Analysis
    def __init__(self, criteria_dict, project, save_csv, exp_data=None, simulator=None):
        assert isinstance(exp_data, Data) or exp_data == None, "exp_data must be type Data or None"
        assert isinstance(simulator, Simulator) or simulator == None, "simulator must be type Simulator or None"
        super().__init__(criteria_dict, project, "act", save_csv)
        self.iter_param_data = []
        self.iter_sse_data = []
        self.iter_l2_norm = []
        self.iter_count = 0
        self.seed = 1
        #Placeholder that will be overwritten if None
        self.simulator = simulator
        self.exp_data = exp_data
        self.num_x = 10 #Default number of x to generate

    # Create a function to optimize, in this case, least squares fitting
    def __ls_scipy_func(self, theta_guess, exp_data, simulator):
        """
        Function to define regression function for least-squares fitting
        Parameters
        ----------
        theta_guess: np.ndarray, the theta values to evaluate
        exp_data: Data, the experimental data to evaluate
        simulator: Simulator, the simulator object to evaluate

        Returns
        -------
        error: np.ndarray, the error between the experimental data and the simulated data
        """
        #Repeat the theta best array once for each x value
        #Need to repeat theta_best such that it can be evaluated at every x value in exp_data using simulator.gen_y_data
        t_guess_repeat = np.repeat(theta_guess.reshape(1,-1), exp_data.get_num_x_vals() , axis =0)
        #Add instance of Data class to theta_best
        theta_guess_data = Data(t_guess_repeat, exp_data.x_vals, None, None, None, None, None, None, 
                              simulator.bounds_theta_reg,  simulator.bounds_x, 1, simulator.seed)
        #Calculate y values and sse for theta_best with noise
        theta_guess_data.y_vals = simulator.gen_y_data(theta_guess_data, simulator.noise_mean, simulator.noise_std)  
        
        error = exp_data.y_vals.flatten() - theta_guess_data.y_vals.flatten()

        #Append intermediate values to list
        self.iter_param_data.append(theta_guess)
        self.iter_sse_data.append(np.sum(error**2))
        #Calculate l2
        del_theta = theta_guess - self.simulator.theta_true
        theta_l2_norm = np.linalg.norm(del_theta, ord = 2)
        self.iter_l2_norm.append(theta_l2_norm)
        self.iter_count += 1

        return error
    
    def __get_simulator_exp_data(self):
        """
        Gets the simulator and experimental data from the job

        Returns
        -------
        simulator: Simulator, the simulator object to evaluate
        exp_data: Data, the experimental data to evaluate
        tot_runs_cs: int, the total number of runs in the case study
        ftol: float, the tolerance for the objective function

        Notes
        -----
        The simulator and experimental data is consistent between all methods of a given case study
        """
        jobs = sorted(self.project.find_jobs(self.criteria_dict), key=lambda job: job._id)
        valid_files = [job.fn("BO_Results.gz") for job in jobs if os.path.exists(job.fn("BO_Results.gz"))]
        if len(valid_files) > 0:
            smallest_file = min(valid_files, key=lambda x: os.path.getsize(x))
            # Find the job corresponding to the smallest file size
            smallest_file_index = valid_files.index(smallest_file)
            job = jobs[smallest_file_index]

            #Open the statepoint of the job
            with open(job.fn("signac_statepoint.json"), 'r') as json_file:
                # Load the JSON data
                sp_data = json.load(json_file)
            #get number of total runs from statepoint
            tot_runs_cs = sp_data["bo_run_tot"]
            ftol = sp_data["obj_tol"]
            if tot_runs_cs == 1:
                if sp_data["cs_name_val"] in [2,3] and sp_data["bo_iter_tot"] == 75:
                    tot_runs_cs = 10
                else:
                    tot_runs_cs = 5
            
            #Open smallest job file
            results = open_file_helper(job.fn("BO_Results.gz"))
            #Get Experimental data and Simulator objects used in problem
            exp_data = results[0].exp_data_class
            simulator = results[0].simulator_class
            
        else:
            #Set tot_runs cs as 5 as a default
            tot_runs_cs = 5
            #Create simulator and exp Data class objects
            simulator = simulator_helper_test_fxns(self.criteria_dict["cs_name_val"], 0, None, self.seed)
            #Set num_x based off cs number
            #DO THIS
            exp_data = simulator.gen_exp_data(self.num_x, Gen_meth_enum(2), self.seed)
            ftol = 1e-7

        self.simulator = simulator
        self.exp_data = exp_data
            
        return simulator, exp_data, tot_runs_cs, ftol

    def least_squares_analysis(self, tot_runs = None):
        """
        Performs least squares regression on the problem equal to what was done with BO

        Parameters
        ----------
        tot_runs: int or None, the total number of runs to perform. Default is None

        Returns
        -------
        ls_results: pd.DataFrame, the results of the least squares regression

        Notes
        -----
        If None, tot_runs will default to 5

        """
        assert isinstance(tot_runs, int) or tot_runs is None, "tot_runs must be int or None"
        if isinstance(tot_runs, int):
            assert tot_runs > 0, "tot_runs must be > 0 if int"
        tot_runs_str = str(tot_runs) if tot_runs is not None else "cs_runs"
        cs_name_dict = {key: self.criteria_dict[key] for key in ["cs_name_val"]}
        ls_data_path = os.path.join(self.make_dir_name_from_criteria(cs_name_dict) , "ls_" + tot_runs_str + ".csv")
        # print([data_file_path for data_file_path in [data_file, data_name_file, data_true_file]])
        found_data1, ls_results = self.load_data(ls_data_path)

        if self.save_csv or not found_data1:
            #Get simulator and exp_data Data class objects
            simulator, exp_data, tot_runs_cs, ftol = self.__get_simulator_exp_data()
            
            num_restarts = tot_runs_cs if tot_runs is None else tot_runs
            len_x = exp_data.get_num_x_vals()

            #Set seed
            np.random.seed(self.seed)
            ## specify initial guesses
            #Note: As of now, I am not necessarily using the same starting points as with GPBO. 
            #MCMC and Sparse grid methods generate based on EI, which do not make sense for NLR starting points
            #Note: Starting points for optimization are saved in the driver, which is not saved in BO_Results.gz
            theta_guess = self.simulator.gen_theta_vals(num_restarts)

            #Initialize results dataframe
            column_names = ["Run", "Iter", 'Min Obj Act', 'Theta Min Obj', "Min Obj Cum.", 
                            "Theta Min Obj Cum.",'MSE', 'l2 norm', "jac evals", "Optimality", "Termination", 
                            "Run Time"]
            column_names_iter = ["Run", "Iter", 'Min Obj Act', 'Theta Min Obj', "Min Obj Cum.", 
                                 "Theta Min Obj Cum.",'MSE', 'l2 norm']
            ls_results = pd.DataFrame(columns=column_names)

            #Loop over number of runs
            for i in range(num_restarts):
                #Start timer
                time_start = time.time()
                #Find least squares solution
                Solution = optimize.least_squares(self.__ls_scipy_func, theta_guess[i] ,
                                                  bounds = simulator.bounds_theta_reg, method='trf',
                                                  args=(self.exp_data, self.simulator), verbose = 0, 
                                                  ftol = ftol)
                #End timer and calculate total run time
                time_end = time.time()
                time_per_run = time_end-time_start
                #Get sse_min from cost
                sse_min = 2*Solution.cost
                
                #Get list of iteration, sse, and parameter data
                iter_list = np.array(range(self.iter_count)) + 1
                sse_list = np.array(self.iter_sse_data)
                l2_norm_list = self.iter_l2_norm
                param_list = self.iter_param_data

                #Create a pd dataframe of all iteration information. Initialize cumulative columns as zero
                ls_iter_res = [i + 1, iter_list, sse_list, param_list, None , None, sse_list/len_x, l2_norm_list]
                iter_df = pd.DataFrame([ls_iter_res], columns = column_names_iter)
                iter_df = iter_df.apply(lambda col: col.explode(), axis=0).reset_index(drop=True).copy(deep =True)

                #Add Theta min obj and theta_sse obj
                #Loop over each iteration to create the min obj columns
                
                # for j in range(len(iter_df)):
                #     min_sse = iter_df.loc[j, "Min Obj Act"].copy()
                #     if j == 0 or min_sse < iter_df["Min Obj Act"].iloc[j-1]:
                #         min_param = iter_df["Theta Min Obj"].iloc[j].copy()
                #     else:
                #         min_sse = iter_df["Min Obj Cum."].iloc[j-1].copy()
                #         min_param = iter_df["Theta Min Obj Cum."].iloc[j-1].copy()

                #     iter_df.loc[j, "Min Obj Cum."] = min_sse
                #     iter_df.at[j, "Theta Min Obj Cum."] = min_param

                iter_df['Theta Min Obj Cum.'] = iter_df['Theta Min Obj']
                iter_df["Min Obj Cum."] = np.minimum.accumulate(iter_df['Min Obj Act'])

                for j in range(len(iter_df)):
                    if j > 0:
                        if iter_df["Min Obj Cum."].iloc[i] >= iter_df["Min Obj Cum."].iloc[i-1]:
                            iter_df.at[i, 'Theta Min Obj Cum.'] = iter_df['Theta Min Obj Cum.'].iloc[i-1].copy()
                    
                iter_df["Run Time"] = time_per_run
                iter_df["jac evals"] = Solution.njev
                iter_df["Termination"] = Solution.status
                iter_df["Optimality"] = Solution.optimality
                
                #Append to results_df
                ls_results = pd.concat([ls_results.astype(iter_df.dtypes), iter_df], ignore_index=True)

                self.seed += 1
                #Reset iter lists
                self.iter_param_data = []
                self.iter_sse_data = []
                self.iter_l2_norm = []
                self.iter_count = 0

            #Reset the index of the pandas df
            ls_results = ls_results.reset_index(drop=True)

            if self.save_csv:
                self.save_data(ls_results, ls_data_path)
        elif found_data1:
            ls_results["Theta Min Obj"] = ls_results["Theta Min Obj"].apply(self.str_to_array_df_col)
            ls_results["Theta Min Obj Cum."] = ls_results["Theta Min Obj Cum."].apply(self.str_to_array_df_col)
        return ls_results
    
    def categ_min(self, tot_runs = None):
        """
        categorize the minima found by least squares

        Parameters
        ----------
        tot_runs: int or None, the total number of runs to perform. Default is None

        Returns
        -------
        local_min_sets: pd.DataFrame, the local minima found by least squares

        Notes
        -----
        If None, tot_runs will default to 5
        """
        assert isinstance(tot_runs, int) or tot_runs is None, "tot_runs must be int or None"
        if isinstance(tot_runs, int):
            assert tot_runs > 0, "tot_runs must be > 0 if int"
        tot_runs_str = str(tot_runs) if tot_runs is not None else "cs_runs"
        cs_name_dict = {key: self.criteria_dict[key] for key in ["cs_name_val"]}
        ls_data_path = os.path.join(self.make_dir_name_from_criteria(cs_name_dict) , "ls_local_min_" + tot_runs_str + ".csv")
        found_data1, local_min_sets = self.load_data(ls_data_path)
        #Set save csv to false so that 500 restarts csv data is not saved
        save_csv_org = self.save_csv
        self.save_csv = False
        
        if self.save_csv or not found_data1:
            #Run Least Squares 500 times
            ls_results = self.least_squares_analysis(tot_runs)
            #Drop all except best iteration for each run
            ls_results_sort = ls_results.sort_values(by=['Min Obj Cum.', 'Iter'], ascending=True)
            ls_results = ls_results_sort.drop_duplicates(subset="Run", keep='first')
            # ls_results = ls_results[ls_results['Iter'] != 1] #Can go through all and not just best

            #Set save csv to True so that best restarts csv data is saved
            self.save_csv = save_csv_org

            #Get samples to filter through and drop true duplicates of parameter sets
            all_sets = ls_results[["Theta Min Obj Cum.", "Min Obj Cum.", "Optimality", "Termination"]].copy(deep=True)
            
            #Make all arrays tuples
            np_theta = all_sets["Theta Min Obj Cum."]
            all_sets["Theta Min Obj Cum."] = tuple(map(tuple, np_theta))
            
            #Drop duplicate minima
            all_sets = all_sets.drop_duplicates(subset="Theta Min Obj Cum.", keep='first')
            #Drop minima with optimality > 1e-4
            all_sets = all_sets[all_sets['Optimality'] < 1e-4]
            
            #make a dataframe to store the discarded and not discarded points
            local_min_sets = pd.DataFrame(columns = all_sets.columns)
            discarded_points = pd.DataFrame(columns=all_sets.columns)
            #Set seed
            if self.seed != None:
                np.random.seed(self.seed)

            #While you have samples to analyze
            while len(local_min_sets) + len(discarded_points) < len(all_sets):
                # Shuffle the points
                all_sets = all_sets.sample(frac=1)
                #Add the 1st object in the shuffled list to your local min sets
                new_points = pd.DataFrame(all_sets.iloc[[0]], columns = local_min_sets.columns)
                local_min_sets = pd.concat([local_min_sets.astype(new_points.dtypes), new_points] , ignore_index = True)
                
                #Set distance to be 1% of the sum of the abs values of the parameters in the new set
                distance = np.sum(abs(list(map(np.array, all_sets["Theta Min Obj Cum."].iloc[[0]].values))[0]))*1e-4
                # calculate l1 norm
                array_sets = np.array(list(map(np.array, all_sets["Theta Min Obj Cum."].values)))
                array_new = np.array(list(map(np.array, new_points["Theta Min Obj Cum."].iloc[[-1]].values)))
                dist = np.abs(array_sets - array_new)
                l1_norm = np.sum(dist, axis=1)
                # Remove any points where l2_norm <= distance
                points_to_remove = np.where(l1_norm <= distance)[0]
                disc_point_df = pd.DataFrame(all_sets.iloc[points_to_remove], columns = local_min_sets.columns)
                discarded_points = pd.concat([discarded_points.astype(disc_point_df.dtypes), disc_point_df] , ignore_index = True)
                all_sets.drop(
                    index=all_sets.index[points_to_remove], inplace=True)
                
            #Change tuples to arrays
            local_min_sets["Theta Min Obj Cum."] = local_min_sets["Theta Min Obj Cum."].apply(np.array)
            #Put in order of lowest sse and reset index
            local_min_sets = local_min_sets.sort_values(by=['Min Obj Cum.'], ascending=True)
            local_min_sets = local_min_sets.reset_index(drop=True)

            if self.save_csv:
                self.save_data(local_min_sets, ls_data_path)
        
        elif found_data1:
            local_min_sets["Theta Min Obj Cum."].apply(self.str_to_array_df_col)

        return local_min_sets

#Outdated functions below
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
        
        #If there is only 1 run, set run num to 0
        if len(loaded_results) == 1:
            run_num = 0

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
            best_error, be_theta, train_idcs = gp_emulator.calc_best_error()
            best_errors_x = None
        else:
            #Type 2 best error must be calculated given the experimental data
            best_error, be_theta, best_errors_x, train_idcs = gp_emulator.calc_best_error(method, exp_data)
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
        os.makedirs(dir_name, exist_ok=True)
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