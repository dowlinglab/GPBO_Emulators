# Import Dependencies
import numpy as np
import pandas as pd
import copy
import signac
from ast import literal_eval
from collections.abc import Iterable
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform

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
    file_path: str
        The file path of the data

    Returns
    -------
    results: pickled object
        The results stored in the .pickle or .gz file

    Raises
    ------
    AssertionError
        If the file path is not a string
    ValueError
        If the file type is not .gz or .pickle
    """
    assert isinstance(file_path, str), "file_path must be a string"
    if file_path.endswith(".pickle") or file_path.endswith(".pkl"):
        with open(file_path, "rb") as fileObj:
            results = pickle.load(fileObj)
    elif file_path.endswith(".gz"):
        with gzip.open(file_path, "rb") as fileObj:
            results = pickle.load(fileObj)
    else:
        raise ValueError("File type must be .gz or .pickle!")

    return results


class General_Analysis:
    """
    The base class for GPBO workflow analysis

    Methods
    --------------
    __init__(*args, **kwargs): Constructor method
    make_dir_name_from_criteria(dict_to_use, is_nested = False): Makes a directory string name from a criteria dictionary
    get_jobs_from_criteria(criteria_dict=None): Gets a pointer of all jobs
    str_to_array_df_col(str_arr): Used to turn arrays from csvs loaded to pd dataframes from strings into arrays
    get_df_all_jobs(criteria_dict = None, save_csv = False): Creates a dataframe of all information for a given experiment
    get_study_data_signac(job, save_csv = None): Get best data from jobs and optionally save the csvs for the data
    get_best(): Gets the best (as described from self.mode) performing data for each method in the criteria dict
    get_median(): Gets the median (as described from self.mode) performing data for each method in the criteria dict
    get_mean(): Gets the mean (as described from self.mode) performing data for each method in the criteria dict
    __get_job_list(df_data): Helper function to pull best jobs from a dataframe
    sort_by_meth(df_data): Sorts a dataframe by the GPBO method used
    __calc_L2_norm(df_data, theta_true_data): Calculates the L2 norm of the theta values in a dataframe
    load_data(path): Loads data from a file based on the file extension
    save_data(data, save_path): Saves data to a file based on the file extension
    __z_choice_helper(z_choices, theta_true_data, data_type): Helper function to get the correct data and data names for plotting
    __preprocess_analyze(job, z_choice, data_type): Helper function to preprocess data for analysis
    analyze_obj_vals(job, z_choices): Compiles objective data for plotting
    analyze_thetas(job, z_choice): Compiles parameter set data for plotting
    analyze_hypers(job): Compiles hyperparameter data for plotting
    __rebuild_cs(sp_data): Rebuilds the CaseStudyParameters instance from the job statepoint data
    analyze_parity_plot_data(job, run_num, bo_iter): Compiles parity plot data for plotting
    analyze_heat_maps(job, run_num, bo_iter, pair_id, get_ei = False): Compiles heat map data for plotting
    """

    # Class variables and attributes

    def __init__(self, criteria_dict, project, mode, save_csv):
        """
        Parameters
        ----------
        criteria_dict: dict
            Signac statepoints to consider for the job. Should include minimum of cs_name_val
        project: signac.project.Project
            The signac project to analyze
        mode: str
            The mode to analyze the data in ('act', 'acq', or 'gp')
        save_csv: bool
            Whether to save csvs

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        # Asserts
        assert isinstance(criteria_dict, dict), "criteria_dict must be a dictionary"
        assert isinstance(
            project, signac.project.Project
        ), "project must be a signac.Project object"
        assert isinstance(save_csv, bool), "save_csv must be a boolean"
        assert mode in ["act", "acq", "gp"], "mode must be 'act', 'acq', or 'gp'"
        # Collect unique statepoints of all jobs
        statepoint_names = set()
        for job in project:
            statepoint_names.update(job.statepoint().keys())
        key_list = list(statepoint_names)
        self.sp_keys_valid = key_list
        assert (
            all(key in key_list for key in list(criteria_dict.keys())) == True
        ), "All keys in criteria_dict must be in project statepoints"

        # Constructor method
        self.mode = mode
        self.criteria_dict = criteria_dict
        self.project = project
        self.study_results_dir = os.path.join(
            self.make_dir_name_from_criteria(self.criteria_dict)
        )
        self.save_csv = save_csv

    def make_dir_name_from_criteria(self, dict_to_use, is_nested=False):
        """
        Makes a directory string name from a criteria dictionary

        Parameters
        ----------
        dict_to_use: dict
            Dictionary to use to make directory name
        is_nested: bool, default False
            Whether the dictionary is nested or not

        Returns
        -------
        result_dir: str
            The directory name from the dictionary

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        For proper results, ALWAYS use this function with is_nested = False
        """
        assert isinstance(dict_to_use, dict), "dict_to_use must be a dictionary"
        assert isinstance(is_nested, bool), "is_nested must be a boolean"
        # Note, criteria dict is only checked when is_nested = False
        if not is_nested:
            assert (
                all(key in self.sp_keys_valid for key in list(dict_to_use.keys()))
                == True
            ), "All keys in criteria_dict must be in project statepoints"
        # Organize Dictionary keys and values sorted from lowest to highest
        sorted_dict = dict(
            sorted(dict_to_use.items(), key=lambda item: (item[0], item[1]))
        )

        # Make list of parts
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

        result_dir = (
            "/".join(parts)
            if is_nested
            else os.path.join("Results_" + self.mode, "/".join(parts))
        )
        return result_dir

    def get_jobs_from_criteria(self, criteria_dict=None):
        """
        Gets a pointer of all jobs
        Parameters
        ----------
        criteria_dict: dict or None, default None
            Dictionary to determine which jobs to analyze. If none, defaults to class value

        Returns
        -------
        jobs: list(siganc.job.Job)
            A list of jobs from Signac that fit criteria dict
        """
        criteria_dict = self.criteria_dict if criteria_dict == None else criteria_dict
        # Find all jobs of a certain cs and method type for the criteria in order of job id
        jobs = sorted(self.project.find_jobs(criteria_dict), key=lambda job: job._id)
        jobs = [job for job in jobs if os.path.exists(job.fn("BO_Results.gz"))]

        return jobs

    def str_to_array_df_col(self, str_arr):
        """
        Used to turn arrays from csvs loaded to pd dataframes from strings into arrays

        Parameters
        ----------
        str_arr: str
            The string to turn into an array

        Returns
        -------
        array_from_str: np.ndarray
            The array from the string

        Raises
        ------
        AssertionError
            If str_arr is not a string

        Notes
        -----
        This function is used as a lambda function in the apply method of a pandas dataframe column. It is not guaraneteed to work outside of this context.
        """
        assert isinstance(str_arr, str), "str_arr must be a string"
        # Find the index of the first space
        first_space_index = str_arr.index(" ")
        # Remove the first space if its the 2nd character (the first will be [)
        if first_space_index == 1:
            str_no_space1 = (
                str_arr[:first_space_index] + str_arr[first_space_index + 1 :]
            )
        else:
            str_no_space1 = str_arr
        # Turn the string into an array be subbing spaces with a ,
        array_from_str = np.array(ast.literal_eval(re.sub(r"\s+", ",", str_no_space1)))
        return array_from_str

    def get_df_all_jobs(self, criteria_dict=None, save_csv=False):
        """
        Creates a dataframe of all information for a given experiment

        Parameters
        ----------
        criteria_dict: dict or None, default None
            Dictionary to determine which jobs to analyze. If none defaults to class value
        save_csv: bool, default False
            Whether to save csvs
        Returns
        -------
        df_all_jobs: pd.DataFrame
            A dataframe of the all of the data for the given dictionary
        job_list: list(siganc.job.Job)
            Jobs from Signac that fit criteria dict for the methods in meth_name_val_list
        theta_true_data_w_bnds: tuple(dict, np.ndarray)
            Tuple of a dictionary of true parameter values and bounds for the parameters

        Raises
        ------
        AssertionError
            If job BO.Results.gz file does not exist
            If save_csv is not a boolean
            If criteria_dict is not a dictionary or None

        Notes
        -----
        If criteria_dict is None, the class value is used

        """
        assert isinstance(save_csv, bool), "save_csv must be a boolean"
        assert isinstance(criteria_dict, dict) or criteria_dict is None, "criteria_dict must be a dictionary or None"
        # Intialize dataframe and job list for all jobs in criteria_dict
        df_all_jobs = pd.DataFrame()
        job_list = []

        # Find all jobs of a certain cs and method type for the criteria in order of job id
        if criteria_dict == None:
            jobs = self.get_jobs_from_criteria(self.criteria_dict)
        else:
            jobs = self.get_jobs_from_criteria(criteria_dict)
        theta_true_data = None

        # Loop over each job
        for job in jobs:
            assert os.path.exists(job.fn("BO_Results.gz")), "File must exist!"
            # Add job to job list and set data_file
            job_list += [job]
            data_file = job.fn("BO_Results.gz")

            # # #See if result data exists, if so add it to df
            tab_data_path = os.path.join(job.fn("analysis_data"), "tabulated_data.csv")
            tab_param_path = os.path.join(
                job.fn("analysis_data"), "true_param_data.json"
            )
            tab_bnds_path = os.path.join(job.fn("analysis_data"), "true_bnds_data.pkl")
            found_data1, df_job = self.load_data(tab_data_path)
            found_data2, theta_true_data = self.load_data(tab_param_path)
            found_data3, theta_bnds_data = self.load_data(tab_bnds_path)
            # If results don't exist or we are overwriting our csvs, create them
            if save_csv or not found_data1 or not found_data2 or not found_data3:
                df_job, theta_true_data_w_bnds = self.get_study_data_signac(
                    job, save_csv
                )
            elif found_data1:
                try:
                    df_job["Theta Opt Acq"] = df_job["Theta Opt Acq"].apply(
                        self.str_to_array_df_col
                    )
                    df_job["Theta Min Obj"] = df_job["Theta Min Obj"].apply(
                        self.str_to_array_df_col
                    )
                    df_job["Theta Obj GP Cum"] = df_job["Theta Obj GP Cum"].apply(
                        self.str_to_array_df_col
                    )
                    df_job["Theta Obj Act Cum"] = df_job["Theta Obj Act Cum"].apply(
                        self.str_to_array_df_col
                    )
                    df_job["Theta Acq Act Cum"] = df_job["Theta Acq Act Cum"].apply(
                        self.str_to_array_df_col
                    )
                except:
                    df_job.head()

                # Add cs name data to the dataframe
                df_job["CS Name Val"] = job.sp.cs_name_val
                df_job["CS Name"] = get_cs_class_from_val(job.sp.cs_name_val).name

            theta_true_data_w_bnds = (theta_true_data, theta_bnds_data)
            # Add job dataframe to dataframe of all jobs
            df_all_jobs = pd.concat([df_all_jobs, df_job], ignore_index=False)

        # Reset index on df_all_jobs after adding all rows
        df_all_jobs = df_all_jobs.reset_index(drop=True)

        # #Open Datafile to get theta_true if necessary
        if theta_true_data is not None:
            results = open_file_helper(data_file)
            theta_true = results[0].simulator_class.theta_true
            theta_true_names = results[0].simulator_class.theta_true_names
            theta_true_bnds = results[0].simulator_class.bounds_theta_reg
            theta_true_data = dict(zip(theta_true_names, theta_true))
            theta_true_data_w_bnds = (theta_true_data, theta_true_bnds)

        return df_all_jobs, job_list, theta_true_data_w_bnds

    def get_study_data_signac(self, job, save_csv=None):
        """
        Get best data from jobs and optionally save the csvs for the data

        Parameters
        ----------
        job: signac.job.Job
            The job to get data from
        save_csv: bool, default None
            Whether to save csvs. Set to the class default if None.

        Returns
        -------
        df_job: pd.DataFrame
            The dataframe of the data for the given job
        theta_true_data_w_bnds: tuple(dict, np.ndarray)
            Tuple of a dictionary of true parameter values and bounds for the parameters

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        If save_csv is None, it is set to the class default
        """
        assert isinstance(job, signac.job.Job), "job must be a signac.job.Job object"
        assert save_csv == None or isinstance(
            save_csv, bool
        ), "save_csv must be a boolean or None"
        save_csv = self.save_csv if save_csv == None else save_csv
        # Initialize df for a single job
        df_job = pd.DataFrame()
        data_file = job.fn("BO_Results.gz")

        # Open the file and get the dataframe
        results = open_file_helper(data_file)

        # Get statepoint info
        with open(job.fn("signac_statepoint.json"), "r") as json_file:
            # Load the JSON data
            sp_data = json.load(json_file)

        # Find number of workflow restarts in that job
        tot_runs = results[0].configuration["Number of Workflow Restarts"]
        num_x_exp = results[0].exp_data_class.get_num_x_vals()
        # get theta_true from 1st run since it never changes within a case study
        theta_true = results[0].simulator_class.theta_true
        theta_true_names = results[0].simulator_class.theta_true_names
        theta_true_bnds = results[0].simulator_class.bounds_theta_reg
        theta_true_data = dict(zip(theta_true_names, theta_true))

        # Loop over runs in each job
        for run in range(tot_runs):
            # Read data as pd.df
            df_run = results[run].results_df
            # Add the EP enum value as a column
            col_vals = job.sp.ep_enum_val
            df_run["EP Method Val"] = Ep_enum(int(col_vals)).name
            # Set index as the first run in the job's run number + the run we're at in the job
            df_run["index"] = int(job.sp.bo_run_num + run)
            df_run["Job ID"] = job.id

            # Set Run numbers as columns
            df_run.rename(columns={"index": "Run Number"}, inplace=True)

            # Add run dataframe to job dataframe after
            df_job = pd.concat([df_job, df_run], ignore_index=False)

        # Reset index on job dataframe
        df_job = df_job.reset_index(drop=True)
        # Add cs name data to the dataframe
        df_job["CS Name Val"] = job.sp.cs_name_val
        df_job["CS Name"] = get_cs_class_from_val(job.sp.cs_name_val).name

        if save_csv:
            all_data_path = os.path.join(job.fn("analysis_data"), "tabulated_data.csv")
            theta_data_path = os.path.join(
                job.fn("analysis_data"), "true_param_data.json"
            )
            theta_bnds_path = os.path.join(
                job.fn("analysis_data"), "true_bnds_data.pkl"
            )
            self.save_data(df_job, all_data_path)
            self.save_data(theta_true_data, theta_data_path)
            self.save_data(theta_true_bnds, theta_bnds_path)

        true_data_w_bnds = (theta_true_data, theta_true_bnds)

        return df_job, true_data_w_bnds

    def get_best_data(self):
        """
        Gets the best (as described from self.mode) performing data for each method in the criteria dict

        Returns
        -------
        df_best: pd.DataFrame
            The best data for each method
        job_list_best: list(signac.job.Job)
            Alist of jobs from Signac corresponding to the ones in df_best
        """
        if self.mode == "act":
            obj_col = "Min Obj Act Cum"
        elif self.mode == "acq":
            obj_col = "Acq Obj Act Cum"
        elif self.mode == "gp":
            obj_col = "Min Obj GP Cum"
        # Get data from Criteria dict if you need it
        df, jobs, theta_true_data_w_bnds = self.get_df_all_jobs()
        data_best_path = os.path.join(self.study_results_dir, "best_results.csv")
        data_exists, df_best = self.load_data(data_best_path)
        if not data_exists or self.save_csv:
            # Start by sorting pd dataframe by lowest obj func value overall
            df_sorted = df.sort_values(by=[obj_col, "BO Iter"], ascending=True)
            # Then take only the 1st instance for each method
            df_best = df_sorted.drop_duplicates(subset="BO Method", keep="first").copy()
            # Calculate the L2 norm of the best runs
            df_best = self.__calc_l2_norm(df_best, theta_true_data_w_bnds)
            # Sort df_best
            df_best = self.sort_by_meth(df_best)

        # Get list of best jobs
        job_list_best = self.__get_job_list(df_best)

        # Put in a csv file in a directory based on the job
        if self.save_csv:
            self.save_data(df_best, data_best_path)

        return df_best, job_list_best

    def get_median_data(self):
        """
        Gets the median (as described from self.mode) performing data for each method in the criteria dict

        Returns
        -------
        df_median: pd.DataFrame
            The median data for each method
        job_list_med: list(signac.job.Job)
            A list of jobs from Signac corresponding to the ones in df_median
        """
        if self.mode == "act":
            obj_col = "Min Obj Act"
        elif self.mode == "acq":
            obj_col = "Acq Obj Act"
        elif self.mode == "gp":
            obj_col = "Min Obj GP"
        # Get data from Criteria dict if you need it
        df, jobs, theta_true_data_w_bnds = self.get_df_all_jobs()
        data_path = os.path.join(self.study_results_dir, "median_results.csv")
        data_exists, df_median = self.load_data(data_path)
        if not data_exists or self.save_csv:
            # Initialize df for median values
            df_median = pd.DataFrame()
            # Loop over all methods
            for meth in df["BO Method"].unique():
                # Create a new dataframe w/ just the data for one method in it
                df_meth = df[df["BO Method"] == meth]
                # Add the row corresponding to the median value of SSE to the list
                if isinstance(df_meth[obj_col].iloc[0], np.ndarray):
                    median_sse = df_meth[obj_col].quantile(interpolation="nearest")[0]
                else:
                    median_sse = df_meth[obj_col].quantile(interpolation="nearest")
                # Ensure that only one values is used if there are multiple
                med_df = pd.DataFrame(
                    [df_meth[df_meth[obj_col] == median_sse].iloc[0]],
                    columns=df_meth.columns,
                )
                # Add df to median
                df_median = pd.concat([df_median, med_df])
            # Calculate the L2 Norm for the median values
            df_median = self.__calc_l2_norm(df_median, theta_true_data_w_bnds)
            # Sort df
            df_median = self.sort_by_meth(df_median)

        # Get list of best jobs
        job_list_med = self.__get_job_list(df_median)

        # Put in a csv file in a directory based on the job
        if self.save_csv:
            self.save_data(df_median, data_path)

        return df_median, job_list_med

    def get_mean_data(self):
        """
        Gets the mean (as described from self.mode) performing data for each method in the criteria dict

        Returns
        -------
        df_mean: pd.DataFrame
            The mean data for each method
        job_list_mean: list(signac.job.Job)
            A list of jobs from Signac corresponding to the ones in df_mean
        """
        if self.mode == "act":
            obj_col = "Min Obj Act"
        elif self.mode == "acq":
            obj_col = "Acq Obj Act"
        elif self.mode == "gp":
            obj_col = "Min Obj GP"
        # Get data from Criteria dict if you need it
        df, jobs, theta_true_data_w_bnds = self.get_df_all_jobs()
        data_path = os.path.join(self.study_results_dir, "mean_results.csv")
        data_exists, df_mean = self.load_data(data_path)
        if not data_exists or self.save_csv:
            # Initialize df for median values
            df_mean = pd.DataFrame()
            # Loop over all methods
            for meth in df["BO Method"].unique():
                # Get dataframe of data for just one method
                df_meth = df[df["BO Method"] == meth]
                # Add find the true mean of the data
                if isinstance(df_meth[obj_col].iloc[0], np.ndarray):
                    df_true_mean = df_meth[obj_col].mean()[0]
                else:
                    df_true_mean = df_meth[obj_col].mean()
                # Find point closest to true mean
                df_closest_to_mean = df_meth.iloc[
                    (df_meth[obj_col] - df_true_mean).abs().argsort()[:1]
                ]
                # Add closest point to mean to df
                df_mean = pd.concat([df_mean, df_closest_to_mean])
            # Calculate the L2 Norm for the mean values
            df_mean = self.__calc_l2_norm(df_mean, theta_true_data_w_bnds)
            # Sort df
            df_mean = self.sort_by_meth(df_mean)

        # Get list of best jobs
        job_list_mean = self.__get_job_list(df_mean)

        # Put in a csv file in a directory based on the job
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
        job_list: list(signac.job.Job)
            A list of jobs from Signac corresponding to the ones in df_data
        """
        assert isinstance(df_data, pd.DataFrame), "df_data must be a pd.DataFrame"
        assert "Job ID" in df_data.columns, "Job ID must be in the columns of df_data"
        # Get list of best jobs
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
        df_data: pd.DataFrame
            The dataframe to sort

        Returns
        -------
        df_data: pd.DataFrame
            The sorted dataframe

        Raises
        ------
        AssertionError
            If df_data is not a pd.DataFrame
            If "BO Method" is not in the columns of df_data
        """
        assert isinstance(df_data, pd.DataFrame), "df_data must be a pd.DataFrame"
        assert "BO Method" in df_data.columns, "Job ID must be in the columns of df_data"
        # Put rows in order of method
        order = [
            "Conventional",
            "Log Conventional",
            "Independence",
            "Log Independence",
            "Sparse Grid",
            "Monte Carlo",
            "E[SSE]",
        ]
        # Reindex the DataFrame with the specified row order
        df_data["BO Method"] = pd.Categorical(
            df_data["BO Method"], categories=order, ordered=True
        )
        # Sort the DataFrame based on the categorical order
        df_data = df_data.sort_values(by="BO Method")
        return df_data

    def __calc_l2_norm(self, df_data, theta_true_data):
        """
        Calculates the L2 norm of the theta values in a dataframe

        Parameters
        ----------
        df_data: pd.DataFrame
            The dataframe to calculate the L2 norm for
        theta_true: tuple(dict, np.ndarray)
            Tuple of a dictionary of true parameter values and bounds for the parameters

        Returns
        -------
        df_data: pd.DataFrame
            The dataframe with the L2 norm values added
        """
        # Calculate the difference between the true values and the GP best values in the dataframe for each parameter
        theta_min_obj = np.array(
            list(df_data["Theta Min Obj"].to_numpy()[:]), dtype=np.float64
        )
        theta_true_dict, theta_bounds = theta_true_data
        theta_true = np.array(list(theta_true_dict.values()), dtype=np.float64)
        # Create scaler to scale values between 0 and 1 based on the bounds
        scaler = MinMaxScaler()
        scaler.fit([theta_bounds[0], theta_bounds[1]])
        # Calculate change in scaled theta for each row in theta_min_obj
        del_theta = scaler.transform(theta_min_obj) - scaler.transform(
            theta_true.reshape(1, -1)
        )
        theta_L2_norm = np.linalg.norm(del_theta, ord=2, axis=1)
        # Normalize scaled L2 norm values by number of dimensions
        num_theta_dims = theta_bounds.shape[1]
        normalized_L2_norms = theta_L2_norm / np.sqrt(num_theta_dims)

        df_data["L2 Norm Theta"] = normalized_L2_norms

        return df_data

    def load_data(self, path):
        """
        Loads data from a file based on the file extension

        Parameters
        ----------
        path: str, The file path of the data

        Returns
        -------
        found_data: bool
            Whether the data was found
        data: np.ndarray or pd.DataFrame or None
            The data from the file or None

        Raises
        ------
        ValueError: If the file type is not .gz, .pkl, .pickle, .npy, .csv, or .json
        """
        assert isinstance(path, str), "path_end must be str"
        # Split path into parts
        ext = os.path.splitext(path)[-1]
        assert ext in [".csv", ".npy", ".pkl", ".pickle", ".gz", ".json"], "File type not supported"
        # Extract directory name
        dirname = os.path.dirname(path)
        # #Make directory if it doesn't already exist
        # Based on extension, save in different ways
        # Check if csv already exists
        if os.path.exists(path):
            # If so, load the file
            if ext == ".csv":
                data = pd.read_csv(path, index_col=0)
            elif ext == ".npy":
                data = np.load(path, allow_pickle=True)
            elif ext == ".pkl" or ext == ".gz" or ext == ".pickle":
                data = open_file_helper(path)
            elif ext == ".json":
                with open(path, "r") as file:
                    data = json.load(file)
            else:
                raise ValueError("NOT a csv, json, npy, pkl, pickle, or gz file")
            return True, data
        else:
            return False, None

    def save_data(self, data, save_path):
        """
        Saves data to a file based on the file extension

        Parameters
        ----------
        data: Object
            The data to save
        save_path: str
            The file path to save the data

        Raises
        ------
        ValueError: If the file type is not .gz, .pkl, .pickle, .npy, .csv, or .json
        """
        assert isinstance(save_path, str), "path_end must be str"
        # Split path into parts
        ext = os.path.splitext(save_path)[-1]
        assert ext in [".csv", ".npy", ".pkl", ".pickle", ".gz", ".json"], "File type not supported"
        # Extract directory name
        dirname = os.path.dirname(save_path)
        # Make directory if it doesn't already exist
        os.makedirs(dirname, exist_ok=True)
        # Based on extension, save in different ways
        if ext == ".csv":
            data.to_csv(save_path)
        elif ext == ".npy":
            np.save(save_path, data)
        elif ext == ".json":
            with open(save_path, "w") as file:
                json.dump(data, file)
        elif ext == ".gz":
            with gzip.open(save_path, "wb", compresslevel=1) as file:
                data = pickle.dump(data, file)
        elif ext == ".pkl" or ext == ".pickle":
            with open(save_path, "wb") as file:
                data = pickle.dump(data, file)
        else:
            raise ValueError("NOT a csv, json, npy, pkl, pickle, or gz file")
        return

    def __z_choice_helper(self, z_choices, theta_true_data, data_type):
        """
        creates column and data names based on data type

        Parameters
        ----------
        z_choices: list(str)
            The choices of data to analyze
        theta_true_data: tuple(dict, np.ndarray)
            Tuples of a dictionary of true parameter values and bounds for the parameters
        data_type: str
            The type of data to analyze (parameter or objective data). Either 'objs' or 'params'.

        Returns
        -------
        col_name: list(str)
            The column names for the data
        data_names: list(str)
            The names of the data

        Raises
        ------
        AssertionError
            If the z_choices are not of the correct type

        """
        if self.mode == "act":
            obj_col_sse = "Min Obj Act"
            obj_col_sse_min = "Min Obj Act Cum"
            param_sse = "Theta Min Obj"
            param_sse_min = "Theta Obj Act Cum"
        elif self.mode == "acq":
            obj_col_sse = "Acq Obj Act"
            obj_col_sse_min = "Acq Obj Act Cum"
            param_sse = "Theta Opt Acq"
            param_sse_min = "Theta Acq Act Cum"
        elif self.mode == "gp":
            obj_col_sse = "Min Obj GP"
            obj_col_sse_min = "Min Obj GP Cum"
            param_sse = "Theta Min Obj"
            param_sse_min = "Theta Obj GP Cum"

        if data_type == "objs":
            assert isinstance(z_choices, list), "z_choices must be list of string."
            assert all(
                isinstance(item, str) for item in z_choices
            ), "z_choices elements must be string"
            assert any(
                item in z_choices for item in ["acq", "min_sse", "sse"]
            ), "z_choices must contain at least 'min_sse', 'acq', or 'sse'"
            col_name = []
            data_names = []

            if self.mode == "gp":
                label_g = "\\tilde{\mathscr{L}}(\mathbf{"
            else:
                label_g = "\mathscr{L}(\mathbf{"

            for z_choice in z_choices:
                if "sse" == z_choice:
                    theta = "\\theta}^o" if self.mode != "acq" else "\\theta^*}"
                    col_name += [obj_col_sse]
                    data_names += [label_g + theta + ")"]
                if "min_sse" == z_choice:
                    theta = "\\theta}^{\prime}"
                    col_name += [obj_col_sse_min]
                    data_names += [label_g + theta + ")"]
                if "acq" == z_choice:
                    col_name += ["Opt Acq"]
                    data_names += ["\Xi(\mathbf{\\theta^*})"]

        elif data_type == "params":
            assert isinstance(z_choices, str), "z_choices must be a string"
            assert any(
                item == z_choices for item in ["acq", "min_sse", "sse"]
            ), "z_choices must be one of 'min_sse', 'acq', or 'sse'"
            data_names = list(theta_true_data.keys())
            if "min_sse" in z_choices:
                col_name = param_sse_min
            elif "sse" == z_choices:
                col_name = param_sse
            elif "acq" in z_choices:
                col_name = "Theta Opt Acq"
            else:
                warnings.warn("z_choices must be 'acq', 'sse', or 'min_sse'.")
        return col_name, data_names

    def __preprocess_analyze(self, job, z_choice, data_type):
        """
        Preprocesses data for analysis based on data type

        Parameters
        ----------
        job: signac.job.Job
            The job to analyze
        z_choice: list(str) or str
            The choices of data to analyze. One of 'min_sse', 'sse', or 'acq'
        data_type: str
            The type of data to analyze (parameter or objective data). Either 'objs' or 'params'.

        Returns
        -------
        df_job: pd.DataFrame
            The dataframe of the data for the given job
        data: np.ndarray
            The data for plotting
        data_true: np.ndarray or None
            The reference values of the data
        sp_data: dict
            The statepoint data for the job
        tot_runs: int
            The total number of runs in the job
        data_median: np.ndarray or None
            The median values of the reference data
        """
        # Look for data if it already exists, if not create it
        # Check if we have theta data and create it if not
        tab_data_path = os.path.join(job.fn("analysis_data"), "tabulated_data.csv")
        true_param_data_path = os.path.join(
            job.fn("analysis_data"), "true_param_data.json"
        )
        found_data1, df_job = self.load_data(tab_data_path)
        found_data2, theta_true_data = self.load_data(true_param_data_path)
        data_median = None
        if not found_data1 or not found_data2:
            df_job, theta_true_data_w_bnds = self.get_study_data_signac(
                job, save_csv=False
            )
            theta_true_data = theta_true_data_w_bnds[0]
        elif found_data1:
            df_job["Theta Opt Acq"] = df_job["Theta Opt Acq"].apply(
                self.str_to_array_df_col
            )
            df_job["Theta Min Obj"] = df_job["Theta Min Obj"].apply(
                self.str_to_array_df_col
            )
            df_job["Theta Obj GP Cum"] = df_job["Theta Obj GP Cum"].apply(
                self.str_to_array_df_col
            )
            df_job["Theta Obj Act Cum"] = df_job["Theta Obj Act Cum"].apply(
                self.str_to_array_df_col
            )
            df_job["Theta Acq Act Cum"] = df_job["Theta Acq Act Cum"].apply(
                self.str_to_array_df_col
            )

        # Get statepoint info
        with open(job.fn("signac_statepoint.json"), "r") as json_file:
            # Load the JSON data
            sp_data = json.load(json_file)
            tot_runs = sp_data["bo_runs_in_job"]
            max_iters = sp_data["bo_iter_tot"]

        if data_type == "objs":
            # Get SSE data from least squares. This is the "True" value
            ls_analyzer = LS_Analysis(self.criteria_dict, self.project, self.save_csv)
            ls_results = ls_analyzer.least_squares_analysis()
            # Make a df that is only the iters of the best run
            df_sorted = ls_results.sort_values(
                by=["Min Obj Cum.", "Iter"], ascending=[True, False]
            )
            best_run = df_sorted["Run"].iloc[0]
            data_true = ls_results[ls_results["Run"] == best_run].copy()
            median_run = df_sorted["Run"].iloc[len(df_sorted) // 2]
            data_median = ls_results[ls_results["Run"] == median_run].copy()
            # data_true = min(ls_results["Min Obj Cum."])
            data = np.zeros((tot_runs, max_iters, len(z_choice)))
        elif data_type == "params":
            data_true = theta_true_data
            data = np.zeros((tot_runs, max_iters, len(list(theta_true_data.keys()))))

        # Sort df_job by run and iter
        df_job = df_job.sort_values(by=["Run Number", "BO Iter"], ascending=True)

        return df_job, data, data_true, sp_data, tot_runs, data_median

    def analyze_obj_vals(self, job, z_choices):
        """
        Gets the data into an array for for plotting any comination of sse, log_sse, and ei

        Parameters
        ----------
        job: signac.job.Job
            The job to analyze
        z_choices: list(str) or str
            The choices of data to analyze. Contains a combination of 'min_sse', 'sse', or 'acq'

        Returns
        -------
        data: np.ndarray
            The data for plotting
        data_names: list(str)
            The names of the data
        data_true: dict or None
            The reference values of the data
        sp_data: dict
            The statepoint data for the job
        data_true_med: dict or None
            The median values of the reference data

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(job, signac.job.Job), "job must be a signac.job.Job object"
        assert isinstance(
            z_choices, (Iterable, str)
        ), "z_choices must be Iterable or str"
        if isinstance(z_choices, str):
            z_choices = [z_choices]
        assert all(
            isinstance(item, str) for item in z_choices
        ), "z_choices elements must be str"
        for i in range(len(z_choices)):
            assert z_choices[i] in [
                "min_sse",
                "sse",
                "acq",
            ], "z_choices items must be 'min_sse', 'sse', or 'acq'"

        df_job, data, data_true_val, sp_data, tot_runs, data_true_med_val = (
            self.__preprocess_analyze(job, z_choices, "objs")
        )
        data_true = {}
        data_true_med = {}
        col_name, data_names = self.__z_choice_helper(z_choices, data_true, "objs")

        unique_run_nums = pd.unique(df_job["Run Number"])
        # Loop over each choice
        for z in range(len(z_choices)):
            # Loop over runs
            for i, run in enumerate(unique_run_nums):
                # Make a df of only the data which meets that run criteria
                df_run = df_job[df_job["Run Number"] == run]
                z_data = df_run[col_name[z]]
                # If sse in log choices, the "true data" is sse data from least squares
                # print(z_choices[z])
                if "sse" in z_choices[z]:
                    data_true[z_choices[z]] = data_true_val
                    data_true_med[z_choices[z]] = data_true_med_val
                    # If the z_choice is sse and the method has a log objective function value, un logscale data
                    if sp_data["meth_name_val"] in [2, 4]:
                        z_data = np.exp(z_data.values.astype(float))
                else:
                    data_true[z_choices[z]] = None
                    data_true_med[z_choices[z]] = None
                # Set data to be where it needs to go in the above data matrix
                data[i, : len(z_data), z] = z_data

        return data, data_names, data_true, sp_data, data_true_med

    def analyze_thetas(self, job, z_choice):
        """
        Gets the data into an array for for plotting parameter values

        Parameters
        ----------
        job: signac.job.Job
            The job to analyze
        z_choice: str
            The choice of data to analyze. One of 'min_sse', 'sse', or 'acq'

        Returns
        -------
        data: np.ndarray
            The data for plotting
        data_names: list(str)
            The names of the data
        data_true: dict or None
            The reference values of the data
        sp_data: dict
            The statepoint data for the job

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(job, signac.job.Job), "job must be a signac.job.Job object"
        assert isinstance(z_choice, (str)), "z_choice must be a str"
        assert z_choice in [
            "min_sse",
            "sse",
            "acq",
        ], "z_choice must be 'min_sse', 'sse', or 'acq'"

        df_job, data, data_true, sp_data, tot_runs, data_true_med = (
            self.__preprocess_analyze(job, z_choice, "params")
        )
        col_name, data_names = self.__z_choice_helper(z_choice, data_true, "params")
        # Loop over runs
        unique_run_nums = pd.unique(df_job["Run Number"])
        for i, run in enumerate(unique_run_nums):
            # Make a df of only the data which meets that run criteria
            df_run = df_job[df_job["Run Number"] == run]
            df_run_arry = np.array(
                [arr.tolist() for arr in df_run[col_name].to_numpy()]
            )
            for param in range(data.shape[-1]):
                z_data = df_run_arry[:, param]
                # Set data to be where it needs to go in the above data matrix
                data[i, : len(z_data), param] = z_data

        data_names = [element.replace("theta", "\\theta") for element in data_names]

        return data, data_names, data_true, sp_data

    def analyze_hypers(self, job):
        """
        Gets the data into an array for for plotting hyperparameters

        Parameters
        ----------
        job: signac.job.Job
            The job to analyze

        Returns
        -------
        data: np.ndarray
            The data for plotting
        data_names: list(str)
            The names of the data
        data_true: None
            The reference values of the data (None for hyperaprameters)
        sp_data: dict
            The statepoint data for the job

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(job, signac.job.Job), "job must be a signac.job.Job object"
        data_true = None
        # Check for prexisting data
        hp_data_path = os.path.join(job.fn("analysis_data"), "hyperparam_data.npy")
        hp_name_path = os.path.join(job.fn("analysis_data"), "hp_name_data.json")
        found_data1, data = self.load_data(hp_data_path)
        found_data2, data_names = self.load_data(hp_name_path)

        # Get statepoint info
        with open(job.fn("signac_statepoint.json"), "r") as json_file:
            # Load the JSON data
            sp_data = json.load(json_file)
            tot_runs = sp_data["bo_runs_in_job"]
            max_iters = sp_data["bo_iter_tot"]

        if self.save_csv or (not found_data1 and not found_data2):
            loaded_results = open_file_helper(job.fn("BO_Results_GPs.gz"))
            dim_hps = (
                len(loaded_results[0].list_gp_emulator_class[0].trained_hyperparams[0])
                + 2
            )
            data = np.zeros((tot_runs, max_iters, dim_hps))
            data_names = [f"\\ell_{i}" for i in range(1, dim_hps + 1)]
            data_names[-2] = "\sigma"
            data_names[-1] = "\\tau"

            for j in range(tot_runs):
                run = loaded_results[j]
                for i in range(len(run.list_gp_emulator_class)):
                    # Extract the array and convert other elements to float
                    array_part = run.list_gp_emulator_class[i].trained_hyperparams[0]
                    rest_part = np.array(
                        run.list_gp_emulator_class[i].trained_hyperparams[1:],
                        dtype=float,
                    )
                    hp = np.concatenate([array_part, rest_part])
                    # Create the resulting array of shape (1, 10)
                    data[j, i, :] = hp

            if self.save_csv:
                self.save_data(data, hp_data_path)
                self.save_data(data_names, hp_name_path)

        return data, data_names, data_true, sp_data

    def __rebuild_cs(self, sp_data):
        """
        builds instance of CaseStudyParameters from saved file data

        Parameters
        ----------
        sp_data: dict
            The statepoint data for the job

        Returns
        -------
        cs_params: CaseStudyParameters
            The case study parameters
        method: GPBO_Methods
            The method used
        gen_meth_theta: Gen_meth_enum
            The method used to generate theta values
        ep_enum: Ep_enum
            The method used to generate exploration bias values
        """
        method = GPBO_Methods(Method_name_enum(sp_data["meth_name_val"]))
        cs_name = (
            get_cs_class_from_val(sp_data["cs_name_val"]).name
            if "cs_name_val" in sp_data
            else "New_CS"
        )
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

        cs_params = CaseStudyParameters(
            cs_name,
            ep0,
            sep_fact,
            normalize,
            kernel,
            lenscl,
            outputscl,
            retrain_GP,
            reoptimize_obj,
            gen_heat_map_data,
            bo_iter_tot,
            bo_run_tot,
            save_data,
            DateTime,
            seed,
            obj_tol,
            ei_tol,
        )

        return cs_params, method, gen_meth_theta, ep_enum

    def analyze_parity_plot_data(self, job, run_num, bo_iter):
        """
        Generates parity plot for testing data

        Parameters
        ----------
        job: signac.job.Job
            The job to analyze
        run_num: int
            The run number to analyze
        bo_iter: int
            The bo iteration to analyze

        Returns
        -------
        test_data_obj: Data
            The evaluated testing data for the given run and iteration

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
            If the run_num or bo_iter are out of bounds
        """
        assert isinstance(job, signac.job.Job), "job must be a signac.job.Job object"
        assert isinstance(run_num, (np.int64, int)), "run_num must be an int"
        assert isinstance(bo_iter, (np.int64, int)), "bo_iter must be an int"
        # Get Best Data
        # Check if data exists, if so, load it
        # Assert that heat map data does not aleady exist
        dir_name = os.path.join(
            job.fn(""),
            "analysis_data",
            "gp_evaluations",
            "run_" + str(run_num),
            "iter_" + str(bo_iter),
        )
        data_name = os.path.join(dir_name, "test_data.pkl")
        found_data1, test_data_obj = self.load_data(data_name)

        # Get statepoint_info
        # Get statepoint info
        with open(job.fn("signac_statepoint.json"), "r") as json_file:
            # Load the JSON data
            sp_data = json.load(json_file)
        bo_runs_in_job = sp_data["bo_runs_in_job"]
        bo_run_num_int = sp_data["bo_run_num"]
        run_idx = run_num - bo_run_num_int
        meth_name_val = sp_data["meth_name_val"]
        meth_name = Method_name_enum(meth_name_val)
        method = GPBO_Methods(meth_name)

        # Otherwise Generate it
        if self.save_csv or not found_data1:
            # Open file
            results = open_file_helper(job.fn("BO_Results.gz"))
            results_GP = open_file_helper(job.fn("BO_Results_GPs.gz"))
            assert len(results) > run_idx, "run_num is out of bounds"
            assert (
                len(results_GP[run_idx].list_gp_emulator_class) > bo_iter - 1
            ), "bo_iter is out of bounds"
            gp_object = copy.copy(
                results_GP[run_idx].list_gp_emulator_class[bo_iter - 1]
            )
            simulator = copy.copy(results[run_idx].simulator_class)
            simulator.indices_to_consider = simulator.indeces_to_consider # For backwards compatibility
            exp_data = copy.copy(
                results[0].exp_data_class
            )  # Experimental data won't change

            # Get testing data if it doesn't exist
            if gp_object.test_data is None or len(gp_object.test_data.theta_vals) == 0:
                # Generate testing data if it doesn't exist
                # Get 10 num_theta points for testing
                num_x = exp_data.get_num_x_vals()
                dim_x = exp_data.get_dim_x_vals()
                use_x = int(num_x ** (1 / dim_x))
                # Make Data (For multi vs 1D X data)
                # For conventional methods, must use same x for testing data as exp values to get same results
                test_data_sim = simulator.gen_sim_data(
                    10,
                    use_x,
                    Gen_meth_enum(1),
                    Gen_meth_enum(2),
                    1.0,
                    simulator.seed,
                    False,
                )
                if method.emulator == False:
                    test_data_sim = simulator.sim_data_to_sse_sim_data(
                        method, test_data_sim, exp_data, 1.0, False
                    )
                gp_object.test_data = test_data_sim
                gp_object.feature_test_data = gp_object.featurize_data(
                    gp_object.test_data
                )
                gp_object.test_data.gp_mean, gp_object.test_data.gp_var = (
                    gp_object.eval_gp_mean_var_test()
                )

            test_data_obj = gp_object.test_data

            if self.save_csv:
                self.save_data(test_data_obj, data_name)

        return test_data_obj

    def analyze_heat_maps(self, job, run_num, bo_iter, pair_id, get_ei=False):
        """
        Generates/analyzes heat map data for the given run and iteration

        Parameters
        ----------
        job: signac.job.Job
            The job to analyze
        run_num: int
            The run number to analyze
        bo_iter: int
            The bo iteration to analyze
        pair_id: int or str
            The pair of parameters to analyze
        get_ei: bool, default False
            Whether to calculate the acquisition function

        Returns
        -------
        all_data: np.ndarray
            The data for plotting
        test_mesh: np.ndarray
            The meshgrid for the testing data
        param_info_dict: dict
            The parameter information for the given pair
        sp_data: dict
            The statepoint data for the job

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
            If the run_num or bo_iter are out of bounds
        ValueError
            pair_id is out of bounds or an invalid string
        """
        assert isinstance(job, signac.job.Job), "job must be a signac.job.Job object"
        assert isinstance(run_num, (np.int64, int)), "run_num must be an int"
        assert isinstance(bo_iter, (np.int64, int)), "bo_iter must be an int"
        assert isinstance(
            pair_id, (np.int64, int, str)
        ), "pair_id must be an int or str"
        assert isinstance(get_ei, bool), "get_ei must be a bool"

        # Assert that heat map data does not aleady exist
        dir_name = os.path.join(
            job.fn(""),
            "analysis_data",
            "gp_evaluations",
            "run_" + str(run_num),
            "iter_" + str(bo_iter),
            "pair_" + str(pair_id),
        )
        hm_path_name = os.path.join(dir_name, "hm_data.gz")
        hm_sse_path_name = os.path.join(dir_name, "hm_sse_data.gz")
        param_info_path = os.path.join(dir_name, "notable_param_info.pkl")
        try:
            found_data1, heat_map_data = self.load_data(hm_path_name)
        except:
            found_data1, heat_map_data = False, None
        found_data2, heat_map_sse_data = self.load_data(hm_sse_path_name)
        found_data3, param_info_dict = self.load_data(param_info_path)

        # Get statepoint info
        with open(job.fn("signac_statepoint.json"), "r") as json_file:
            # Load the JSON data
            sp_data = json.load(json_file)

        # Set run number to the index of the run number in the job by subtracting
        # The number of the first run in the job
        run_num -= sp_data["bo_run_num"]
        bo_iter -= 1
        cs_params, method, gen_meth_theta, ep_method = self.__rebuild_cs(sp_data)

        data_not_found = not found_data1 or not found_data2 or not found_data3
        # Initialize data_needs_ei as true
        data_needs_ei = True
        # If we don't need acq data, set data_needs_ei to False
        if get_ei == False:
            data_needs_ei = False
        # If we have all the data and we need to calculate acq, check if we have acq data
        elif not data_not_found and not self.save_csv and get_ei:
            # If we have all the data, we won't need to calculate ei
            if heat_map_sse_data.acq is not None:
                data_needs_ei = False

        # Generate driver class/ emulator class if data doesn't exist or we need to calculate acq
        if self.save_csv or data_not_found or data_needs_ei:
            loaded_results = open_file_helper(job.fn("BO_Results.gz"))
            loaded_results_GPs = open_file_helper(job.fn("BO_Results_GPs.gz"))
            assert len(loaded_results_GPs) > run_num, "run_num is out of bounds"
            assert (
                len(loaded_results_GPs[run_num].list_gp_emulator_class) > bo_iter
            ), "bo_iter is out of bounds"

            # Create Heat Map Data for a run and iter
            # Regeneate class objects
            gp_emulator = loaded_results_GPs[run_num].list_gp_emulator_class[bo_iter]
            exp_data = loaded_results[run_num].exp_data_class
            simulator = loaded_results[run_num].simulator_class
            simulator.indices_to_consider = simulator.indeces_to_consider # For backwards compatibility
            ep_at_iter = (
                loaded_results[run_num].results_df["Exploration Bias"].iloc[bo_iter]
            )
            ep_bias = Exploration_Bias(
                None, ep_at_iter, ep_method, None, None, None, None, None, None, None
            )
            driver = GPBO_Driver(
                cs_params,
                method,
                simulator,
                exp_data,
                gp_emulator.gp_sim_data,
                gp_emulator.gp_sim_data,
                gp_emulator.gp_val_data,
                gp_emulator.gp_val_data,
                gp_emulator,
                ep_bias,
                gen_meth_theta,
            )

            # Get best error metrics
            be_data, best_error_metrics = driver._GPBO_Driver__get_best_error()

        # Create heat map data if it doesn't exists
        if self.save_csv or data_not_found:
            if self.mode == "act":
                param_sse_min = "Theta Obj Act Cum"
            elif self.mode == "acq":
                param_sse_min = "Theta Acq Act Cum"
            elif self.mode == "gp":
                param_sse_min = "Theta Obj GP Cum"

            # Get important theta values
            theta_true = loaded_results[run_num].simulator_class.theta_true
            theta_opt = loaded_results[run_num].results_df[param_sse_min].iloc[bo_iter]
            theta_next = (
                loaded_results[run_num].results_df["Theta Opt Acq"].iloc[bo_iter]
            )
            train_theta = (
                loaded_results_GPs[run_num]
                .list_gp_emulator_class[bo_iter]
                .train_data.theta_vals
            )

            # Get specific heat map data or generate it
            num_x = exp_data.get_num_x_vals()
            n_points_set = len(driver.gp_emulator.gp_sim_data.get_unique_theta())
            if num_x * n_points_set**2 >= 5000:
                n_points_set = int(np.sqrt(5000 / num_x))
            loaded_results_GPs[0].heat_map_data_dict = (
                driver.create_heat_map_param_data(n_points_set)
            )
            heat_map_data_dict = loaded_results_GPs[0].heat_map_data_dict

            # Get pair ID
            if isinstance(pair_id, str):
                assert (
                    pair_id in loaded_results_GPs[0].heat_map_data_dict.keys()
                ), "pair_id is an invalid string"
                param_names = pair_id
            elif isinstance(pair_id, int):
                assert pair_id < len(
                    loaded_results_GPs[0].heat_map_data_dict.keys()
                ), "pair_id is out of bounds"
                param_names = list(loaded_results_GPs[0].heat_map_data_dict.keys())[
                    pair_id
                ]
            else:
                raise ValueError("Invalid pair_id!")

            # Initialize heat map data class
            heat_map_data_org = heat_map_data_dict[param_names]

            # Calculate GP mean and var for heat map data
            featurized_hm_data = gp_emulator.featurize_data(heat_map_data_org)
            try:
                hm_org_mean, hm_org_var = gp_emulator.eval_gp_mean_var_misc(
                    heat_map_data_org, featurized_hm_data
                )
            except:
                print(n_points_set)

            # Get index of param set and best error
            idcs_to_plot = [
                loaded_results[run_num].simulator_class.theta_true_names.index(name)
                for name in param_names
            ]

            # Set param info
            param_info_dict = {
                "true": theta_true,
                "min_sse": theta_opt,
                "opt_acq": theta_next,
                "train": train_theta,
                "names": param_names,
                "idcs": idcs_to_plot,
            }

            # If the emulator is a conventional method, create heat map data in emulator form to calculate y_vals
            if not method.emulator:
                # Make surrogate heat map data for full theta and x grid to calculate y_vals
                n_points = int(np.sqrt(heat_map_data_org.get_num_theta()))
                repeat_x = n_points**2  # Square because only 2 values at a time change
                x_vals = np.vstack(
                    [exp_data.x_vals] * repeat_x
                )  # Repeat x_vals n_points**2 number of times
                repeat_theta = (
                    exp_data.get_num_x_vals()
                )  # Repeat theta len(x) number of times
                theta_vals = np.repeat(
                    heat_map_data_org.theta_vals, repeat_theta, axis=0
                )  # Create theta data repeated
                heat_map_data = Data(
                    theta_vals,
                    x_vals,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    simulator.bounds_theta_reg,
                    simulator.bounds_x,
                    cs_params.sep_fact,
                    cs_params.seed,
                )
            else:
                heat_map_data = heat_map_data_org

            # Generate heat map data and sse heat map data sim y values
            heat_map_data.y_vals = simulator.gen_y_data(heat_map_data, 0, 0)

            # Create sse data from regular y data
            heat_map_sse_data = simulator.sim_data_to_sse_sim_data(
                method, heat_map_data, exp_data, cs_params.sep_fact, gen_val_data=False
            )
            # Set the mean and variance to the correct heat map data object
            if not method.emulator:
                heat_map_sse_data.gp_mean = hm_org_mean
                heat_map_sse_data.gp_var = hm_org_var
                heat_map_sse_data.gp_covar = heat_map_data_org.gp_covar
            else:
                heat_map_data.gp_mean = hm_org_mean
                heat_map_data.gp_var = hm_org_var
                heat_map_data.gp_covar = heat_map_data_org.gp_covar

            # Calculate SSE and SSE var
            if method.emulator == False:
                heat_map_sse_data.sse, heat_map_sse_data.sse_var = (
                    gp_emulator.eval_gp_sse_var_misc(heat_map_sse_data)
                )
            else:
                heat_map_sse_data.sse, heat_map_sse_data.sse_var = (
                    gp_emulator.eval_gp_sse_var_misc(heat_map_data, method, exp_data)
                )

        # Get EI if needed. This operation can be expensive which is why it's optional
        if data_needs_ei:
            if method.method_name.value == 7:
                heat_map_sse_data.acq = heat_map_sse_data.sse + np.sum(
                    heat_map_sse_data.sse_var
                )
            elif method.emulator == False:
                heat_map_sse_data.acq = gp_emulator.eval_ei_misc(
                    heat_map_sse_data, exp_data, ep_bias, best_error_metrics
                )[0]
            # In older data, sparse grid depth is not a set parameter. Therefore, we set the number of points to 2000
            # This will be irrelevant for non-MC and SG data anyway
            else:
                try:
                    sg_mc_samples = loaded_results[run_num].configuration[
                        "MC SG Max Points"
                    ]
                except:
                    sg_mc_samples = 2000

                # For SG and MC data, we must get the sse mean and covar for each point individually
                ei_vals = []
                for t_val in range(len(heat_map_sse_data.get_unique_theta())):
                    # Create feature data for candidate point
                    theta = heat_map_sse_data.theta_vals[t_val]
                    candidate_theta_vals = np.repeat(
                        theta.reshape(1, -1), exp_data.get_num_x_vals(), axis=0
                    )
                    candidate = Data(
                        None,
                        exp_data.x_vals,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        simulator.bounds_theta_reg,
                        simulator.bounds_x,
                        cs_params.sep_fact,
                        cs_params.seed,
                    )
                    candidate.theta_vals = candidate_theta_vals
                    gp_emulator.cand_data = candidate
                    # Set candidate point feature data
                    gp_emulator.feature_cand_data = gp_emulator.featurize_data(
                        gp_emulator.cand_data
                    )
                    # Evaluate GP mean/ stdev at theta
                    cand_mean, cand_var = gp_emulator.eval_gp_mean_var_cand()
                    # For Type 2 GP, the sse and sse_var are calculated from the gp_mean, gp_var, and experimental data
                    cand_sse_mean, cand_sse_var = gp_emulator.eval_gp_sse_var_cand(
                        method, exp_data
                    )
                    # Otherwise objective is ei
                    ei_output = gp_emulator.eval_ei_cand(
                        exp_data, ep_bias, best_error_metrics, method, sg_mc_samples
                    )[0]
                    ei_vals.append(ei_output)

                heat_map_sse_data.acq = np.array(ei_vals)

        # Save data if necessary
        if self.save_csv:
            self.save_data(heat_map_data, hm_path_name)
            self.save_data(heat_map_sse_data, hm_sse_path_name)
            self.save_data(param_info_dict, param_info_path)

        # Find the theta_vals in the given Data class to be only the 2D (varying) parts you want to plot
        theta_mesh_vals = heat_map_sse_data.theta_vals[:, param_info_dict["idcs"]]
        # Back out the number of theta points from the hm_sse_data
        theta_pts = int(np.sqrt(len(theta_mesh_vals)))
        # Create test mesh for that specific pair and set it as the new sse data theta vals.
        test_mesh = theta_mesh_vals.reshape(theta_pts, theta_pts, -1).T

        # Define sse_sim, sse_gp_mean, and sse_gp_var, and ei based on whether to report log scaled data
        sse_sim = heat_map_sse_data.y_vals
        sse_var = heat_map_sse_data.sse_var
        sse_mean = heat_map_sse_data.sse

        # Reshape data to correct shape and add to list to return
        reshape_list = [sse_sim, sse_mean, sse_var]
        all_data = [var.reshape(theta_pts, theta_pts).T for var in reshape_list]
        if get_ei:  # and heat_map_sse_data.acq is not None
            try:
                acq_new = copy.deepcopy(heat_map_sse_data.acq)
                all_data += [acq_new.reshape(theta_pts, theta_pts).T]
            except:
                print(
                    heat_map_sse_data.sse,
                    heat_map_sse_data.sse_var,
                    get_ei,
                    data_needs_ei,
                    heat_map_sse_data.acq,
                )
                all_data += [heat_map_sse_data.acq.reshape(theta_pts, theta_pts).T]
        else:
            all_data += [None]

        return all_data, test_mesh, param_info_dict, sp_data


class All_CS_Analysis(General_Analysis):
    """
    Class for analyzing GPBO workflow results from all case studies. Child class of General_Analysis

    Methods
    -------
    __init__(cs_list, meth_val_list, project, mode, save_csv): Initializes the class
    __sort_by_cs_meth(df_data): Sorts a dataframe by the method used
    get_all_data(): Gets all data for all case studies
    get_acq_last10_avg(): Get the average acquisition function value for the last 10 iterations of each run
    get_averages(): Get average computational time, max function evaltuations, and sse values for all case studies
    get_percent_true_found(cs_nums): Get the percentage of how often the true parameter value was found
    get_averages_best(): Get the average computational time, max function evaltuations, and sse values for the best run
    """

    def __init__(self, cs_list, meth_val_list, project, mode, save_csv):
        """
        Parameters
        ----------
        cs_list: list(int)
            The list of case studies (by number marker) to analyze
        meth_val_list: list(int)
            The list of methods (by number marker) to analyze
        project: signac.project.Project
            The signac project to analyze
        mode: str
            The mode to analyze the data in ('act', 'acq', or 'gp')
        save_csv: bool
            Whether to save csvs.
        """
        if len(cs_list) == 1:
            cs_name_val = cs_list[0]
        else:
            cs_name_val = {"$in": cs_list}

        if len(meth_val_list) == 1:
            meth_name_val_list = meth_val_list[0]
        else:
            meth_name_val_list = {"$in": meth_val_list}

        criteria_dict = {
            "cs_name_val": cs_name_val,
            "meth_name_val": meth_name_val_list,
        }

        super().__init__(criteria_dict, project, mode, save_csv)

        self.cs_list = cs_list
        self.meth_val_list = meth_val_list
        self.cs_x_dict = {
            "Simple Linear": 5,
            "Muller x0": 25,
            "Muller y0": 25,
            "Yield-Loss": 10,
            "Large Linear": 25,
            "BOD Curve": 10,
            "Log Logistic": 10,
            "2D Log Logistic": 25,
        }

    def __sort_by_cs_meth(self, df_data):
        """
        Sorts a dataframe by the method used

        Parameters
        ----------
        df_data: pd.DataFrame
            The dataframe to sort

        Returns
        -------
        df_data: pd.DataFrame
            The sorted dataframe
        """
        # Put rows in order of method
        order = [
            "Conventional",
            "Log Conventional",
            "Independence",
            "Log Independence",
            "Sparse Grid",
            "Monte Carlo",
            "E[SSE]",
        ]
        # Reindex the DataFrame with the specified row order
        df_data["BO Method"] = pd.Categorical(
            df_data["BO Method"], categories=order, ordered=True
        )
        # Sort the DataFrame based on the categorical order
        df_data = df_data.sort_values(by=["CS Name Val", "BO Method"])
        return df_data

    def get_all_data(self):
        """
        Gets all data for all case studies

        Returns
        -------
        df_all_jobs: pd.DataFrame
            The dataframe of all of the data for the given dictionary
        """
        # Sort by method and CS Name
        df_all_jobs, job_list, __ = self.get_df_all_jobs()
        df_all_jobs = self.__sort_by_cs_meth(df_all_jobs)

        if self.save_csv or not os.path.exists(self.study_results_dir + "all_data.csv"):
            # Save the data to a csv
            os.makedirs(self.study_results_dir, exist_ok=True)
            path_save = os.path.join(self.study_results_dir, "all_data.csv")
            df_all_jobs.to_csv(path_save, index=False)
        return df_all_jobs

    def get_acq_last10_avg(self):
        """
        Get the average acquisition function value for the last 10 iterations of each run

        Returns
        -------
        df_acq_10_avg: pd.DataFrame
            The dataframe of the average acquisition function value for the last 10 iterations of
        """
        df_all_jobs = self.get_all_data()
        # Get the last 10 of iterations of each run
        df_last_10 = df_all_jobs[
            df_all_jobs["BO Iter"] > (df_all_jobs["Max Evals"] - 10)
        ]
        # Group by CS Name and BO Method and get the mean of the acquisition function value
        grouped_stats = (
            df_last_10.groupby(["CS Name", "BO Method"])
            .agg({"Opt Acq": ["mean", "std"]})
            .reset_index()
        )
        # Flatten the MultiIndex columns
        grouped_stats.columns = ["CS Name", "BO Method", "Avg Opt Acq", "Std Opt Acq"]
        df_acq_10_avg = grouped_stats[
            ["CS Name", "BO Method", "Avg Opt Acq", "Std Opt Acq"]
        ]
        return df_acq_10_avg

    def __get_iqr(self, series):
        """
        Get the interquartile range of a series

        Parameters
        ----------
        series: pd.Series
            The series to get the interquartile range of

        Returns
        -------
        iqr: float
            The interquartile range of the series
        """

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return iqr

    def get_averages(self):
        """
        Get average computational time, max function evaltuations, and sse values for all case studies

        Returns
        -------
        df_avg_all: pd.DataFrame
            The dataframe of the average computational time, max function evaltuations, and sse values for all case studies
        """

        df_all_jobs = self.get_all_data()

        if self.mode == "act":
            obj_col_sse_min = "Min Obj Act Cum"
        elif self.mode == "acq":
            obj_col_sse_min = "Acq Obj Act Cum"
        elif self.mode == "gp":
            obj_col_sse_min = "Min Obj GP Cum"

        # Scale the objective function values for log conv and log indep
        condition = df_all_jobs["BO Method"].isin(
            ["Log Conventional", "Log Independence"]
        )
        # Multiply values in column B by 3 where the condition is true
        df_all_jobs.loc[condition, obj_col_sse_min] = np.exp(
            df_all_jobs.loc[condition, obj_col_sse_min]
        )

        grouped_stats = (
            df_all_jobs.groupby(["CS Name", "BO Method"])
            .agg(
                {
                    obj_col_sse_min: ["median", self.__get_iqr],
                    "Total Run Time": ["mean", "std"],
                    "Max Evals": ["mean", "std"],
                }
            )
            .reset_index()
        )

        # Add nonlinear least squares results
        # Loop over each case study
        for i, cs_name in enumerate(self.cs_list):
            for j, meth_val in enumerate(self.meth_val_list):
                # Create a criteria dictionary for the case study
                criteria_dict_ls = {
                    "cs_name_val": cs_name,
                    "meth_name_val": self.meth_val_list,
                }
                ls_analyzer = LS_Analysis(criteria_dict_ls, self.project, self.save_csv)
                df_best_ls = ls_analyzer.least_squares_analysis()
                df_best_ls["CS Name"] = get_cs_class_from_val(cs_name).name
                df_best_ls["BO Method"] = "NLS"
                df_best_ls.rename(columns={"Iter": "BO Iter"}, inplace=True)
                if i == 0 and len(df_best_ls) > 0:
                    df_all_ls_best = df_best_ls
                # Otherwise, concatenate the DataFrame to df_all_best
                else:
                    df_all_ls_best = pd.concat([df_all_ls_best, df_best_ls], axis=0)
        if "Max Evals" not in df_all_ls_best.columns:
            # Compute the maximum 'iter' for each 'run'
            df_all_ls_best["Max Evals"] = df_all_ls_best.groupby(["CS Name", "Run"])[
                "BO Iter"
            ].transform("max")

        # Group the data by CS Name and BO Method, and get the mean and std for each group over all runs
        grouped_stats_ls = (
            df_all_ls_best.groupby(["CS Name", "BO Method"])
            .agg(
                {
                    "Min Obj Cum.": ["median", self.__get_iqr],
                    "Run Time": ["mean", "std"],
                    "Max Evals": ["mean", "std"],
                }
            )
            .reset_index()
        )

        # Flatten the MultiIndex columns
        grouped_stats.columns = [
            "CS Name",
            "BO Method",
            "Median Loss",
            "IQR Loss",
            "Avg Time",
            "Std Time",
            "Avg Evals",
            "Std Evals",
        ]
        grouped_stats_ls.columns = [
            "CS Name",
            "BO Method",
            "Median Loss",
            "IQR Loss",
            "Avg Time",
            "Std Time",
            "Avg Evals",
            "Std Evals",
        ]

        # Create a new DataFrame with results
        df_averages = grouped_stats[
            [
                "CS Name",
                "BO Method",
                "Median Loss",
                "IQR Loss",
                "Avg Time",
                "Std Time",
                "Avg Evals",
                "Std Evals",
            ]
        ]
        df_avg_ls_best = grouped_stats_ls[
            [
                "CS Name",
                "BO Method",
                "Median Loss",
                "IQR Loss",
                "Avg Time",
                "Std Time",
                "Avg Evals",
                "Std Evals",
            ]
        ]
        df_avg_all = pd.concat([df_averages, df_avg_ls_best], axis=0)

        if self.save_csv:
            save_path = os.path.join(self.study_results_dir, "all_cs_avg_overall.csv")
            self.save_data(df_avg_all, save_path)

        return df_avg_all

    def get_percent_true_found(self, cs_nums):
        """
        Gets the percentage of how often the true parameter value was found

        Returns
        -------
        results_df: pd.DataFrame
            The dataframe of the percentage of how often the true parameter value was found

        Raises
        ------
        AssertionError
            If the cs_nums is not a list of case study numbers in [1,2,3,10,11,12,13,14]
        """
        assert all(
            item in [1, 2, 3, 10, 11, 12, 13, 14] for item in cs_nums
        ), "cs_nums must be a list of case study numbers [1,2,3,10,11,12,13,14]"
        assert isinstance(cs_nums, list), "cs_nums must be a list"
        em_cs = [get_cs_class_from_val(cs_val).name for cs_val in cs_nums]

        if self.mode == "act":
            obj_col_sse_min = "Min Obj Act Cum"
        elif self.mode == "acq":
            obj_col_sse_min = "Acq Obj Act Cum"
        elif self.mode == "gp":
            obj_col_sse_min = "Min Obj GP Cum"

        # Loop over each case study
        for i, cs_name in enumerate(self.cs_list):
            for j, meth_val in enumerate(self.meth_val_list):
                # Create a criteria dictionary for the case study
                criteria_dict = {"cs_name_val": cs_name, "meth_name_val": meth_val}
                # Evaluate the best run for the case study for each method
                try:
                    df_best_runs, job_list_best_runs = self.get_best_all_runs(
                        criteria_dict
                    )
                    # On iter 1, create the DataFrame df_all_best
                    if i == 0 and j == 0 and len(df_best_runs) > 0:
                        df_all_best = df_best_runs
                    # Otherwise, concatenate the DataFrame to df_all_best
                    else:
                        df_all_best = pd.concat([df_all_best, df_best_runs], axis=0)
                except:
                    pass
            # Add nonlinear least squares results
            # Get SSE data from least squares
            # Create a criteria dictionary for the case study
            criteria_dict_ls = {
                "cs_name_val": cs_name,
                "meth_name_val": self.meth_val_list,
            }
            ls_analyzer = LS_Analysis(criteria_dict_ls, self.project, self.save_csv)
            ls_results = ls_analyzer.least_squares_analysis()
            # Make a df that is only the iters of the best run
            df_sorted = ls_results.sort_values(
                by=["Min Obj Cum.", "Run", "Iter"], ascending=True
            )
            # Keep only the highest value for each run
            df_best_ls = df_sorted.groupby(["Run"]).first().reset_index()
            df_best_ls["CS Name"] = get_cs_class_from_val(cs_name).name
            df_best_ls["BO Method"] = "NLS"
            df_best_ls.rename(columns={"Iter": "BO Iter"}, inplace=True)
            if i == 0 and len(df_best_ls) > 0:
                df_all_ls_best = df_best_ls
            # Otherwise, concatenate the DataFrame to df_all_best
            else:
                df_all_ls_best = pd.concat([df_all_ls_best, df_best_ls], axis=0)

        # print(df_all_ls_best.head())
        if "Max Evals" not in df_all_ls_best.columns:
            # Compute the maximum 'iter' for each 'run'
            df_all_ls_best["Max Evals"] = df_all_ls_best.groupby(["CS Name", "Run"])[
                "BO Iter"
            ].transform("max")

        # Methods of interest
        em_meths = [
            "Independence",
            "Log Independence",
            "Sparse Grid",
            "Monte Carlo",
            "E[SSE]",
        ]
        st_meths = ["Conventional", "Log Conventional"]

        # Scale the objective function values for log conv and log indep
        condition = df_all_best["BO Method"].isin(
            ["Log Conventional", "Log Independence"]
        )
        # Multiply values in column B by 3 where the condition is true
        df_all_best.loc[condition, obj_col_sse_min] = np.exp(
            df_all_best.loc[condition, obj_col_sse_min]
        )

        # Calculate how often the
        results = []

        nls_df = df_all_ls_best[(df_all_ls_best["CS Name"].isin(em_cs))]
        # Calculate the number of rows where L2 Norm Theta < 10^-2
        try:
            nls_df["l2 norm"] = nls_df["l2 norm"].str.strip("[]").astype(float)
        except:
            pass
        nls_small_l2 = nls_df[nls_df["l2 norm"].values <= 10**-2]
        # Calculate the percentage
        per_nls = (len(nls_small_l2) / len(nls_df)) * 100
        range_min_specified = nls_df["l2 norm"].min()
        range_max_specified = nls_df["l2 norm"].max()
        median_specified = nls_df["l2 norm"].median()
        # Append NLS results
        results.append(
            {
                "Method": "NLS",
                "CS Names": str(em_cs),
                "Type": "NLS",
                "Percentage L2 < 10^-2": per_nls,
                "Range Min": range_min_specified,
                "Range Max": range_max_specified,
                "Median": median_specified,
            }
        )

        for method in st_meths:
            st_df = df_all_best[
                (df_all_best["CS Name"].isin(em_cs))
                & (df_all_best["BO Method"].isin([method]))
            ]
            # Calculate the number of rows where L2 Norm Theta < 10^-2
            st_small_l2 = st_df[st_df["L2 Norm Theta"] <= 10**-2]
            # Calculate the percentage
            per_st = (len(st_small_l2) / len(st_df)) * 100
            range_min_specified = st_df["L2 Norm Theta"].min()
            range_max_specified = st_df["L2 Norm Theta"].max()
            median_specified = st_df["L2 Norm Theta"].median()
            results.append(
                {
                    "Method": method,
                    "CS Names": str(em_cs),
                    "Type": "Standard",
                    "Percentage L2 < 10^-2": per_st,
                    "Range Min": range_min_specified,
                    "Range Max": range_max_specified,
                    "Median": median_specified,
                }
            )

        # Filter rows with these methods
        for method in em_meths:
            em_df = df_all_best[
                (df_all_best["CS Name"].isin(em_cs))
                & (df_all_best["BO Method"].isin([method]))
            ]
            em_small_l2 = em_df[em_df["L2 Norm Theta"] <= 10**-2]
            per_em = (len(em_small_l2) / len(em_df)) * 100
            range_min_specified = em_df["L2 Norm Theta"].min()
            range_max_specified = em_df["L2 Norm Theta"].max()
            median_specified = em_df["L2 Norm Theta"].median()
            results.append(
                {
                    "Method": method,
                    "CS Names": str(em_cs),
                    "Type": "Emulator",
                    "Percentage L2 < 10^-2": per_em,
                    "Range Min": range_min_specified,
                    "Range Max": range_max_specified,
                    "Median": median_specified,
                }
            )

        # Convert results list to DataFrame
        results_df = pd.DataFrame(results)

        st_df = df_all_best[
            (df_all_best["CS Name"].isin(em_cs))
            & (df_all_best["BO Method"].isin(st_meths))
        ]
        # Calculate the number of rows where L2 Norm Theta < 10^-2
        st_small_l2 = st_df[st_df["L2 Norm Theta"] <= 10**-2]
        # Calculate the percentage
        per_st = (len(st_small_l2) / len(st_df)) * 100
        range_min_specified = st_df["L2 Norm Theta"].min()
        range_max_specified = st_df["L2 Norm Theta"].max()
        median_specified = st_df["L2 Norm Theta"].median()
        print("All St")
        print(per_st)
        print(range_min_specified, range_max_specified, median_specified)

        em_df = df_all_best[
            (df_all_best["CS Name"].isin(em_cs))
            & (df_all_best["BO Method"].isin(em_meths))
        ]
        em_small_l2 = em_df[em_df["L2 Norm Theta"] <= 10**-2]
        per_em = (len(em_small_l2) / len(em_df)) * 100
        range_min_specified = em_df["L2 Norm Theta"].min()
        range_max_specified = em_df["L2 Norm Theta"].max()
        median_specified = em_df["L2 Norm Theta"].median()
        print("All Em")
        print(per_em)
        print(range_min_specified, range_max_specified, median_specified)

        if self.save_csv:
            os.makedirs(self.study_results_dir, exist_ok=True)
            save_path = os.path.join(self.study_results_dir, "percent_true.csv")
            self.save_data(results_df, save_path)
            self.save_data(df_all_best, self.study_results_dir + "df_all_best.csv")

        return results_df

    def get_averages_best(self):
        """
        Get median/average data for multiple properties for all case studies. Used to reporduce Figure 2 in the paper

        Returns
        -------
        df_avg_all: pd.DataFrame
            The dataframe of the median/average data for multiple properties for all case studies. Data used in Figure 2 in the paper
        """
        if self.mode == "act":
            obj_col_sse_min = "Min Obj Act Cum"
        elif self.mode == "acq":
            obj_col_sse_min = "Acq Obj Act Cum"
        elif self.mode == "gp":
            obj_col_sse_min = "Min Obj GP Cum"

        # Loop over each case study
        for i, cs_name in enumerate(self.cs_list):
            for j, meth_val in enumerate(self.meth_val_list):
                # Create a criteria dictionary for the case study
                criteria_dict = {"cs_name_val": cs_name, "meth_name_val": meth_val}
                # Evaluate the best run for the case study for each method
                try:
                    df_best_runs, job_list_best_runs = self.get_best_all_runs(
                        criteria_dict
                    )
                    # On iter 1, create the DataFrame df_all_best
                    if i == 0 and j == 0 and len(df_best_runs) > 0:
                        df_all_best = df_best_runs
                    # Otherwise, concatenate the DataFrame to df_all_best
                    else:
                        df_all_best = pd.concat([df_all_best, df_best_runs], axis=0)
                    df_all_best["F Max Evals"] = df_all_best["Max Evals"] * df_all_best[
                        "CS Name"
                    ].map(self.cs_x_dict)
                    df_all_best["F Evals"] = df_all_best["BO Iter"] * df_all_best[
                        "CS Name"
                    ].map(self.cs_x_dict)
                except:
                    pass
            # Add nonlinear least squares results
            # Get SSE data from least squares
            # Create a criteria dictionary for the case study
            criteria_dict_ls = {
                "cs_name_val": cs_name,
                "meth_name_val": self.meth_val_list,
            }
            ls_analyzer = LS_Analysis(criteria_dict_ls, self.project, self.save_csv)
            ls_results = ls_analyzer.least_squares_analysis()
            # Make a df that is only the iters of the best run
            df_sorted = ls_results.sort_values(
                by=["Min Obj Cum.", "Run", "Iter"], ascending=True
            )
            # Keep only the highest value for each run
            df_best_ls = df_sorted.groupby(["Run"]).first().reset_index()
            df_best_ls["CS Name"] = get_cs_class_from_val(cs_name).name
            df_best_ls["BO Method"] = "NLS"
            df_best_ls.rename(columns={"Iter": "BO Iter"}, inplace=True)
            if i == 0 and len(df_best_ls) > 0:
                df_all_ls_best = df_best_ls
            # Otherwise, concatenate the DataFrame to df_all_best
            else:
                df_all_ls_best = pd.concat([df_all_ls_best, df_best_ls], axis=0)

        if "Max Evals" not in df_all_ls_best.columns:
            # Compute the maximum 'iter' for each 'run'
            df_all_ls_best["Max Evals"] = df_all_ls_best.groupby(["CS Name", "Run"])[
                "BO Iter"
            ].transform("max")

        df_all_ls_best["F Max Evals"] = df_all_ls_best["Max Evals"] * df_all_ls_best[
            "CS Name"
        ].map(self.cs_x_dict)
        df_all_ls_best["F Evals"] = df_all_ls_best["BO Iter"] * df_all_ls_best[
            "CS Name"
        ].map(self.cs_x_dict)

        # Scale the objective function values for log conv and log indep
        condition = df_all_best["BO Method"].isin(
            ["Log Conventional", "Log Independence"]
        )
        # Multiply values in column B by 3 where the condition is true
        df_all_best.loc[condition, obj_col_sse_min] = np.exp(
            df_all_best.loc[condition, obj_col_sse_min]
        )

        # Group the data by CS Name and BO Method, and get the mean and std for each group over all runs
        grouped_stats = (
            df_all_best.groupby(["CS Name", "BO Method"])
            .agg(
                {
                    obj_col_sse_min: ["median", self.__get_iqr],
                    "BO Iter": ["mean", "std"],
                    "Max Evals": ["mean", "std"],
                    "Total Run Time": ["mean", "std"],
                    "F Evals": ["mean", "std"],
                    "F Max Evals": ["mean", "std"],
                }
            )
            .reset_index()
        )
        grouped_stats_ls = (
            df_all_ls_best.groupby(["CS Name", "BO Method"])
            .agg(
                {
                    "Min Obj Cum.": ["median", self.__get_iqr],
                    "BO Iter": ["mean", "std"],
                    "Max Evals": ["mean", "std"],
                    "Run Time": ["mean", "std"],
                    "F Evals": ["mean", "std"],
                    "F Max Evals": ["mean", "std"],
                }
            )
            .reset_index()
        )

        cols = [
            "CS Name",
            "BO Method",
            "Median Loss",
            "IQR Loss",
            "Avg Evals",
            "Std Evals",
            "Avg Evals Tot",
            "Std Evals Tot",
            "Avg Time",
            "Std Time",
            "Avg F Evals",
            "Std F Evals",
            "Avg F Evals Tot",
            "Std F Evals Tot",
        ]
        # Flatten the MultiIndex columns
        grouped_stats.columns = cols
        grouped_stats_ls.columns = cols

        # Create a new DataFrame with results
        df_acq_opt = self.get_acq_last10_avg()
        df_avg_best = grouped_stats[cols]
        df_avg_ls_best = grouped_stats_ls[cols]
        df_avg_best_w_acq = pd.merge(
            df_acq_opt, df_avg_best, on=["CS Name", "BO Method"]
        )
        df_avg_all = pd.concat([df_avg_best_w_acq, df_avg_ls_best], axis=0)

        if self.save_csv:
            save_path = os.path.join(self.study_results_dir, "all_cs_avg_best.csv")
            self.save_data(df_avg_all, save_path)
        return df_avg_all


class LS_Analysis(General_Analysis):
    """
    The class for Least Squares regression analysis. Child class of General_Analysis

    Methods:
    --------
    __init__(criteria_dict, project, save_csv, exp_data=None, simulator=None): Initializes the class
    __ls_scipy_func(theta_guess, exp_data, simulator): Function to define regression function for least-squares fitting
    __get_simulator_exp_data(): Gets the simulator and experimental data from the job
    least_squares_analysis(tot_runs = None): Performs least squares regression on the problem equal to what was done with BO
    categ_min(tot_runs = None): Categorizes the number of unique minima through multiple restarts of nonlinear least squares
    """

    # Inherit objects from General_Analysis
    def __init__(self, criteria_dict, project, save_csv, exp_data=None, simulator=None):
        """
        Parameters
        ----------
        criteria_dict: dict
            The criteria dictionary to analyze
        project: signac.project.Project
            The signac project to analyze
        save_csv: bool
            Whether to save csvs
        exp_data: Data, default None
            The experimental data to evaluate
        simulator: Simulator, default None
            The simulator object to evaluate

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert (
            isinstance(exp_data, Data) or exp_data == None
        ), "exp_data must be type Data or None"
        assert (
            isinstance(simulator, Simulator) or simulator == None
        ), "simulator must be type Simulator or None"
        super().__init__(criteria_dict, project, "act", save_csv)
        self.iter_param_data = []
        self.iter_sse_data = []
        self.iter_l2_norm = []
        self.iter_count = 0
        self.seed = 1
        # Placeholder that will be overwritten if None
        self.simulator = simulator
        self.exp_data = exp_data
        self.num_x = 10  # Default number of x to generate

    # Create a function to optimize, in this case, least squares fitting
    def __ls_scipy_func(self, theta_guess, exp_data, simulator):
        """
        Function to define regression function for least-squares fitting
        Parameters
        ----------
        theta_guess: np.ndarray
            The parameter set values to evaluate
        exp_data: Data
            The experimental data to evaluate
        simulator: Simulator
            The simulator object to evaluate

        Returns
        -------
        error: np.ndarray
            The error between the experimental data and the simulated data
        """
        # Repeat the theta best array once for each x value
        # Need to repeat theta_best such that it can be evaluated at every x value in exp_data using simulator.gen_y_data
        t_guess_repeat = np.repeat(
            theta_guess.reshape(1, -1), exp_data.get_num_x_vals(), axis=0
        )
        # Add instance of Data class to theta_best
        theta_guess_data = Data(
            t_guess_repeat,
            exp_data.x_vals,
            None,
            None,
            None,
            None,
            None,
            None,
            simulator.bounds_theta_reg,
            simulator.bounds_x,
            1,
            simulator.seed,
        )
        # Calculate y values and sse for theta_best with noise
        theta_guess_data.y_vals = simulator.gen_y_data(
            theta_guess_data, simulator.noise_mean, simulator.noise_std
        )

        error = exp_data.y_vals.flatten() - theta_guess_data.y_vals.flatten()

        # Append intermediate values to list
        self.iter_param_data.append(theta_guess)
        self.iter_sse_data.append(np.sum(error**2))

        # Create scaler to scale between 0 and 1
        scaler = MinMaxScaler()
        scaler.fit(
            [self.simulator.bounds_theta_reg[0], self.simulator.bounds_theta_reg[1]]
        )
        # Calculate scaled l2
        del_theta = scaler.transform(theta_guess.reshape(1, -1)) - scaler.transform(
            self.simulator.theta_true.reshape(1, -1)
        )
        theta_l2_norm = np.linalg.norm(del_theta, ord=2, axis=1) / np.sqrt(
            len(theta_guess)
        )
        self.iter_l2_norm.append(float(theta_l2_norm))
        self.iter_count += 1

        return error

    def __get_simulator_exp_data(self):
        """
        Gets the simulator and experimental data from the job

        Returns
        -------
        simulator: Simulator
            The simulator object to evaluate
        exp_data: Data
            The experimental data to evaluate
        tot_runs_cs: int
            The total number of runs in the case study
        ftol: float
            The tolerance for the objective function

        Notes
        -----
        The simulator and experimental data is consistent between all methods of a given case study
        """
        jobs = sorted(
            self.project.find_jobs(self.criteria_dict), key=lambda job: job._id
        )
        valid_files = [
            job.fn("BO_Results.gz")
            for job in jobs
            if os.path.exists(job.fn("BO_Results.gz"))
        ]
        if len(valid_files) > 0:
            smallest_file = min(valid_files, key=lambda x: os.path.getsize(x))
            # Find the job corresponding to the smallest file size
            smallest_file_index = valid_files.index(smallest_file)
            job = jobs[smallest_file_index]

            # Open the statepoint of the job
            with open(job.fn("signac_statepoint.json"), "r") as json_file:
                # Load the JSON data
                sp_data = json.load(json_file)
            # get number of total runs from statepoint
            tot_runs_cs = sp_data["bo_run_tot"]
            ftol = sp_data["obj_tol"]
            if tot_runs_cs == 1:
                if sp_data["cs_name_val"] in [2, 3] and sp_data["bo_iter_tot"] == 75:
                    tot_runs_cs = 10
                else:
                    tot_runs_cs = 5

            # Open smallest job file
            results = open_file_helper(job.fn("BO_Results.gz"))
            # Get Experimental data and Simulator objects used in problem
            exp_data = results[0].exp_data_class
            simulator = results[0].simulator_class
            simulator.indices_to_consider = simulator.indeces_to_consider # For backwards compatibility

        else:
            # Set tot_runs cs as 5 as a default
            tot_runs_cs = 5
            # Create simulator and exp Data class objects
            simulator = simulator_helper_test_fxns(
                self.criteria_dict["cs_name_val"], 0, None, self.seed
            )

            # Get criteria dict name from cs number
            cs_name_dict = get_cs_class_from_val(self.criteria_dict["cs_name_val"]).name
            # Set num_x based off cs number
            self.cs_x_dict = {
                "Simple Linear": 5,
                "Muller x0": 25,
                "Muller y0": 25,
                "Yield-Loss": 10,
                "Large Linear": 25,
                "BOD Curve": 10,
                "Log Logistic": 10,
                "2D Log Logistic": 25,
            }
            self.num_x = self.cs_x_dict[cs_name_dict]
            exp_data = simulator.gen_exp_data(self.num_x, Gen_meth_enum(2), self.seed)
            ftol = 1e-7

        self.simulator = simulator
        self.exp_data = exp_data

        return simulator, exp_data, tot_runs_cs, ftol

    def least_squares_analysis(self, tot_runs=None):
        """
        Performs least squares regression on the problem equal to what was done with BO

        Parameters
        ----------
        tot_runs: int or None, default None
            The total number of runs to perform

        Returns
        -------
        ls_results: pd.DataFrame
            The results of the least squares regression

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        If None, tot_runs will default to 5

        """
        assert (
            isinstance(tot_runs, int) or tot_runs is None
        ), "tot_runs must be int or None"
        if isinstance(tot_runs, int):
            assert tot_runs > 0, "tot_runs must be > 0 if int"
        tot_runs_str = str(tot_runs) if tot_runs is not None else "cs_runs"
        cs_name_dict = {key: self.criteria_dict[key] for key in ["cs_name_val"]}
        ls_data_path = os.path.join(
            self.make_dir_name_from_criteria(cs_name_dict),
            "ls_" + tot_runs_str + ".csv",
        )
        found_data1, ls_results = self.load_data(ls_data_path)

        if self.save_csv or not found_data1:
            # Get simulator and exp_data Data class objects
            simulator, exp_data, tot_runs_cs, ftol = self.__get_simulator_exp_data()

            num_restarts = tot_runs_cs if tot_runs is None else tot_runs
            len_x = exp_data.get_num_x_vals()

            # Set seed
            np.random.seed(self.seed)
            ## specify initial guesses
            # Note: We do not use the same starting points as with GPBO.
            # MCMC and Sparse grid methods generate based on EI, which do not make sense for NLR starting points
            # Note: Starting points for optimization are saved in the driver, which is not saved in BO_Results.gz
            theta_guess = self.simulator.gen_theta_vals(num_restarts)

            # Initialize results dataframe
            column_names = [
                "Run",
                "Iter",
                "Min Obj Act",
                "Theta Min Obj",
                "Min Obj Cum.",
                "Theta Min Obj Cum.",
                "MSE",
                "l2 norm",
                "jac evals",
                "Optimality",
                "Termination",
                "Run Time",
                "Max Evals",
            ]
            column_names_iter = [
                "Run",
                "Iter",
                "Min Obj Act",
                "Theta Min Obj",
                "Min Obj Cum.",
                "Theta Min Obj Cum.",
                "MSE",
                "l2 norm",
            ]
            ls_results = pd.DataFrame(columns=column_names)

            # Loop over number of runs
            for i in range(num_restarts):
                # Start timer
                time_start = time.time()
                # Find least squares solution
                Solution = optimize.least_squares(
                    self.__ls_scipy_func,
                    theta_guess[i],
                    jac="3-point",
                    bounds=simulator.bounds_theta_reg,
                    method="trf",
                    args=(self.exp_data, self.simulator),
                    verbose=0,
                    ftol=ftol,
                )
                # End timer and calculate total run time
                time_end = time.time()
                time_per_run = time_end - time_start
                # Get sse_min from cost
                sse_min = 2 * Solution.cost

                # Get list of iteration, sse, and parameter data
                iter_list = np.array(range(self.iter_count)) + 1
                sse_list = np.array(self.iter_sse_data)
                l2_norm_list = self.iter_l2_norm
                param_list = self.iter_param_data

                # Create a pd dataframe of all iteration information. Initialize cumulative columns as zero
                ls_iter_res = [
                    i + 1,
                    iter_list,
                    sse_list,
                    param_list,
                    None,
                    None,
                    sse_list / len_x,
                    l2_norm_list,
                ]
                iter_df = pd.DataFrame([ls_iter_res], columns=column_names_iter)
                iter_df = (
                    iter_df.apply(lambda col: col.explode(), axis=0)
                    .reset_index(drop=True)
                    .copy(deep=True)
                )

                iter_df["Theta Min Obj Cum."] = iter_df["Theta Min Obj"]
                iter_df["Min Obj Cum."] = np.minimum.accumulate(iter_df["Min Obj Act"])

                for j in range(len(iter_df)):
                    if j > 0:
                        if (
                            iter_df["Min Obj Cum."].iloc[j]
                            >= iter_df["Min Obj Cum."].iloc[j - 1]
                        ):
                            iter_df.at[j, "Theta Min Obj Cum."] = (
                                iter_df["Theta Min Obj Cum."].iloc[j - 1].copy()
                            )

                iter_df["Run Time"] = time_per_run
                iter_df["Max Evals"] = len(iter_list)
                iter_df["jac evals"] = Solution.njev
                iter_df["Termination"] = Solution.status
                iter_df["Optimality"] = Solution.optimality

                # Append to results_df
                ls_results = pd.concat(
                    [ls_results.astype(iter_df.dtypes), iter_df], ignore_index=True
                )

                self.seed += 1
                # Reset iter lists
                self.iter_param_data = []
                self.iter_sse_data = []
                self.iter_l2_norm = []
                self.iter_count = 0

            # Reset the index of the pandas df
            ls_results = ls_results.reset_index(drop=True)

            if self.save_csv:
                self.save_data(ls_results, ls_data_path)
        elif found_data1:
            ls_results["Theta Min Obj"] = ls_results["Theta Min Obj"].apply(
                self.str_to_array_df_col
            )
            ls_results["Theta Min Obj Cum."] = ls_results["Theta Min Obj Cum."].apply(
                self.str_to_array_df_col
            )

        return ls_results

    def categ_min(self, tot_runs=None):
        """
        categorize the minima found by least squares

        Parameters
        ----------
        tot_runs: int or None, default None
            The total number of runs to perform

        Returns
        -------
        local_min_sets: pd.DataFrame
            The local minima found by least squares

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        If None, tot_runs will default to 5
        """
        assert self.simulator != None, "simulator must be defined"
        assert (
            isinstance(tot_runs, int) or tot_runs is None
        ), "tot_runs must be int or None"
        if isinstance(tot_runs, int):
            assert tot_runs > 0, "tot_runs must be > 0 if int"
        tot_runs_str = str(tot_runs) if tot_runs is not None else "cs_runs"
        cs_name_dict = {key: self.criteria_dict[key] for key in ["cs_name_val"]}
        ls_data_path = os.path.join(
            self.make_dir_name_from_criteria(cs_name_dict),
            "ls_local_min_" + tot_runs_str + ".csv",
        )
        found_data1, local_min_sets = self.load_data(ls_data_path)
        
        save_csv_org = self.save_csv

        if self.save_csv or not found_data1:
            # Set save csv to false so that tot_runs restarts csv data is not saved
            self.save_csv = False
            # Run Least Squares tot_runs times
            ls_results = self.least_squares_analysis(tot_runs)
            # Drop all except best iteration for each run
            ls_results_sort = ls_results.sort_values(
                by=["Min Obj Cum.", "Iter"], ascending=True
            )
            ls_results = ls_results_sort.drop_duplicates(subset="Run", keep="first")

            # Set save csv to True so that best restarts csv data is saved
            self.save_csv = save_csv_org

            # Get samples to filter through and drop true duplicates of parameter sets
            all_sets = ls_results[
                ["Theta Min Obj Cum.", "Min Obj Cum.", "Optimality", "Termination"]
            ].copy(deep=True)

            # Make all arrays tuples
            np_theta = all_sets["Theta Min Obj Cum."]
            all_sets["Theta Min Obj Cum."] = tuple(map(tuple, np_theta))

            # Drop duplicate minima
            all_sets = all_sets.drop_duplicates(
                subset="Theta Min Obj Cum.", keep="first"
            )
            # Drop minima with optimality > 1e-4
            all_sets = all_sets[all_sets["Optimality"] < 1e-4]

            # Set seed
            if self.seed != None:
                np.random.seed(self.seed)

            #Scale values between 0 and 1 with minmax scaler
            theta_bounds = self.simulator.bounds_theta_reg
            scaler = MinMaxScaler()
            scaler = MinMaxScaler()
            scaler.fit([theta_bounds[0], theta_bounds[1]])
            all_param_sets = np.array(list(map(np.array, all_sets["Theta Min Obj Cum."].values)))
            all_param_sets_scaled = scaler.transform(all_param_sets)
            #Calculate the scaled euclidean distance between each pair of scaled points
            dist = pdist(all_param_sets_scaled)/np.sqrt(all_param_sets.shape[1])
            #Convert the condensed distance matrix to square form
            dist_sq = squareform(dist)

            #Initialize a boolean array to keep track of unique sets
            unique_mask = np.ones(all_param_sets.shape[0], dtype=bool)

            # Iterate over the upper triangle of the distance matrix
            for i in range(all_param_sets.shape[0]):
                # If the current set is already marked as non-unique, skip it
                if not unique_mask[i]:
                    continue
                # Mark sets within the threshold distance as non-unique
                within_threshold = dist_sq[i] <= 0.01
                unique_mask[within_threshold] = False
                unique_mask[i] = True  # Keep the current set

            # Filter out the unique sets from the pandas df
            local_min_sets = all_sets[unique_mask]

            # Change tuples to arrays
            # local_min_sets.loc[:,"Theta Min Obj Cum."] = local_min_sets[
            #     "Theta Min Obj Cum."
            # ].apply(np.array)

            local_min_sets = local_min_sets.copy()  # Ensure you're working with a copy
            local_min_sets["Theta Min Obj Cum."] = local_min_sets["Theta Min Obj Cum."].apply(np.array)

            # Put in order of lowest sse and reset index
            local_min_sets = local_min_sets.sort_values(
                by=["Min Obj Cum."], ascending=True
            )
            local_min_sets = local_min_sets.reset_index(drop=True)

            if self.save_csv:
                self.save_data(local_min_sets, ls_data_path)

        elif found_data1:
            local_min_sets["Theta Min Obj Cum."].apply(self.str_to_array_df_col)

        return local_min_sets
