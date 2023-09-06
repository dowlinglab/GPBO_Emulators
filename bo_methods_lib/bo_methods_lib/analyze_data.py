#Import Dependencies
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from scipy.stats import qmc
import itertools
from itertools import combinations_with_replacement, combinations, permutations
import copy

from .GPBO_Classes_New import * #Fix this later
from .GPBO_Class_fxns import * #Fix this later
import pickle

#Open Data File
def analyze_SF_data_for_plot(date_time_str, bo_method, sep_fact_list, name_cs_str):
    """
    Returns the results of the Separation Factor Study SSE data for plotting
    
    Parameters
    ----------
    date_time_str: str, The DateTime string in format year/month/day (numbers only)
    bo_method: str, the BO method name to consider
    sep_fact_list: list of str, the Separation Factors to consider
    name_cs_str: str, The case study name. Ex CS1
    
    Returns
    -------
    y_data: ndarray (n_sfs x 2), The array of the gp predicted minimum sse and actual sse data
    param_names: list of str, the names of the gp predicted minimum sse and actual sse data
    """
    
    y_data = np.zeros((len(sep_fact_list) , 2))
    sse_min_all = np.ones(len(sep_fact_list))*np.inf
    sse_min_act_all = np.ones(len(sep_fact_list))*np.inf

    for i in range(len(sep_fact_list)):
        #Pull out file
        path = date_time_str + "Data_Files/" + name_cs_str + "_BO_method_" + bo_method + "_sep_fact_" + sep_fact_list[i]
        fileObj = open(path + ".pickle", 'rb')
        results = pickle.load(fileObj)   
        fileObj.close()
        tot_runs = results[0].configuration["Number of Workflow Restarts"]
        #Loop over runs
        for j in range(tot_runs):
            run_results = results[j].results_df
            #Find lowest sse and corresponding theta
            min_sse_index = np.argmin(run_results['Min Obj']) #Should use Actual or GP min?
            min_sse = run_results['Min Obj'].iloc[min_sse_index]
            min_sse_act = run_results['Min Obj Act'].iloc[min_sse_index]

            if abs(min_sse_act) < abs(sse_min_all[i]): #Use Abs for cases of 1B and 1A?
                sse_min_all[i] = min_sse
                sse_min_act_all[i] = min_sse_act

            min_sse = run_results['Min Obj']
            min_sse_act = run_results['Min Obj Act']

    y_data[:,0] = sse_min_all
    y_data[:,1] = sse_min_act_all
    data_names = ['Min Obj', 'Min Obj Act']
    
    return y_data, data_names
    
def analyze_ep_sep_fact_study(date_time_str, bo_meth_list, study_param_list, study_id, name_cs_str, save_csv):
    """
    Saves or prints results for Exploration Bias or Separation factor Study
    
    Parameters:
    -----------
    date_time_str: str, The DateTime string in format year/month/day (numbers only)
    bo_meth_list: list of str, the BO method names to consider
    study_param_list: list of str, the Exploration Bias Methods/ Separation Factors to consider. Capital letters 2 param numbers
    name_cs_str: str, The case study name. Ex CS1
    save_csv: bool, Determines whether to print results or save them as a csv
    """
    # DateTime = "2023/09/01/"
    # DateTime = "2023/09/04/"

    #Get theta dimensions from any file
    if study_id == "EP":
        path_study_name = "_ep_method_"
        col_name = 'Best EP Method'
        csv_name = "Exploration_Bias_Exp.csv"
    else:
        path_study_name = "_sep_fact_"
        col_name = 'Best Sep Fact'
        csv_name = "Separation_Factor_Exp.csv"
        
    path = date_time_str + "Data_Files/" + name_cs_str + "_BO_method_" + bo_meth_list[0] + path_study_name + study_param_list[0]
    fileObj = open(path + ".pickle", 'rb')
    results = pickle.load(fileObj) 
    fileObj.close()
    try:
        theta_dim = results[0].configuration["Number of Parameters"]
    except:
        theta_dim = 2
    
    #Initialize overall min params
    sse_min_all = np.ones(len(bo_meth_list))*np.inf
    sse_min_act_all = np.ones(len(bo_meth_list))*np.inf
    theta_min_all = np.zeros((len(bo_meth_list),theta_dim))
    run_all = np.zeros(len(bo_meth_list))
    iter_all = np.zeros(len(bo_meth_list))
    best_study_param = ["None"]*len(bo_meth_list)

    #Loop over methods
    for i in range(len(bo_meth_list)):
        #Loop over ep_methods
        for j in range(len(study_param_list)):
            #Pull out file
            path = date_time_str + "Data_Files/" + name_cs_str + "_BO_method_" + bo_meth_list[i] + path_study_name + study_param_list[j]
            fileObj = open(path + ".pickle", 'rb')
            results = pickle.load(fileObj)   
            fileObj.close()
            param_names = results[0].simulator_class.theta_true_names
            tot_runs = results[0].configuration["Number of Workflow Restarts"]
            #Loop over runs
            for k in range(tot_runs):
                run_results = results[k].results_df
                #Find lowest sse and corresponding theta
                min_sse_index = np.argmin(run_results['Min Obj']) #Should use Actual or GP min?
                min_sse = run_results['Min Obj'].iloc[min_sse_index]
                min_sse_act = run_results['Min Obj Act'].iloc[min_sse_index]
                min_sse_theta = run_results['Theta Min Obj'].iloc[min_sse_index]

                if abs(min_sse_act) < abs(sse_min_all[i]):
                    sse_min_all[i] = min_sse
                    sse_min_act_all[i] = min_sse_act
                    theta_min_all[i,:] = min_sse_theta
                    iter_all[i] = min_sse_index + 1 #Plus one to start count at 1 and not 0
                    run_all[i] = k + 1 #Plus one to start count at 1 and not 0
                    best_study_param[i] = study_param_list[j]
        #Pandas dataframe of the lowest sse overall and corresponding theta
        column_names = ['BO Method', col_name, 'Min SSE', 'Min SSE Act'] + [f'{param_names[i]}' for i in range(theta_dim)] + ['BO Restart', 'BO Iter']
        data = {'BO Method': bo_meth_list,
            col_name: best_study_param,
            'Min SSE': sse_min_all,
            'Min SSE Act': sse_min_act_all}
        for i in range(theta_dim):
            data[param_names[i]] = theta_min_all[:, i]
        data['BO Restart'] = run_all
        data['BO Iter'] = iter_all
        EP_Analysis = pd.DataFrame(data, columns=column_names)
        #Save CSV Data

    if save_csv == True:
        path_to_save_df = date_time_str + csv_name
        EP_Analysis.to_csv(path_to_save_df, index=False)
    else:
        print(EP_Analysis)
        print(results[0].simulator_class.theta_true)
        
    return

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
    fileObj = open(file_path, 'rb')
    loaded_results = pickle.load(fileObj)
    fileObj.close()
    runs = loaded_results[run_num].configuration["Number of Workflow Restarts"]
    num_sets = loaded_results[run_num].configuration["Max BO Iters"]
    dim_hps = len(loaded_results[run_num].list_gp_emulator_class[run_num].trained_hyperparams[run_num]) + 2
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

def analyze_sse_min_sse_ei(file_path, run_num, value_names):
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
    fileObj = open(file_path, 'rb')
    loaded_results = pickle.load(fileObj)
    fileObj.close()
    runs = loaded_results[run_num].configuration["Number of Workflow Restarts"]
    num_sets = loaded_results[run_num].configuration["Max BO Iters"]
    dim_data = len(value_names) #obj, min obj, and ei
    data = np.zeros((runs, num_sets, dim_data))
    data_true = None

    for j in range(runs):
        run = loaded_results[j]
        # Extract the array and convert other elements to float
        for i in range(len(run.results_df["Min Obj"])):
            sse = run.results_df["Min Obj"].to_numpy().astype(float)[i]
            min_sse = run.results_df["Min Obj Cum."].to_numpy().astype(float)[i]
            max_ei = run.results_df["Max EI"].to_numpy().astype(float)[i]
            # Create the resulting array of shape (1, 10)
            data[j,i,0] = sse
            data[j,i,1] = min_sse
            data[j,i,2] = max_ei
            
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
    fileObj = open(file_path, 'rb')
    loaded_results = pickle.load(fileObj)
    fileObj.close()
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
    fileObj = open(file_path, 'rb')
    loaded_results = pickle.load(fileObj)
    fileObj.close()
    
    x_exp = loaded_results[run_num].exp_data_class.x_vals
    dim_data = loaded_results[run_num].list_gp_emulator_class[run_num].get_dim_gp_data() #dim training data
    data_true = loaded_results[run_num].simulator_class.theta_true

    param_names = loaded_results[run_num].simulator_class.theta_true_names
    x_names = [f"Xexp_{i}" for i in range(1, x_exp.shape[1]+1)]
    data_names = param_names+x_names

    train_data = loaded_results[run_num].list_gp_emulator_class[bo_iter].feature_train_data
    test_data = loaded_results[run_num].list_gp_emulator_class[bo_iter].feature_test_data
    val_data = loaded_results[run_num].list_gp_emulator_class[bo_iter].feature_val_data
    
    return train_data, test_data, val_data, x_exp, data_names, data_true

def analyze_heat_maps(file_path, run_num, bo_iter, pair_id):
    """
    Gets the heat map data necessary for plotting heat maps
    
    Parameters
    ----------
    file_path: str, The file path of the data
    run_num: int, The run you want to analyze. Note, run_num 1 corresponds to index 0
    bo_iter: int, The BO iteration you want to analyze. Note, bo_iter 1 corresponds to index 0
    pair_id: int or str, The pair of data parameters
    
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
    fileObj = open(file_path, 'rb')
    loaded_results = pickle.load(fileObj)
    fileObj.close()
    
    #Create Heat Map Data for a run and iter
    #Just choose the 1st key for example purposes
    if isinstance(pair_id, str):
        param_names = pair_id
    elif isinstance(pair_id, int):
        param_names = list(loaded_results[run_num].heat_map_data_dict.keys())[0]
    else:
        raise Warning("Invalid pair_id!")

    #Regeneate simulator, gp_emulator, exerimental data, best error, true theta, lowest obj theta, and highest ei theta
    gp_emulator = loaded_results[run_num].list_gp_emulator_class[bo_iter]
    heat_map_data_dict = loaded_results[run_num].heat_map_data_dict
    heat_map_data = heat_map_data_dict[param_names]
    featurized_hm_data = gp_emulator.featurize_data(heat_map_data)
    simulator = loaded_results[run_num].simulator_class

    #Get index of param set
    idcs_to_plot = [loaded_results[run_num].simulator_class.theta_true_names.index(name) for name in param_names]
    exp_data = loaded_results[run_num].exp_data_class
    best_error =  loaded_results[run_num].results_df["Best Error"].iloc[bo_iter]
    theta_true = loaded_results[run_num].simulator_class.theta_true
    theta_opt =  loaded_results[run_num].results_df["Theta Min Obj Cum."].iloc[bo_iter]
    theta_next = loaded_results[run_num].results_df["Theta Max EI"].iloc[bo_iter]
    train_theta = loaded_results[run_num].list_gp_emulator_class[bo_iter].train_data.theta_vals
    enum_method = loaded_results[run_num].configuration["Method Name Enum Value"]
    enum_ep = Ep_enum(loaded_results[run_num].configuration["Exploration Bias Method Value"])
    sep_fact = loaded_results[run_num].configuration["Separation Factor"]
    seed = loaded_results[run_num].configuration["Seed"]
    ep_at_iter = loaded_results[run_num].results_df["Exploration Bias"].iloc[bo_iter]
    meth_name = Method_name_enum(enum_method)
    method = GPBO_Methods(meth_name)
    ep_bias = Exploration_Bias(None, ep_at_iter, enum_ep, None, None, None, None, None, None, None)
    
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
        heat_map_data.sse, heat_map_data.sse_var = gp_emulator.eval_gp_sse_var_misc(heat_map_data, exp_data)

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
    ei = heat_map_data.ei.reshape(theta_pts,theta_pts).T
    
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
    fileObj = open(file_path, 'rb')
    loaded_results = pickle.load(fileObj)
    fileObj.close()
    
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

def compare_muller_heat_map(file_path, run_num, bo_iter, x_val_num, theta_choice, seed):
    """
    Compares simulation and GP data for the Muller potential over a heat map
    
    Parameters
    ----------
    file_path: str, The file path of the data
    run_num: int, The run you want to analyze. Note, run_num 1 corresponds to index 0
    bo_iter: int, The BO iteration you want to analyze. Note, bo_iter 1 corresponds to index 0
    x_val_num: int, The number of x values to make heat maps over in each dimension of x data
    theta_choice: 1D ndarray or None, the theta_value to evaluate the heat map at. If none, chosen based off seed
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
    fileObj = open(file_path, 'rb')
    loaded_results = pickle.load(fileObj)
    fileObj.close()
    
    #get exp_data and theta_opt
    exp_data = loaded_results[run_num].exp_data_class
    gp_emulator = loaded_results[run_num].list_gp_emulator_class[bo_iter]
    simulator = loaded_results[run_num].simulator_class
    sep_fact = loaded_results[run_num].configuration["Separation Factor"]
    theta_true = loaded_results[run_num].simulator_class.theta_true
    theta_obj_min =  loaded_results[run_num].results_df["Theta Min Obj Cum."].iloc[bo_iter]
    theta_ei_max = loaded_results[run_num].results_df["Theta Max EI"].iloc[bo_iter]
    train_theta = loaded_results[run_num].list_gp_emulator_class[bo_iter].train_data.theta_vals
    param_names = list(loaded_results[run_num].heat_map_data_dict.keys())[0]
    idcs_to_plot = [loaded_results[run_num].simulator_class.theta_true_names.index(name) for name in param_names]
    idcs_to_plot = [loaded_results[run_num].simulator_class.theta_true_names.index(name) for name in param_names]
    
    #Generate simulation data for x given 1 theta
    simulator.seed = seed
    sim_data_x = simulator.gen_sim_data(1, x_val_num, Gen_meth_enum(1), Gen_meth_enum(2), sep_fact, False)
    if theta_choice is not None:
        sim_data_x.theta_vals[:] = theta_choice
    
    theta_value = sim_data_x.theta_vals[0]
    featurized_sim_x_data = gp_emulator.featurize_data(sim_data_x)
    sim_data_x.gp_mean, sim_data_x.gp_var = gp_emulator.eval_gp_mean_var_misc(sim_data_x, featurized_sim_x_data)
    
    #Create a meshgrid with x and y values fron the uniwue theta values of that array
    test_mesh = sim_data_x.x_vals.reshape(x_val_num, x_val_num,-1).T

    #Calculate valus
    y_sim = sim_data_x.y_vals.reshape(x_val_num, x_val_num).T
    gp_mean = sim_data_x.gp_mean.reshape(x_val_num, x_val_num).T
    gp_var = sim_data_x.gp_var.reshape(x_val_num, x_val_num).T
    
    return test_mesh, y_sim, gp_mean, gp_var, theta_value, exp_data.x_vals, idcs_to_plot

def analyze_parity_plot_data(file_path, run_num, bo_iter):
    """
    Generates parity plot for testing data
    """
    run_num -= 1
    bo_iter -= 1
    fileObj = open(file_path, 'rb')
    loaded_results = pickle.load(fileObj)
    fileObj.close()
    
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
        test_data.sse, test_data.sse_var = gp_emulator.eval_gp_sse_var_test(exp_data)
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
    fileObj = open(file_path, 'rb')
    loaded_results = pickle.load(fileObj)
    fileObj.close()
        
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