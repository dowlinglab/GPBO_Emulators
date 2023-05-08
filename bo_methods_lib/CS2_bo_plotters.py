from matplotlib import pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import pandas as pd
import os
import matplotlib.pyplot as plt

# def save_csv(df, path, ext='csv', verbose=False):
def save_csv(df, path, ext='npy', verbose=False):
    """Save a figure from pyplot.
    Parameters
    ----------
    df: pd.DataFrame, Dataframe you want to save to a csv
    path : string
        The path (and filename, without the extension) to save the
        figure to.
    ext : string (default='csv')
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'
        
#     print(directory, filename)
    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)
    
    #If the df is a dataframe, make it a numpy array before saving, otherwise, just save it 
    if ext == "npy":
        if isinstance(df, pd.DataFrame):
            df = df.to_numpy()
        np.save(savepath, df)
    else:
        assert isinstance(df, pd.DataFrame)
        df.to_csv(savepath)
        
    if verbose:
        print("Saving to '%s'..." % savepath),

    if verbose:
        print("Done")
    return savepath

def save_fig(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.
    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.
    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    
    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")
        
def path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo = None, bo_iter= None, title_save = None, run = None, tot_iter=1, tot_runs=1, DateTime = None, sep_fact = None, is_figure = True, csv_end = None, normalize = False):
    """
    names a path
    
    Parameters
    ----------
        emulator: True/False, Determines if GP will model the function or the function error
        ep: float, float,int,tensor,ndarray (1 value) The original exploration bias parameter
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        fxn: str, The name of the function whose file path name will be created
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        t: int, int, Number of initial training points to use
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        mesh_combo: str, the name of the combination of parameters - Used to make a folder name
        bo_iter: int, integer, number of the specific BO iterations
        title_save: str or None,  A string containing the title of the file of the plot
        run, int or None, The iteration of the number of times new training points have been picked
        tot_iter: int, The total number of iterations. Printed at top of job script
        tot_runs: int, The total number of times training data/ testing data is reshuffled. Printed at top of job script
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        sep_fact: float, Between 0 and 1. Determines fraction of all data that will be used to train the GP. Default is 1.
        is_figure: bool, used for saving CSVs as part of this function and for calling the data from a CSV to make a plot
        csv_end: str, the name of the csv file
    Returns:
        path: str, The path to which the file is saved
    
    """

    obj_str = "/"+str(obj)
    len_scl = "/len_scl_varies"
    org_TP_str = "/TP_"+ str(t)
    if ep != None:
        ep = str(np.round(float(ep),3))
        exp_str = "/ep_"+ep
    Bo_itr_str = ""
    sep_fact_str = ""
    run_str = "/Single_Run"
    
    if emulator == False:
        Emulator = "/GP_Error_Emulator"
        method = ""
    else:
        Emulator = "/GP_Emulator"
        if sparse_grid == True:
            method = "/Sparse"
        elif sparse_grid == None:
            method = ""
        else:
            method = "/Approx"
            
    fxn_dict = {"plot_obj":"/SSE_Conv" , "plot_Theta":"/Param_Conv" , "plot_obj_abs_min":"/Min_SSE_Conv" , "plot_org_train":"/org_TP", "value_plotter":"/"+ str(title_save), "plot_sep_fact_min":"/Sep_Analysis", "plot_Theta_min":"/Param_Conv_min", "plot_EI_abs_max":"/Max_EI_Conv", "GP_mean_vals":"/GP_mean_vals", "GP_var_vals":"/GP_var_vals", "time_per_iter":"/time_per_iter"}
    plot = fxn_dict[fxn]
    
    if mesh_combo is not None:
        mesh_title = "/" + mesh_combo 
    else:
        mesh_title = ""
    
    if sep_fact is not None:
        sep_fact_str = "/Sep_Fact_"+str(np.round(float(sep_fact),3))
    
    if set_lengthscale is not None:
        len_scl = "/len_scl_"+ str(set_lengthscale)         
    
    if bo_iter is not None and tot_iter > 1:
        Bo_itr_str = "/Iter_" + str(bo_iter+1).zfill(len(str(tot_iter)))  
#         print("BO",len(str(tot_iter)) , tot_iter)

    if tot_runs > 1:
        if run == None:
            run_str = "/Total_Runs_" + str(tot_runs).zfill(len(str(tot_runs))) 
#         print("Rest",len(str(tot_runs)) , tot_runs)
        else: run_str = "/Run_" + str(run+1).zfill(len(str(tot_runs)))  
        
    if DateTime is not None:
#         path_org = "../"+DateTime #Will send to the Datetime folder outside of CS1
        path_org = DateTime #Will send to the Datetime folder outside of CS1
    else:
        path_org = "Test_Figs"
        
    if normalize == True:
        path_org = path_org + "/Norm_Data"
        
#         path_org = "Test_Figs"+"/Sep_Analysis2"+"/Figures"
    if is_figure == True:
        path_org = path_org + "/Figures"
    else:
        path_org = path_org + "/CSV_Data"
        
    path_end = Emulator + method + org_TP_str + obj_str + exp_str + len_scl + sep_fact_str + run_str+ plot + Bo_itr_str + mesh_title   
    
    if fxn in ["value_plotter", "plot_org_train"]:
        path = path_org + path_end      

    else:
        path = path_org + "/Convergence_Figs" + path_end 
        
    if csv_end is not None:
        path = path + csv_end
#     print(path)   
    return path

def plot_hyperparams(iterations, hyperparam, title):
    '''
    Plots Hyperparameters
    Parameters
    ----------
        Iterations: Float, number of training iterations
        hyperparam: ndarray, array of hyperparameter value at each training iteration 
        title: string, title of the graph
     
    Returns
    -------
        plt.show(), A plot of iterations and hyperparameter
    '''
    #Assert Statements and making an axis for training iterations
    assert isinstance(title, str)==True, "Title must be a string"
    iters_axis = np.linspace(0,iterations, iterations)
    assert len(iters_axis) == len(hyperparam), "Hyperparameter array must have length of # of training iterations"
    
    #Plotting hyperparameters vs training iterations
    plt.figure()
    plt.plot(iters_axis, hyperparam)
#     plt.grid(True)
    plt.xlabel('Iterations',weight='bold')
    plt.ylabel('Hyperparameter Value',weight='bold')
#     plt.title("Plot of "+title, weight='bold',fontsize = 16)
    return plt.show()

#This needs to take indecies as an argument and link indecies to a list of parameters
def plot_org_train(test_set,train_p, test_p, p_true, Xexp, emulator, sparse_grid, obj, ep, len_scl, run, save_figure, param_names_list, tot_iter=1, tot_runs=1, DateTime=None, verbose = True, sep_fact = None, save_CSV = True, normalize = False):
    '''
    Plots original training data with true value
    Parameters
    ----------
        test_set: ndarray (len_set x dim_param), array of Theta values. Created with np.meshgrid() or LHS samples
        train_p: tensor or ndarray, The training parameter space data
        test_p: tensor or ndarray, The training parameter space data
        p_true: ndarray, A 2x1 containing the true input parameters
        Xexp: ndarray, The list of Xs that will be used to generate Y
        emulator: True/False, Determines if GP will model the function or the function error
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        ep: float, float,int,tensor,ndarray (1 value) The exploration bias parameter
        len_scl: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        run, int or None, The iteration of the number of times new training points have been picked
        save_figure: True/False, Determines whether figures will be saved
        param_names_list: list, list of names of each parameter that will be plotted
        tot_iter: int, The total number of iterations. Printed at top of job script
        tot_runs: int, The total number of times training data/ testing data is reshuffled. Printed at top of job script
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        verbose: bool, Determines whether extra information about file saving is printed, Default = False
        sep_fact: float, in (0,1]. Determines fraction of all data that will be used to train the GP. Default is 1.
        save_CSV: bool, determines whether a CSV is saved when this function is called. Prevents accidental overwrite of CSVs
     
    Returns
    -------
        plt.show(), A plot of the original training data points and the true value
    '''
    #Create name of combination of parameters: ex "a1-a2"
    mesh_combo = str(param_names_list[0]) + "-" + str(param_names_list[1])
    
    #Turn tensors into np arrays
    if torch.is_tensor(train_p) == True:
        train_p = train_p.numpy()
    if torch.is_tensor(test_p) == True:
        test_p = test_p.numpy()
 
    #Define fxn name and infer total number of data points for file saving
    fxn = "plot_org_train"
    t = int(len(train_p[:,0])) + int(len(test_p[:,0]))
    
    #If using standard approach, plot in 2D
    if emulator == False:
        #Set figure details
        plt.figure(figsize = (6.4,4))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tick_params(direction="in",top=True, right=True)
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=5)
        plt.minorticks_on() # turn on minor ticks
        plt.tick_params(which="minor",direction="in",top=True, right=True)
#         plt.gca().axes.xaxis.set_ticklabels([]) # remove tick labels
#         plt.gca().axes.yaxis.set_ticklabels([])
        
        #plot training data, testing data, and true values
        plt.scatter(train_p[:,0],train_p[:,1], color="green",s=50, label = "Training Data", marker = "x", zorder = 1)
        
        plt.scatter(test_p[:,0],test_p[:,1], color="red",s=25, label = "Testing Data", marker = "x", zorder = 2)

        plt.scatter(p_true[0],p_true[1], color="blue", label = "True argmin" + r'$(e(\theta))$', s=100, marker = (5,1), zorder = 3)
        #Set plot details
        plt.legend(fontsize=10,bbox_to_anchor=(0, 1.05, 1, 0.2),borderaxespad=0)
#         plt.legend(loc = "best")
        x_label = r'$\mathbf{'+ param_names_list[0] +'}$'
        y_label = r'$\mathbf{'+ param_names_list[1] +'}$'
#         plt.xlabel(r'$\mathbf{\theta_1}$', fontsize=16, fontweight='bold')
        plt.xlabel(x_label, fontsize=16, fontweight='bold')
        plt.ylabel(y_label, fontsize=16, fontweight='bold')
        #Set axis limits based on the maximum and minimum of the parameter search space      
        x_lim_l = np.amin(np.concatenate((train_p[:,0], test_p[:,0]), axis = None))
        y_lim_l = np.amin(np.concatenate((train_p[:,1], test_p[:,1]), axis = None))
        x_lim_u = np.amax(np.concatenate((train_p[:,0], test_p[:,0]), axis = None))
        y_lim_u = np.amax(np.concatenate((train_p[:,1], test_p[:,1]), axis = None))
        plt.xlim((x_lim_l,x_lim_u))
        plt.ylim((y_lim_l,y_lim_u))
#         plt.title("Starting Training Data")
#         plt.grid(True)
        
    else:
        x_space = Xexp
        len_x = len(x_space)

        p_true_3D = np.tile(p_true, (len(Xexp),1))
#         p_true_3D = np.repeat(p_true,len_x).reshape(-1,len_x).T
        p_true_3D_full = np.hstack((p_true_3D, x_space.reshape(len_x,-1)))
#         print(p_true_3D_full)
        # Create the figure
        fig = plt.figure(figsize = (6.4,4))
        ax = fig.add_subplot(111, projection='3d')
        ax.zaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.xaxis.set_tick_params(labelsize=16)
#         ax.set_yticks(fontsize=16)
#         ax.set_zticks(fontsize=16)
        ax.tick_params(direction="in",top=True, right=True) 
        ax.locator_params(axis='y', nbins=5)
        ax.locator_params(axis='x', nbins=5)
        ax.locator_params(axis='z', nbins=5)
        ax.minorticks_on() # turn on minor ticks
        ax.tick_params(which="minor",direction="in",top=True, right=True)
#         plt.gca().axes.xaxis.set_ticklabels([]) # remove tick labels
#         plt.gca().axes.yaxis.set_ticklabels([])

    
        # Plot the values
        ax.scatter(train_p[:,0], train_p[:,1], train_p[:,2], color = "green", s=50, label = "Training Data", marker='o', zorder = 1)
#         if len(test_p) > 0:
#             try:
        ax.scatter(test_p[:,0],test_p[:,1], test_p[:,2], color="red", s=25, label = "Testing Data", marker = "x", zorder = 2)
#             except:
#                 ax.scatter(test_p[0],test_p[1], test_p[2], color="red",s=25, label = "Testing Data", marker = "x")
            
        ax.scatter(p_true_3D_full[:,0], p_true_3D_full[:,1], p_true_3D_full[:,2], color="blue", label = "True argmin" + r'$(e(\theta))$', 
                    s=100, marker = (5,1), zorder = 3)
        
        plt.legend(fontsize=10,bbox_to_anchor=(0, 1.05, 1, 0.2),borderaxespad=0)
        x_label = r'$\mathbf{'+ param_names_list[0] +'}$'
        y_label = r'$\mathbf{'+ param_names_list[1] +'}$'
#         plt.xlabel(r'$\mathbf{\theta_1}$', fontsize=16, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=16, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=16, fontweight='bold')
        ax.set_zlabel('X-Value', fontsize=16, fontweight='bold')
#         ax.legend(loc = "best")
#         plt.legend(fontsize=10,bbox_to_anchor=(0, 1.05, 1, 0.2),borderaxespad=0)
#         ax.legend(fontsize=10,bbox_to_anchor=(1.02, 0.3),borderaxespad=0)
        x_lim_l = np.amin(np.concatenate((train_p[:,0], test_p[:,0]), axis = None))
        y_lim_l = np.amin(np.concatenate((train_p[:,1], test_p[:,1]), axis = None))
        x_lim_u = np.amax(np.concatenate((train_p[:,0], test_p[:,0]), axis = None))
        y_lim_u = np.amax(np.concatenate((train_p[:,1], test_p[:,1]), axis = None))
        
#         print(x_lim_l, x_lim_u, y_lim_l, y_lim_u)
        plt.xlim((x_lim_l,x_lim_u))
        plt.ylim((y_lim_l,y_lim_u))
        ax.grid(False)
#         ax.set_zlim((np.amin(z), np.amax(zz)))
#         plt.grid(True)
#         plt.title("Starting Training Data")
    
    df_list_ends = ["test_theta", "train_theta"]
    df_list = [test_p, train_p]
    
    if save_CSV == True:
        for i in range(len(df_list_ends)):
            array_df = pd.DataFrame(df_list[i])
            path_csv = path_name(emulator, ep, sparse_grid, fxn, len_scl, t, obj, mesh_combo, bo_iter=None, title_save = None, run = run, tot_iter=tot_iter, tot_runs=tot_runs, DateTime=DateTime, sep_fact = sep_fact, is_figure = False, csv_end = "/" + df_list_ends[i], normalize = normalize)
        #How to save more efficiently without hardcoding number of columns?
            save_csv(array_df, path_csv, ext = "npy")
    
    if save_figure == True:
        path = path_name(emulator, ep, sparse_grid, fxn, len_scl, t, obj, mesh_combo, bo_iter=None, title_save = None, run = run, tot_iter=tot_iter, tot_runs=tot_runs, DateTime=DateTime, sep_fact = sep_fact, normalize = normalize)
        save_fig(path, ext='png', close=True, verbose=False)  
    
    if save_figure == False:
#     if verbose == True and save_figure == False:
        plt.show()
        plt.close()
    
#     print(fxn)
    return 

def plot_xy(x_line, x_exp, y_exp, y_GP,y_GP_long,y_true,title = "XY Comparison"):
    '''
    Plots Hyperparameters
    Parameters
    ----------
        x_line: ndarray, array of many values for which to graph y_true
        x_exp: ndarray, array of X_exp values
        y_exp: ndarray, array of Y_exp values
        y_GP: ndarray, array of y_GP values given based on GP Theta_Best
        y_GP_long: ndarray, array of y_GP values given based on GP Theta_Best using x_line
        y_true: ndarray, array of y_true values at all points in x_line
        title: str, Title of the plot
     
    Returns
    -------
        plt.show(), A plot of iterations and hyperparameter
    '''
    #assert statments
    assert isinstance(title, str)==True, "Title must be a string"
    if y_GP:
        assert len(x_exp) == len(y_exp) == len(y_GP), "Xexp, Yexp, and Y_GP must be the same length"
    else:
        assert len(x_exp) == len(y_exp), "Xexp, Yexp, and Y_GP must be the same length"
    
    #Plot x vs Y for experimental and extrapolated data
    plt.figure(figsize = (6.4,4))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(direction="in",top=True, right=True)     
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.minorticks_on() # turn on minor ticks
    plt.tick_params(which="minor",direction="in",top=True, right=True)
#     plt.gca().axes.xaxis.set_ticklabels([]) # remove tick labels
#     plt.gca().axes.yaxis.set_ticklabels([])
    
    plt.scatter(x_exp, y_exp, label = r"y $\theta_{true}$", color = "orange")
    if y_GP:
        plt.scatter(x_exp, y_GP)
    plt.plot(x_line, y_true, color = "orange")
    plt.plot(x_line, y_GP_long, "--", label = r"y $\theta_{GP}$")
    
    #Set plot details
#     plt.grid(True)
#     plt.legend(loc = "best")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, loc = "upper left")
    plt.tight_layout()
    plt.xlabel('X Value',fontsize=16, fontweight='bold')
    plt.ylabel('Y Value', fontsize=16, fontweight='bold')
#     plt.title("Plot of "+title, weight='bold',fontsize = 16)
    
    plt.show()
    
    return 

def plot_obj_abs_min(obj_abs_min, emulator, ep, sparse_grid, set_lengthscale, t, obj, save_figure, tot_iter=1, tot_runs=1,DateTime=None, sep_fact = None, save_CSV = True, normalize = False):
    '''
    Plots the absolute minimum of the objective over BO iterations
    Parameters
    ----------
        obj_abs_min: ndarray, An array containing the absolute minimum of SSE found so far at each iteration
        runs: int, The number of times to choose new training points
        emulator: True/False, Determines if GP will model the function or the function error
        ep: float, float,int,tensor,ndarray (1 value) The exploration bias parameter
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        t: int, int, Number of initial training points to use
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        save_figure: True/False, Determines whether figures will be saved
        tot_iter: int, The total number of iterations. Printed at top of job script
        tot_runs: int, The total number of times training data/ testing data is reshuffled. Printed at top of job script
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        sep_fact: float, in (0,1]. Determines fraction of all data that will be used to train the GP. Default is 1.
        save_CSV: bool, determines whether a CSV is saved when this function is called. Prevents accidental overwrite of CSVs
     
    Returns
    -------
        plt.show(), A plot of the minimum ln(SSE) vs BO iteration for each run
    '''
    fxn = "plot_obj_abs_min"
    #Make Data Frames
#     obj_mins_df = pd.DataFrame(data = obj_abs_min)
#     obj_mins_df_T = obj_mins_df.T
#     print("Obj mins", obj_mins_df)
    #Create bo_iters as an axis
#     bo_space = np.linspace(1,bo_iters,bo_iters)
    
    #Plot Minimum SSE value at each run
    plt.figure(figsize = (6.4,4))
  
    for i in range(tot_runs):
#         bo_space = np.linspace(1,bo_iters[i],bo_iters[i])
        if tot_runs == 1:
            label = "Minimum "+ r'$log(e(\theta))$'
        else:  
            label = "Run: "+str(i+1) 
        obj_mins_df_run = pd.DataFrame(data = obj_abs_min[i])
        obj_mins_df_i = obj_mins_df_run.loc[(abs(obj_mins_df_run) > 1e-6).any(axis=1),0]
        bo_len = len(obj_mins_df_i)
        bo_space = np.linspace(1,bo_len,bo_len)
        plt.step(bo_space, obj_mins_df_i, label = label)
        
    #Set plot details        
#     plt.legend(loc = "best")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, loc = "upper left")
    plt.tight_layout()
#     plt.legend(fontsize=10,bbox_to_anchor=(1.02, 0.3),borderaxespad=0)
    plt.xlabel("BO Iterations", fontsize=16, fontweight='bold')
    plt.ylabel(r'$\mathbf{log(e(\theta))}$', fontsize=16, fontweight='bold')
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(direction="in",top=True, right=True)
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=5)
    plt.minorticks_on() # turn on minor ticks
    plt.tick_params(which="minor",direction="in",top=True, right=True)
#     plt.gca().axes.xaxis.set_ticklabels([]) # remove tick labels
#     plt.gca().axes.yaxis.set_ticklabels([])
#     plt.title("BO Iteration Results: Lowest Overall ln(SSE)")
#     plt.grid(True)
    
    #Save CSVs - How to save column names as run #s automatically?
    obj_abs_min_df = pd.DataFrame(obj_abs_min)
    if save_CSV == True:
        path_csv = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo = None, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact, is_figure = False, normalize = normalize)
        save_csv(obj_abs_min_df, path_csv, ext = "npy")
        
    #Save figure path
    if save_figure == True:
        path = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo = None, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact, normalize = normalize)
        save_fig(path, ext='png', close=True, verbose=False)
    
    plt.show()
    plt.close()
    
    return 

def plot_EI_abs_max(EI_abs_max, emulator, ep, sparse_grid, set_lengthscale, t, obj, save_figure, tot_iter=1, tot_runs=1,DateTime=None, sep_fact = None, save_CSV = True, normalize = False):
    '''
    Plots the absolute minimum of the objective over BO iterations
    Parameters
    ----------
        obj_abs_min: ndarray, An array containing the absolute minimum of SSE found so far at each iteration
        runs: int, The number of times to choose new training points
        emulator: True/False, Determines if GP will model the function or the function error
        ep: float, float,int,tensor,ndarray (1 value) The exploration bias parameter
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        t: int, int, Number of initial training points to use
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        save_figure: True/False, Determines whether figures will be saved
        tot_iter: int, The total number of iterations. Printed at top of job script
        tot_runs: int, The total number of times training data/ testing data is reshuffled. Printed at top of job script
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        sep_fact: float, in (0,1]. Determines fraction of all data that will be used to train the GP. Default is 1.
        save_CSV: bool, determines whether a CSV is saved when this function is called. Prevents accidental overwrite of CSVs
     
    Returns
    -------
        plt.show(), A plot of the minimum ln(SSE) vs BO iteration for each run
    '''
    fxn = "plot_EI_abs_max"
    #Make Data Frames
#     obj_mins_df = pd.DataFrame(data = obj_abs_min)
#     obj_mins_df_T = obj_mins_df.T
#     print("Obj mins", obj_mins_df)
    #Create bo_iters as an axis
#     bo_space = np.linspace(1,bo_iters,bo_iters)
    
    #Plot Minimum SSE value at each run
    plt.figure(figsize = (6.4,4))
  
    for i in range(tot_runs):
#         bo_space = np.linspace(1,bo_iters[i],bo_iters[i])
        if tot_runs == 1:
            label = "Minimum "+ r'$log(e(\theta))$'
        else:  
            label = "Run: "+str(i+1) 
        EI_max_df_run = pd.DataFrame(data = EI_abs_max[i])
        EI_max_df_bo_axis = EI_max_df_run.loc[(abs(EI_max_df_run) > 1e-6).any(axis=1),0]
        EI_max_df_i = EI_max_df_run.loc[:,0]
        if len(EI_max_df_bo_axis) != len(EI_max_df_i):
            EI_max_df_i = EI_max_df_i[0:int(len(EI_max_df_bo_axis)+3)] #+2 for stopping criteria + 1 to include last point
        bo_len = len(EI_max_df_i)
        bo_space = np.linspace(1,bo_len,bo_len)
        plt.step(bo_space, EI_max_df_i, label = label)
        
    #Set plot details        
#     plt.legend(loc = "best")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, loc = "upper left")
    plt.tight_layout()
#     plt.legend(fontsize=10,bbox_to_anchor=(1.02, 0.3),borderaxespad=0)
    plt.xlabel("BO Iterations", fontsize=16, fontweight='bold')
    plt.ylabel(r'$\mathbf{E(I(\theta))}$', fontsize=16, fontweight='bold')
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(direction="in",top=True, right=True)
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
    plt.minorticks_on() # turn on minor ticks
    plt.tick_params(which="minor",direction="in",top=True, right=True)
#     plt.gca().axes.xaxis.set_ticklabels([]) # remove tick labels
#     plt.gca().axes.yaxis.set_ticklabels([])
#     plt.title("BO Iteration Results: Lowest Overall ln(SSE)")
#     plt.grid(True)
    
    #Save CSVs - How to save column names as run #s automatically?
    EI_abs_max_df = pd.DataFrame(EI_abs_max)
    if save_CSV == True:
        path_csv = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo = None, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact, is_figure = False, normalize = normalize)
#         print(path_csv)
        save_csv(EI_abs_max_df, path_csv, ext = "npy")
        
    #Save figure path
    if save_figure == True:
        path = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo = None, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact, normalize = normalize)
        save_fig(path, ext='png', close=True, verbose=False)
    
    plt.show()
    plt.close()
    
    return 

def plot_sep_fact_min(bo_iters, obj_abs_min, emulator, ep, sparse_grid, set_lengthscale, t, obj, save_figure, tot_iter=1 ,DateTime=None, sep_list = None, save_CSV = True, normalize = False):
    '''
    Plots the absolute minimum of the objective over BO iterations
    Parameters
    ----------
        bo_iters: integer, number of BO iterations
        obj_abs_min: ndarray, An array containing the absolute minimum of SSE found so far at each iteration
        runs: int, The number of times to choose new training points
        emulator: True/False, Determines if GP will model the function or the function error
        ep: float, float,int,tensor,ndarray (1 value) The exploration bias parameter
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        t: int, int, Number of initial training points to use
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        save_figure: True/False, Determines whether figures will be saved
        tot_iter: int, The total number of iterations. Printed at top of job script
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        sep_list: list/ndarray, elements in (0,1]. Array of fractions of all data that will be used to train the GP. Default is 1.
        save_CSV: bool, determines whether a CSV is saved when this function is called. Prevents accidental overwrite of CSVs     
    Returns
    -------
        plt.show(), A plot of the minimum ln(SSE) vs BO iteration for each run
    '''
    fxn = "plot_sep_fact_min"
    #Create bo_iters as an axis
    bo_space = np.linspace(1,bo_iters,bo_iters)
    
    #Plot Minimum SSE value at each run
    plt.figure(figsize = (6.4,4))
    
    tot_runs = int(len(sep_list))
    for i in range(tot_runs):
        if tot_runs == 1:
            label = "Minimum " + "r'$log(e(\theta))$'"
        else:  
            label = "Sep: "+str(np.round(sep_list[i],3))
        plt.step(bo_space, obj_abs_min[i], label = label)
        
        
    #Set plot details        
#     plt.legend(loc = "best")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, loc = "upper left")
    plt.tight_layout()
    
    plt.xlabel("BO Iterations", fontsize=16, fontweight='bold')
    plt.ylabel(r'$\mathbf{log(e(\theta))}$', fontsize=16, fontweight='bold')
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(direction="in",top=True, right=True)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.minorticks_on() # turn on minor ticks
    plt.tick_params(which="minor",direction="in",top=True, right=True)

#     plt.gca().axes.xaxis.set_ticklabels([]) # remove tick labels
#     plt.gca().axes.yaxis.set_ticklabels([])
#     plt.title("BO Iteration Results: Lowest Overall ln(SSE)")
#     plt.grid(True)
    
    #Save CSVs
    obj_abs_min_df = pd.DataFrame(obj_abs_min)
    if save_CSV == True:
        path_csv = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo = None, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact, is_figure = False, csv_end = "Min_SSE_Conv_Sep_Fact", normalize = normalize)
        save_csv(obj_abs_min_df, path_csv, ext = "npy")
    
    #Save figure path
    if save_figure == True:
        path = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo = None, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = None, normalize = normalize)
        save_fig(path, ext='png', close=True, verbose=False)
    
    plt.show()
    plt.close()
    
    return 

def plot_obj(obj_array, t, obj, ep, emulator, sparse_grid, set_lengthscale, save_figure, tot_iter=1, tot_runs=1, DateTime=None, sep_fact = None, save_CSV = True, normalize = False):
    """
    Plots the objective function and Theta values vs BO iteration
    
    Parameters
    ----------
        obj_array: ndarry, (nx1): The output array containing objective function values
        t: int, Number of initial training points to use
        obj: string, name of objective function. Default "obj"
        ep: int or float, exploration parameter. Used for naming
        emulator: True/False, Determines if GP will model the function or the function error
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_figure: True/False, Determines whether figures will be saved
        tot_iter: int, The total number of iterations. Printed at top of job script
        tot_runs: int, The total number of times training data/ testing data is reshuffled. Printed at top of job script
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        sep_fact: float, in (0,1]. Determines fraction of all data that will be used to train the GP. Default is 1.
        save_CSV: bool, determines whether a CSV is saved when this function is called. Prevents accidental overwrite of CSVs
    
    Returns:
    --------
        Plots of obj vs BO_iter and Plots of Theta vs BO_iter
    """
    assert isinstance(obj,str)==True, "Objective function name must be a string" 
    
    fxn = "plot_obj"
#     obj_df = pd.DataFrame(data = obj_array)
    
    #Create x axis as # of bo iterations
#     bo_space = np.linspace(1,bo_iters,bo_iters)
    fig = plt.figure(figsize = (6.4,4))
#     plt.gca().axes.xaxis.set_ticklabels([]) # remove tick labels
#     plt.gca().axes.yaxis.set_ticklabels([])
    
    #Plots either 1 or multiple lines for objective function values depending on whether there are runs     
    #Loop over number of runs
    for i in range(tot_runs):
        if tot_runs > 1:
            label = "Run: "+str(i+1)      
        else:
            label = r'$log(e(\theta))$'
        #Plot data
        obj_df_run = pd.DataFrame(data = obj_array[i])
        obj_df_i = obj_df_run.loc[(abs(obj_df_run) > 1e-6).any(axis=1),0]
        bo_len = len(obj_df_i)
        bo_space = np.linspace(1,bo_len,bo_len)
        plt.step(bo_space, obj_df_i, label = label)
#         plt.step(bo_space, obj_array[i], label = label)
    
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, loc = "upper left")
    plt.tight_layout()
    #Set plot details
    plt.xlabel("BO Iterations", fontsize=16,fontweight='bold')
    plt.ylabel(r'$\mathbf{log(e(\theta))}$', fontsize=16,fontweight='bold')
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(direction="in",top=True, right=True)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.minorticks_on() # turn on minor ticks
    plt.tick_params(which="minor",direction="in",top=True, right=True)
    
#     if emulator == False:
#         plt.ylim(0,2)
#     plt.title("BO Iteration Results: ln(SSE) Metric")
#     plt.grid(True)
#     plt.legend(loc = "upper right")
       
    #Save Data to CSV
    obj_min_df = pd.DataFrame(obj_array)
    if save_CSV == True:
        path_csv = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo = None, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact, is_figure = False, normalize = normalize)
        save_csv(obj_min_df, path_csv, ext = "npy")
    
    #Save path and figure
    if save_figure == True:
        path = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo = None, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact, normalize = normalize)
        save_fig(path, ext='png', close=True, verbose=False)
    
    plt.show()
    plt.close()
    
    return 

def save_misc_data(data_array, fxn, t, obj, ep, emulator, sparse_grid, set_lengthscale, save_figure, tot_iter=1, tot_runs=1, DateTime=None, sep_fact = None, normalize = False):
    """
    Creates .npy files to save best theta GP mean and GP variance, or iter times
    Parameters
    ----------
        data_array: ndarray, array containing values of the best theta GP mean or GP variance for each iteration of all runs
        fxn: fxn associated with data array being saved
        t: int, Number of initial training points to use
        obj: string, name of objective function. Default "obj"
        ep: int or float, exploration parameter. Used for naming
        emulator: True/False, Determines if GP will model the function or the function error
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_figure: True/False, Determines whether figures will be saved
        tot_iter: int, The total number of iterations. Printed at top of job script
        tot_runs: int, The total number of times training data/ testing data is reshuffled. Printed at top of job script
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        sep_fact: float, in (0,1]. Determines fraction of all data that will be used to train the GP. Default is 1.
        GP_mean: bool, determines whether the GP mean or GP variance is being saved
    Returns
    -------
        Creates npy file storing the data
    """
#         print(Theta_array_df)
    path_csv = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo = None, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact, is_figure = False, normalize = normalize)
    save_csv(data_array, path_csv, ext = "npy")
    return

def plot_Theta(Theta_array, Theta_True, t, obj, ep, emulator, sparse_grid, set_lengthscale, save_figure, param_dict, tot_iter=1, tot_runs=1, DateTime=None, sep_fact = None, nbins = 6, save_CSV = True, normalize = False):
    """
    Plots the objective function and Theta values vs BO iteration
    
    Parameters
    ----------
        Theta_array: ndarray, (nxq): The output array containing objective function values
        Theta_True: ndarray, Used for plotting Theta Values
        t: int, Number of initial training points to use
        obj: string, name of objective function. Default "obj"
        ep: int or float, exploration parameter. Used for naming
        emulator: True/False, Determines if GP will model the function or the function error
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_figure: True/False, Determines whether figures will be saved
        param_dict: dictionary, dictionary of names of each parameter that will be plotted named by indecie w.r.t Theta_True
        tot_iter: int, The total number of iterations. Printed at top of job script
        tot_runs: int, The total number of times training data/ testing data is reshuffled. Printed at top of job script
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        sep_fact: float, in (0,1]. Determines fraction of all data that will be used to train the GP. Default is 1.
        n_bins: int, number of bins with which to plot axes. Default is 6
        save_CSV: bool, determines whether a CSV is saved when this function is called. Prevents accidental overwrite of CSVs
    
    Returns:
    --------
        Plots of obj vs BO_iter and Plots of Theta vs BO_iter
    """
    assert isinstance(obj,str)==True, "Objective function name must be a string"
    if not isinstance(Theta_array, np.ndarray):
        Theta_array = np.array(Theta_array)
    fxn = "plot_Theta"
    #Find value of q from given information
    q = len(Theta_True)
    #Create x axis as # of bo iterations
#     bo_space = np.linspace(1,bo_iters,bo_iters)
    #Set a string for exploration parameter and initial number of training points
    bo_lens = np.zeros(tot_runs)
    #Make multiple plots for each parameter
    #Loop over number of parameters
#     print(Theta_array.shape)
    for j in range(q):
        plt.figure(figsize = (6.4,4))
        
        #Loop over runs and plot
        for i in range(tot_runs):
            Theta_j_df = pd.DataFrame(data = Theta_array[i])
            #Plot more than 1 line if there are many runs
            if tot_runs > 1:
                label = r'$' + str(param_dict[j]) +'$'+ " Run: "+str(i+1)  
#                 label = r'$\theta_' +str({j+1})+"$" + " Run: "+str(i+1)         
            else:
                label = r'$' + str(param_dict[j]) +'$'
#                 label = r'$\theta_' +str({j+1})+"$"
                
            Theta_j_df_i = Theta_j_df.loc[(abs(Theta_j_df) > 1e-6).any(axis=1),j]
            bo_len = len(Theta_j_df_i)
            bo_lens[i] = bo_len
            bo_space = np.linspace(1,bo_len,bo_len)
            plt.step(bo_space, Theta_j_df_i, label = label)
#             plt.step(bo_space, Theta_array[i,:,j], label = label)
        
        #Set plot details
        
        bo_len_max = int(np.max(bo_lens))
        bo_space_long = np.linspace(1,bo_len_max,bo_len_max)
        plt.step(bo_space_long, np.repeat(Theta_True[j], bo_len_max), label = r'$' + str(param_dict[j]) + '$ True')
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, loc = "upper left")
        plt.tight_layout()
        plt.xlabel("BO Iterations",fontsize=16,fontweight='bold')
        plt.ylabel(r'$\mathbf{'+ str(param_dict[j])+ '}$',fontsize=16,fontweight='bold')
#         plt.title("BO Iteration Results: "+"$\Theta_"+str({j+1})+"$")
#         plt.grid(True)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tick_params(direction="in",top=True, right=True)
        plt.locator_params(axis='y', nbins=nbins)
        plt.locator_params(axis='x', nbins=nbins)
        plt.minorticks_on() # turn on minor ticks
        plt.tick_params(which="minor",direction="in",top=True, right=True)
        
        
        #Save CSVs
        Theta_array_df = pd.DataFrame(Theta_array.T[j])
        if save_CSV == True:
#         print(Theta_array_df)
            path_csv = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo = None, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact, is_figure = False, csv_end = "/" + str(param_dict[j]), normalize = normalize)
            save_csv(Theta_array_df, path_csv, ext = "npy")
        
        #Save path and figure
        if save_figure == True:
            path = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo = None, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact, normalize = normalize) + "/" + str(param_dict[j])
            save_fig(path, ext='png', close=True, verbose=False)
            
        plt.show()
        plt.close()

    return

def plot_Theta_min(Theta_array, Theta_True, t, obj, ep, emulator, sparse_grid, set_lengthscale, save_figure, param_dict, tot_iter=1, tot_runs=1, DateTime=None, sep_fact = None, nbins = 5, save_CSV = True, normalize = False):
    """
    Plots the objective function and best Theta values so far vs BO iteration
    
    Parameters
    ----------
        Theta_array: ndarray, (nxq): The output array containing objective function values
        Theta_True: ndarray, Used for plotting Theta Values
        t: int, Number of initial training points to use
        obj: string, name of objective function. Default "obj"
        ep: int or float, exploration parameter. Used for naming
        emulator: True/False, Determines if GP will model the function or the function error
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_figure: True/False, Determines whether figures will be saved
        param_dict: dictionary, dictionary of names of each parameter that will be plotted named by indecie w.r.t Theta_True
        tot_iter: int, The total number of iterations. Printed at top of job script
        tot_runs: int, The total number of times training data/ testing data is reshuffled. Printed at top of job script
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        sep_fact: float, in (0,1]. Determines fraction of all data that will be used to train the GP. Default is 1.
        n_bins: int, number of bins with which to plot axes. Default is 6
        save_CSV: bool, determines whether a CSV is saved when this function is called. Prevents accidental overwrite of CSVs
    
    Returns:
    --------
        Plots of obj vs BO_iter and Plots of Theta vs BO_iter
    """
    assert isinstance(obj,str)==True, "Objective function name must be a string"
    if not isinstance(Theta_array, np.ndarray):
        Theta_array = np.array(Theta_array)
    fxn = "plot_Theta_min"
    #Find value of q from given information
    q = len(Theta_True)
    #Create x axis as # of bo iterations
#     bo_space = np.linspace(1,bo_iters,bo_iters)
    #Set a string for exploration parameter and initial number of training points
    bo_lens = np.zeros(tot_runs)
    #Make multiple plots for each parameter
    #Loop over number of parameters
    for j in range(q):
        plt.figure(figsize = (6.4,4))
        
        #Loop over runs and plot
        for i in range(tot_runs):
            Theta_j_df = pd.DataFrame(data = Theta_array[i])
            #Plot more than 1 line if there are many runs
            if tot_runs > 1:
                label = r'$' + param_dict[j] +'$' + " Run: "+str(i+1)  
#                 label = r'$\theta_' +str({j+1})+"$" + " Run: "+str(i+1)         
            else:
                label = r'$' + param_dict[j] +'$'
#                 label = r'$\theta_' +str({j+1})+"$"
                
            Theta_j_df_i = Theta_j_df.loc[(abs(Theta_j_df) > 1e-6).any(axis=1),j]
#             Theta_j_df_i = Theta_j_df.loc[:,j] #Use this if I want to show all the zeros
            bo_len = len(Theta_j_df_i)
            bo_lens[i] = bo_len
            bo_space = np.linspace(1,bo_len,bo_len)
            plt.step(bo_space, Theta_j_df_i, label = label)
#             plt.step(bo_space, Theta_array[i,:,j], label = label)
        
        #Set plot details
        
        bo_len_max = int(np.max(bo_lens))
        bo_space_long = np.linspace(1,bo_len_max,bo_len_max)
        plt.step(bo_space_long, np.repeat(Theta_True[j], bo_len_max), label = r'$' + param_dict[j] + '$ True')
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, loc = "upper left")
        plt.tight_layout()
        plt.xlabel("BO Iterations",fontsize=16,fontweight='bold')
        plt.ylabel(r'$\mathbf{'+ param_dict[j]+ '}$',fontsize=16,fontweight='bold')
#         plt.ylabel(r'$\mathbf{\theta_' + str({j+1})+"}$",fontsize=16,fontweight='bold')
#         plt.title("BO Iteration Results: "+"$\Theta_"+str({j+1})+"$")
#         plt.grid(True)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tick_params(direction="in",top=True, right=True)
        plt.locator_params(axis='y', nbins=nbins)
        plt.locator_params(axis='x', nbins=nbins)
        plt.minorticks_on() # turn on minor ticks
        plt.tick_params(which="minor",direction="in",top=True, right=True)
        
        
        #Save CSVs
        Theta_array_df = pd.DataFrame(Theta_array.T[j])
#         print(Theta_array_df)
        if save_CSV == True:
            path_csv = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo = None, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact, is_figure = False, csv_end = "/" + param_dict[j], normalize = normalize)
            save_csv(Theta_array_df, path_csv, ext = "npy")
        
        #Save path and figure
        if save_figure == True:
            path = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo = None, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact, normalize = normalize) + "/" + param_dict[j]
#             print(path)
            save_fig(path, ext='png', close=True, verbose=False)
            
        plt.show()
        plt.close()

    return

#Will need to have a loop run this for each combination of theta values
def value_plotter(test_mesh, z, p_true, p_GP_opt, p_GP_best, train_p,title,title_save, obj,ep, emulator, sparse_grid, set_lengthscale, save_figure, param_names_list, Bo_iter, run = 0, tot_iter = 1, tot_runs = 1, DateTime=None, t = 100, sep_fact = None, levels = 20, save_CSV = True, normalize = False):
    '''
    Plots heat maps for 2 input GP
    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: ndarray or tensor, An NxN Array containing all points that will be plotted
        p_true: ndarray, A 2x1 containing the true input parameters
        p_GP_Opt: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
        p_GP_Best: ndarray, A 2x1 containing the input parameters predicted by the GP to have the best EI
        title: str, A string containing the title of the plot
        title_save: str, A string containing the title of the file of the plot
        obj: str, The name of the objective function. Used for saving figures
        ep: int or float, the exploration parameter
        emulator: True/False, Determines if GP will model the function or the function error
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_figure: True/False, Determines whether figures will be saved
        param_names_list: list, list of names of each parameter that will be plotted
        Bo_iter: int or None, Determines if figures are save, and if so, which iteration they are
        run, int or None, The iteration of the number of times new training points have been picked
        tot_iter: int, The total number of iterations. Printed at top of job script
        tot_runs: int, The total number of times training data/ testing data is reshuffled. Printed at top of job script
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        t: int, Number of total data points to use (from LHS or meshgrid)
        sep_fact: float, in (0,1]. Determines fraction of all data that will be used to train the GP. Default is 1.
        levels: int, Number of levels to skip when drawing contour lines. Default is 20
        save_CSV: bool, determines whether a CSV is saved when this function is called. Prevents accidental overwrite of CSVs     
    Returns
    -------
        plt.show(), A heat map of test_mesh and z
    '''
#     print(z)
    #Backtrack out number of parameters from given information
    fxn = "value_plotter"
    mesh_combo = str(param_names_list[0]) + "-" + str(param_names_list[1])
    q=len(p_true)
    xx , yy = test_mesh #NxN, NxN
    #Assert sattements
    assert isinstance(z, np.ndarray)==True or torch.is_tensor(z)==True, "The values in the heat map must be numpy arrays or torch tensors."
    assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
    assert z.shape==xx.shape, "Array z must be NxN"
    assert len(p_true) ==len(p_GP_opt)==len(p_GP_best)==q, "p_true, p_GP_opt, and p_GP_best must be qx1 for a q input GP"
#     assert isinstance(title, str)==True, "Title must be a string"
    assert len(train_p.T) >= q, "Train_p must have at least q columns"
    assert isinstance(Bo_iter,int) == True or Bo_iter == None, "Bo_iter must be an integer or None"
    assert isinstance(obj,str)==True, "Objective function name must be a string" 
    
    #Set plot details
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(direction="in",top=True, right=True)
    
    #   cmap defines the overall color within the heatmap 
    #   levels: determines the number and positions of the contour lines / regions.
    cs = plt.contourf(xx, yy,z, levels = 100, cmap = "autumn")
    ax.set_box_aspect(1)
    # plot color bar
    if np.amax(z) < 1e-1 or np.amax(z) > 1000:
        cbar = plt.colorbar(cs, format='%.2e')
    else:
        cbar = plt.colorbar(cs, format = '%2.2f')
    
    if isinstance(title, str)!=True:
        title = str(title)
        
    # plot title in color bar
    cbar.ax.set_ylabel(r'$\mathbf{' + title +'}$', fontsize=16, fontweight='bold')
#     print(p_GP_opt[0],p_GP_opt[1])

    # set font size in color bar
    cbar.ax.tick_params(labelsize=16)
#     cbar.ax.set_aspect('equal')
    # Plot equipotential line
    cs2 = plt.contour(cs, levels=cs.levels[::levels], colors='k', alpha=0.7, linestyles='dashed', linewidths=3)
    
    #Plot heatmap label
    
    if np.amax(z) < 1e-1 or np.amax(z) > 1000:
        plt.clabel(cs2, fmt='%.2e', colors='k', fontsize=16)
    else:
        plt.clabel(cs2, fmt='%2.2f', colors='k', fontsize=16)
    
    #Can only plot np arrays
    if torch.is_tensor(train_p) == True:
        train_p = train_p.numpy() 
    
    #Plots the true optimal value, the best EI value, the GP value, and all training points
    plt.scatter(p_true[0],p_true[1], color="blue", label = "True argmin" + r'$(e(\theta))$', s=100, marker = (5,1))
        
    plt.scatter(train_p[:,0],train_p[:,1], color="green",s=25, label = "Training Data", marker = "x")
    
    plt.scatter(p_GP_opt[0],p_GP_opt[1], color="white", s=90, label = "GP argmin" + r'$(e(\theta))$', marker = ".", edgecolors= "k", linewidth=0.3)
    
    plt.scatter(p_GP_best[0],p_GP_best[1], color="black", s=25, label = "GP argmax" + r'$(E(I(\theta)))$', marker = ".")
    
    #Plots grid and legend
#     plt.grid()
#     plt.legend(loc = 'upper right')
    plt.legend(fontsize=10,bbox_to_anchor=(0, 1.05, 1, 0.2),borderaxespad=0)

    #Creates axis labels and title
    plt.xlabel(r'$\mathbf{'+ param_names_list[0]+ '}$',fontsize=16,fontweight='bold')
#     plt.xlabel(r'$\mathbf{\theta_1}$',fontsize=16,fontweight='bold')
    plt.ylabel(r'$\mathbf{'+ param_names_list[1]+ '}$',fontsize=16,fontweight='bold')
#     plt.ylabel(r'$\mathbf{\theta_2}$',fontsize=16,fontweight='bold')
    plt.xlim((np.amin(xx), np.amax(xx)))
#     print((np.amin(xx), np.amax(xx)))
    plt.ylim((np.amin(yy),np.amax(yy)))   
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.minorticks_on() # turn on minor ticks
    plt.tick_params(which="minor",direction="in",top=True, right=True)
#     ax = plt.gca() #you first need to get the axis handle
#     ax.set_aspect('box') #sets the height to width ratio to 1
#     plt.gca().axes.xaxis.set_ticklabels([]) # remove tick labels
#     plt.gca().axes.yaxis.set_ticklabels([])
    fig.tight_layout()
    #Plots axes such that they are scaled the same way (eg. circles look like circles) 
    
    #Back out number of original training points for saving figures and CSVs   
    df_list = [z, p_GP_opt, p_GP_best, p_true]
    df_list_ends = [str(title_save), "GP_Min_SSE_Pred", "GP_Best_EI_Pred", "True_p"]
    
    if save_CSV == True:
        for i in range(len(df_list)):
            array_df = pd.DataFrame(df_list[i])
            path_csv = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo, Bo_iter, title_save, run, tot_iter=tot_iter, tot_runs=tot_runs, DateTime=DateTime, sep_fact = sep_fact, is_figure = False, csv_end = "/" + df_list_ends[i], normalize = normalize)
            save_csv(array_df, path_csv, ext = "npy")
    
    if tot_iter >= 1:
#         plt.title(title+" BO iter "+str(Bo_iter+1), weight='bold',fontsize=16)
        
#         t = t = str(t - Bo_iter )
#         if emulator == True:
#             t = str(t - 5*(Bo_iter) )
#         else:
#             t = str(t - Bo_iter )
        
        #Generate path and save figures     
        if save_figure == True:
            path = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo, Bo_iter, title_save, run, tot_iter=tot_iter, tot_runs=tot_runs, DateTime=DateTime, sep_fact = sep_fact, normalize = normalize)
            save_fig(path, ext='png', close=True, verbose=False)
#             print(path)
        else:
            plt.show()
    #Don't save if there's only 1 BO iteration
#     else:
#         plt.title("Heat Map of "+title, weight='bold',fontsize=16)     

    plt.close()
    
#     print(fxn)
    return 

def value_plotter_remake(test_mesh, z, p_true, p_GP_opt, p_GP_best, train_p,title,title_save, obj,ep, emulator, sparse_grid, set_lengthscale, save_figure, param_names_list, Bo_iter, run = 0, tot_iter = 1, tot_runs = 1, DateTime=None, t = 100, sep_fact = None, levels = 20, save_CSV = True, normalize = False):
    '''
    Plots heat maps for 2 input GP
    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: ndarray or tensor, An NxN Array containing all points that will be plotted
        p_true: ndarray, A 2x1 containing the true input parameters
        p_GP_Opt: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
        p_GP_Best: ndarray, A 2x1 containing the input parameters predicted by the GP to have the best EI
        title: str, A string containing the title of the plot
        title_save: str, A string containing the title of the file of the plot
        obj: str, The name of the objective function. Used for saving figures
        ep: int or float, the exploration parameter
        emulator: True/False, Determines if GP will model the function or the function error
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_figure: True/False, Determines whether figures will be saved
        param_names_list: list, list of names of each parameter that will be plotted
        Bo_iter: int or None, Determines if figures are save, and if so, which iteration they are
        run, int or None, The iteration of the number of times new training points have been picked
        tot_iter: int, The total number of iterations. Printed at top of job script
        tot_runs: int, The total number of times training data/ testing data is reshuffled. Printed at top of job script
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        t: int, Number of total data points to use (from LHS or meshgrid)
        sep_fact: float, in (0,1]. Determines fraction of all data that will be used to train the GP. Default is 1.
        levels: int, Number of levels to skip when drawing contour lines. Default is 20
        save_CSV: bool, determines whether a CSV is saved when this function is called. Prevents accidental overwrite of CSVs     
    Returns
    -------
        plt.show(), A heat map of test_mesh and z
    '''
#     print(z)
    #Backtrack out number of parameters from given information
    fxn = "value_plotter"
    mesh_combo = str(param_names_list[0]) + "-" + str(param_names_list[1])
    q=len(p_true)
    xx , yy = test_mesh #NxN, NxN
    #Assert sattements
    assert isinstance(z, np.ndarray)==True or torch.is_tensor(z)==True, "The values in the heat map must be numpy arrays or torch tensors."
    assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
    assert z.shape==xx.shape, "Array z must be NxN"
    assert len(p_true) ==len(p_GP_opt)==len(p_GP_best)==q, "p_true, p_GP_opt, and p_GP_best must be qx1 for a q input GP"
#     assert isinstance(title, str)==True, "Title must be a string"
    assert len(train_p.T) >= q, "Train_p must have at least q columns"
    assert isinstance(Bo_iter,int) == True or Bo_iter == None, "Bo_iter must be an integer or None"
    assert isinstance(obj,str)==True, "Objective function name must be a string" 
    
    #Set plot details
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(direction="in",top=True, right=True)
    
    #   cmap defines the overall color within the heatmap 
    #   levels: determines the number and positions of the contour lines / regions.
    cs = plt.contourf(xx, yy,z, levels = 100, cmap = "autumn")
    ax.set_box_aspect(1)
    # plot color bar
    if np.amax(z) < 1e-1 or np.amax(z) > 1000:
        cbar = plt.colorbar(cs, format='%.2e')
    else:
        cbar = plt.colorbar(cs, format = '%2.2f')
    
    if isinstance(title, str)!=True:
        title = str(title)
        
    # plot title in color bar
    cbar.ax.set_ylabel(r'$\mathbf{' + title +'}$', fontsize=16, fontweight='bold')
#     print(p_GP_opt[0],p_GP_opt[1])

    # set font size in color bar
    cbar.ax.tick_params(labelsize=16)
#     cbar.ax.set_aspect('equal')
    # Plot equipotential line
    cs2 = plt.contour(cs, levels=cs.levels[::levels], colors='k', alpha=0.7, linestyles='dashed', linewidths=3)
    
    #Plot heatmap label
    
    if np.amax(z) < 1e-1 or np.amax(z) > 1000:
        plt.clabel(cs2, fmt='%.2e', colors='k', fontsize=16)
    else:
        plt.clabel(cs2, fmt='%2.2f', colors='k', fontsize=16)
    
    #Can only plot np arrays
    if torch.is_tensor(train_p) == True:
        train_p = train_p.numpy() 
    
    #Plots the true optimal value, the best EI value, the GP value, and all training points
    plt.scatter(p_true[0],p_true[1], color="blue", label = "True argmin" + r'$(e(\theta))$', s=100, marker = (5,1))
        
    plt.scatter(train_p[:,0],train_p[:,1], color="green",s=25, label = "Training Data", marker = "x")
    
    plt.scatter(p_GP_opt[0],p_GP_opt[1], color="white", s=90, label = "GP argmin" + r'$(e(\theta))$', marker = ".", edgecolors= "k", linewidth=0.3)
    
    plt.scatter(p_GP_best[0],p_GP_best[1], color="black", s=25, label = "GP argmax" + r'$(E(I(\theta)))$', marker = ".")
    
    #Plots grid and legend
#     plt.grid()
#     plt.legend(loc = 'upper right')
    plt.legend(fontsize=10,bbox_to_anchor=(0, 1.05, 1, 0.2),borderaxespad=0)

    #Creates axis labels and title
    plt.xlabel(r'$\mathbf{'+ param_names_list[0]+ '}$',fontsize=16,fontweight='bold')
#     plt.xlabel(r'$\mathbf{\theta_1}$',fontsize=16,fontweight='bold')
    plt.ylabel(r'$\mathbf{'+ param_names_list[1]+ '}$',fontsize=16,fontweight='bold')
#     plt.ylabel(r'$\mathbf{\theta_2}$',fontsize=16,fontweight='bold')
    plt.xlim((np.amin(xx), np.amax(xx)))
#     print((np.amin(xx), np.amax(xx)))
    plt.ylim((np.amin(yy),np.amax(yy)))   
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.minorticks_on() # turn on minor ticks
    plt.tick_params(which="minor",direction="in",top=True, right=True)
#     ax = plt.gca() #you first need to get the axis handle
#     ax.set_aspect('box') #sets the height to width ratio to 1
#     plt.gca().axes.xaxis.set_ticklabels([]) # remove tick labels
#     plt.gca().axes.yaxis.set_ticklabels([])
    fig.tight_layout()
    #Plots axes such that they are scaled the same way (eg. circles look like circles) 
    
    #Back out number of original training points for saving figures and CSVs   
    df_list = [z, p_GP_opt, p_GP_best, p_true]
    df_list_ends = [str(title_save), "GP_Min_SSE_Pred", "GP_Best_EI_Pred", "True_p"]
    
    if save_CSV == True:
        for i in range(len(df_list)):
            array_df = pd.DataFrame(df_list[i])
            path_csv = "../" + path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo, Bo_iter, title_save, run, tot_iter=tot_iter, tot_runs=tot_runs, DateTime=DateTime, sep_fact = sep_fact, is_figure = False, csv_end = "/" + df_list_ends[i], normalize = normalize)
            save_csv(array_df, path_csv, ext = "npy")
    
    if tot_iter > 1:
#         plt.title(title+" BO iter "+str(Bo_iter+1), weight='bold',fontsize=16)
        
#         t = t = str(t - Bo_iter )
#         if emulator == True:
#             t = str(t - 5*(Bo_iter) )
#         else:
#             t = str(t - Bo_iter )
        
        #Generate path and save figures     
        if save_figure == True:
            path = "../" + path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, mesh_combo, Bo_iter, title_save, run, tot_iter=tot_iter, tot_runs=tot_runs, DateTime=DateTime, sep_fact = sep_fact, normalize = normalize)
            save_fig(path, ext='png', close=True, verbose=False)
#             print(path)
        else:
            plt.show()
    #Don't save if there's only 1 BO iteration
#     else:
#         plt.title("Heat Map of "+title, weight='bold',fontsize=16)     

    plt.close()
    
#     print(fxn)
    return 



def plot_3GP_performance(X_space, Y_sim, GP_mean, GP_stdev, Theta, Xexp, train_p = None, train_y = None, test_p = None, test_y = None, verbose = True):
    """
    """
#     if verbose == True:
#         print("GP Mean",GP_mean)
#         print("GP Stdev",GP_stdev)
#         print("SSE",sum(GP_mean-Y_sim)**2)
#         plt.close()
    fig, ax = plt.subplots()
    
    ax.plot(X_space, GP_mean, lw=2, label="GP_mean")
    ax.plot(X_space, Y_sim, color = "green", label = "Y_sim")
#     if train_p != None:
#         ax.scatter(train_p[:,-1], train_y, color = "black", label = "Training")
    if test_p != None:
        ax.scatter(Xexp, test_y, color = "red", label = "Testing")
    
    ax.fill_between(
        X_space,
        GP_mean - 1.96 * GP_stdev,
        GP_mean + 1.96 * GP_stdev,
        alpha=0.3
    )
#     ax.set_title("GP Mean + confidence interval at"+ str(Theta))
    ax.set_xlabel("Xexp")
    ax.set_ylabel("Function Value")
    ax.legend()
    
    return plt.show()



# def plot_2GP_performance(X_space, Y_sim, y_mean, GP_stdev, Theta, Xexp, train_p = None, train_y = None, test_p = None, test_y = None, verbose = True):
#     """
#     """
# #     if verbose == True:
# #         print("GP Mean",GP_mean)
# #         print("GP Stdev",GP_stdev)
# #         print("SSE",sum(GP_mean-Y_sim)**2)
# #         plt.close()
#     fig, ax = plt.subplots()
    
#     ax.plot(X_space, GP_mean, lw=2, label="GP_mean")
#     ax.plot(X_space, Y_sim, color = "green", label = "Y_sim")
# #     if train_p != None:
# #         ax.scatter(train_p[:,-1], train_y, color = "black", label = "Training")
#     if test_p != None:
#         ax.scatter(Xexp, test_y, color = "red", label = "Testing")
    
#     ax.fill_between(
#         X_space,
#         GP_mean - 1.96 * GP_stdev,
#         GP_mean + 1.96 * GP_stdev,
#         alpha=0.3
#     )
# #     ax.set_title("GP Mean + confidence interval at"+ str(Theta))
#     ax.set_xlabel("Xexp")
#     ax.set_ylabel("Function Value")
#     ax.legend()
    
#     return plt.show()