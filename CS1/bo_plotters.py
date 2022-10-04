from matplotlib import pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import pandas as pd
import os
import matplotlib.pyplot as plt


# rc_fonts = {
#     "text.usetex": True,
#     'text.latex.preview': True, # Gives correct legend alignment.
#     'mathtext.default': 'regular',
#     'text.latex.preamble': [r"""\usepackage{bm}"""],
# }
# matplotlib.rcParams.update(rc_fonts)


def save_csv(df, path, ext='csv', verbose=False):
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

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)
    
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
        
def path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, bo_iter= None, title_save = None, run = None, tot_iter=1, tot_runs=1, DateTime = None, sep_fact = None, is_figure = True, csv_end = None):
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
        bo_iter: int, integer, number of the specific BO iterations
        title_save: str or None,  A string containing the title of the file of the plot
        run, int or None, The iteration of the number of times new training points have been picked
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
            
    fxn_dict = {"plot_obj":"/SSE_Conv" , "plot_Theta":"/Theta_Conv" , "plot_obj_abs_min":"/Min_SSE_Conv" , "plot_org_train":"/org_TP", "value_plotter":"/"+ str(title_save), "plot_sep_fact_min":"/Sep_Analysis"}
    plot = fxn_dict[fxn]
    
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
        path_org = "../"+DateTime #Will send to the Datetime folder outside of CS1
    else:
        path_org = "Test_Figs"
#         path_org = "Test_Figs"+"/Sep_Analysis2"+"/Figures"
    if is_figure == True:
        path_org = path_org + "/Figures"
    else:
        path_org = path_org + "/CSV_Data"
        
    path_end = Emulator + method + org_TP_str + obj_str + exp_str + len_scl + sep_fact_str + run_str+ plot + Bo_itr_str   
    
    if fxn in ["value_plotter", "plot_org_train"]:
        path = path_org + path_end      

    else:
        path = path_org + "/Convergence_Figs" + path_end 
        
    if csv_end is not None:
        path = path + csv_end
       
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

def plot_org_train(test_mesh,train_p, test_p, p_true, emulator, sparse_grid, obj, ep, len_scl, run, save_figure, tot_iter=1, tot_runs=1, DateTime=None, verbose = True, sep_fact = None):
    '''
    Plots original training data with true value
    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        train_p: tensor or ndarray, The training parameter space data
        test_p: tensor or ndarray, The training parameter space data
        p_true: ndarray, A 2x1 containing the true input parameters
        emulator: True/False, Determines if GP will model the function or the function error
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        ep: float, float,int,tensor,ndarray (1 value) The exploration bias parameter
        len_scl: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        run, int or None, The iteration of the number of times new training points have been picked
        save_figure: True/False, Determines whether figures will be saved
     
    Returns
    -------
        plt.show(), A plot of the original training data points and the true value
    '''
    fxn = "plot_org_train"
#     t = int(len(train_p))
    t = int(len(train_p)) + int(len(test_p))
    #xx and yy are the values of the parameter sets
    xx,yy = test_mesh
    if emulator == False:
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
        
        #plot training data and true values
        plt.scatter(train_p[:,0],train_p[:,1], color="green",s=50, label = "Training Data", marker = "x")
        try:
            plt.scatter(test_p[:,0],test_p[:,1], color="red",s=25, label = "Testing Data", marker = "x")
        except:
            plt.scatter(test_p[0],test_p[1], color="red",s=25, label = "Testing Data", marker = "x")
        plt.scatter(p_true[0],p_true[1], color="blue", label = "True argmin" + r'$(e(\theta))$', s=100, marker = (5,1))
        #Set plot details
        plt.legend(fontsize=10,bbox_to_anchor=(0, 1.05, 1, 0.2),borderaxespad=0)
#         plt.legend(loc = "best")
        plt.xlabel(r'$\mathbf{\theta_1}$', fontsize=16, fontweight='bold')
        plt.ylabel(r'$\mathbf{\theta_2}$', fontsize=16, fontweight='bold')
        #Set axis limits based on the maximum and minimum of the parameter search space
        plt.xlim((np.amin(xx), np.amax(xx)))
        plt.ylim((np.amin(yy), np.amax(yy)))
#         plt.title("Starting Training Data")
#         plt.grid(True)
        
    else:
        x_space = np.array([-2,-1,0,1,2])
        len_x = len(x_space)
        p_true_3D = np.repeat(p_true,len_x).reshape(-1,len_x).T
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
        ax.scatter(train_p[:,0], train_p[:,1], train_p[:,2], color = "green", s=50, label = "Training Data", marker='o')
        try:
            ax.scatter(test_p[:,0],test_p[:,1], test_p[:,2], color="red", s=25, label = "Testing Data", marker = "x")
        except:
            ax.scatter(test_p[0],test_p[1], test_p[2], color="red",s=25, label = "Testing Data", marker = "x")
            
        ax.scatter(p_true_3D_full[:,0], p_true_3D_full[:,1], p_true_3D_full[:,2], color="blue", label = "True argmin" + r'$(e(\theta))$', 
                    s=100, marker = (5,1))
        
        plt.legend(fontsize=10,bbox_to_anchor=(0, 1.05, 1, 0.2),borderaxespad=0)
        ax.set_xlabel(r'$\mathbf{\theta_1}$', fontsize=16, fontweight='bold')
        ax.set_ylabel(r'$\mathbf{\theta_2}$', fontsize=16, fontweight='bold')
        ax.set_zlabel('X-Value', fontsize=16, fontweight='bold')
#         ax.legend(loc = "best")
#         plt.legend(fontsize=10,bbox_to_anchor=(0, 1.05, 1, 0.2),borderaxespad=0)
#         ax.legend(fontsize=10,bbox_to_anchor=(1.02, 0.3),borderaxespad=0)
        ax.set_xlim((np.amin(xx), np.amax(xx)))
        ax.set_ylim((np.amin(yy), np.amax(yy)))
        ax.grid(False)
#         ax.set_zlim((np.amin(z), np.amax(zz)))
#         plt.grid(True)
#         plt.title("Starting Training Data")
    
    df_list_ends = ["test_theta", "train_theta"]
    df_list = [test_p, train_p]
    
    for i in range(len(df_list_ends)):
        array_df = pd.DataFrame(df_list[i])
        path_csv = path_name(emulator, ep, sparse_grid, fxn, len_scl, t, obj, bo_iter=None, title_save = None, run = run, tot_iter=tot_iter, tot_runs=tot_runs, DateTime=DateTime, sep_fact = sep_fact, is_figure = False, csv_end = "/" + df_list_ends[i])
    #How to save more efficiently without hardcoding number of columns?
        save_csv(array_df, path_csv, ext = "csv")
    
    
    if save_figure == True:
        path = path_name(emulator, ep, sparse_grid, fxn, len_scl, t, obj, bo_iter=None, title_save = None, run = run, tot_iter=tot_iter, tot_runs=tot_runs, DateTime=DateTime, sep_fact = sep_fact)
        save_fig(path, ext='png', close=True, verbose=False)  
    
    if save_figure == False:
#     if verbose == True and save_figure == False:
        plt.show()
        
    plt.close()
    
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
    assert len(x_exp) == len(y_exp) == len(y_GP), "Xexp, Yexp, and Y_GP must be the same length"
    
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
    
    plt.scatter(x_exp, y_exp, label = "y $\theta_{true}$", color = "orange")
    plt.scatter(x_exp, y_GP)
    plt.plot(x_line, y_true, color = "orange")
    plt.plot(x_line, y_GP_long, "--", label = "y $\theta_{GP}$")
    
    #Set plot details
#     plt.grid(True)
#     plt.legend(loc = "best")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, loc = "upper left")
    plt.tight_layout()
    plt.xlabel('X Value',fontsize=16, fontweight='bold')
    plt.ylabel('Y Value', fontsize=16, fontweight='bold')
#     plt.title("Plot of "+title, weight='bold',fontsize = 16)
    
    return plt.show()

def plot_obj_abs_min(obj_abs_min, emulator, ep, sparse_grid, set_lengthscale, t, obj, save_figure, tot_iter=1, tot_runs=1,DateTime=None, sep_fact = None):
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
            label = "Minimum "+ "r'$log(e(\theta))$" +" Value Found"
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
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.minorticks_on() # turn on minor ticks
    plt.tick_params(which="minor",direction="in",top=True, right=True)
#     plt.gca().axes.xaxis.set_ticklabels([]) # remove tick labels
#     plt.gca().axes.yaxis.set_ticklabels([])
#     plt.title("BO Iteration Results: Lowest Overall ln(SSE)")
#     plt.grid(True)
    
    #Save CSVs - How to save column names as run #s automatically?
    obj_abs_min_df = pd.DataFrame(obj_abs_min)
    path_csv = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact, is_figure = False)
    save_csv(obj_abs_min_df, path_csv, ext = "csv")
        
    #Save figure path
    if save_figure == True:
        path = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact)
        save_fig(path, ext='png', close=True, verbose=False)
    
    plt.show()
    
    return 

def plot_sep_fact_min(bo_iters, obj_abs_min, emulator, ep, sparse_grid, set_lengthscale, t, obj, save_figure, tot_iter=1 ,DateTime=None, sep_list = None):
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
            label = "Minimum " + "r'$log(e(\theta))$'" +" Value Found"
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
    
    plt.show()
    
    #Save CSVs
    obj_abs_min_df = pd.DataFrame(obj_abs_min)
    path_csv = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact, is_figure = False, csv_end = "Min_SSE_Conv_Sep_Fact")
    save_csv(obj_abs_min_df, path_csv, ext = "csv")
    
    #Save figure path
    if save_figure == True:
        path = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = None)
        save_fig(path, ext='png', close=True, verbose=False)
    
    return 

def plot_obj(obj_array, t, obj, ep, emulator, sparse_grid, set_lengthscale, save_figure, tot_iter=1, tot_runs=1, DateTime=None, sep_fact = None):
    """
    Plots the objective function and Theta values vs BO iteration
    
    Parameters
    ----------
        obj_array: ndarry, (nx1): The output array containing objective function values
        Theta_array: ndarray, (nxq): The output array containing objective function values
        Theta_True: ndarray, Used for plotting Theta Values
        t: int, Number of initial training points to use
        obj: string, name of objective function. Default "obj"
        ep: int or float, exploration parameter. Used for naming
        emulator: True/False, Determines if GP will model the function or the function error
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_figure: True/False, Determines whether figures will be saved
        runs: int, The number of times to choose new training points
    
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
    path_csv = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact, is_figure = False)
    save_csv(obj_min_df, path_csv, ext = "csv")
    
    #Save path and figure
    if save_figure == True:
        path = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact)
        save_fig(path, ext='png', close=True, verbose=False)
    
    plt.show()
    
    return 

def plot_Theta(Theta_array, Theta_True, t, bo_iters, obj, ep, emulator, sparse_grid, set_lengthscale, save_figure,tot_iter=1, tot_runs=1, DateTime=None, sep_fact = None):
    """
    Plots the objective function and Theta values vs BO iteration
    
    Parameters
    ----------
        obj_array: ndarry, (nx1): The output array containing objective function values
        Theta_array: ndarray, (nxq): The output array containing objective function values
        Theta_True: ndarray, Used for plotting Theta Values
        t: int, Number of initial training points to use
        bo_iters: integer, number of BO iterations
        obj: string, name of objective function. Default "obj"
        ep: int or float, exploration parameter. Used for naming
        emulator: True/False, Determines if GP will model the function or the function error
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_figure: True/False, Determines whether figures will be saved
        runs: int, The number of times to choose new training points
    
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
    for j in range(q):
        plt.figure(figsize = (6.4,4))
        
        #Loop over runs and plot
        for i in range(tot_runs):
            Theta_j_df = pd.DataFrame(data = Theta_array[i])
            #Plot more than 1 line if there are many runs
            if tot_runs > 1:
                label = r'$\theta_' +str({j+1})+"$" + " Run: "+str(i+1)         
            else:
                label = r'$\theta_' +str({j+1})+"$"
                
            Theta_j_df_i = Theta_j_df.loc[(abs(Theta_j_df) > 1e-6).any(axis=1),j]
            bo_len = len(Theta_j_df_i)
            bo_lens[i] = bo_len
            bo_space = np.linspace(1,bo_len,bo_len)
            plt.step(bo_space, Theta_j_df_i, label = label)
#             plt.step(bo_space, Theta_array[i,:,j], label = label)
        
        #Set plot details
        
        bo_len_max = int(np.max(bo_lens))
        bo_space_long = np.linspace(1,bo_len_max,bo_len_max)
        plt.step(bo_space_long, np.repeat(Theta_True[j], bo_len_max), label = r'$\theta_{true,'+str(j+1)+'}$')
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, loc = "upper left")
        plt.tight_layout()
        plt.xlabel("BO Iterations",fontsize=16,fontweight='bold')
        plt.ylabel(r'$\mathbf{\theta_' + str({j+1})+"}$",fontsize=16,fontweight='bold')
#         plt.title("BO Iteration Results: "+"$\Theta_"+str({j+1})+"$")
#         plt.grid(True)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tick_params(direction="in",top=True, right=True)
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=5)
        plt.minorticks_on() # turn on minor ticks
        plt.tick_params(which="minor",direction="in",top=True, right=True)
        
        
        #Save CSVs
        Theta_array_df = pd.DataFrame(Theta_array.T[j])
#         print(Theta_array_df)
        path_csv = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact, is_figure = False, csv_end = "/Theta_Conv_" +str(j+1))
        save_csv(Theta_array_df, path_csv, ext = "csv")
        
        #Save path and figure
        if save_figure == True:
            path = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, bo_iter=None, title_save = None, run = None, tot_iter=tot_iter, tot_runs=tot_runs,DateTime=DateTime, sep_fact = sep_fact) + "_" + str(j+1)
            save_fig(path, ext='png', close=True, verbose=False)
            
        plt.show() 
    #Save CSVs

    return


def value_plotter(test_mesh, z, p_true, p_GP_opt, p_GP_best, train_p,title,title_save, obj,ep, emulator, sparse_grid, set_lengthscale, save_figure, Bo_iter, run = 0, tot_iter = 1, tot_runs = 1, DateTime=None, t = 100, sep_fact = None):
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
        Bo_iter: int or None, Determines if figures are save, and if so, which iteration they are
        run, int or None, The iteration of the number of times new training points have been picked
     
    Returns
    -------
        plt.show(), A heat map of test_mesh and z
    '''
    #Backtrack out number of parameters from given information
    fxn = "value_plotter"
    q=len(p_true)
    xx , yy = test_mesh #NxN, NxN
    #Assert sattements
    assert isinstance(z, np.ndarray)==True or torch.is_tensor(z)==True, "The values in the heat map must be numpy arrays or torch tensors."
    assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
    assert z.shape==xx.shape, "Array z must be NxN"
    assert len(p_true) ==len(p_GP_opt)==len(p_GP_best)==q, "p_true, p_GP_opt, and p_GP_best must be qx1 for a q input GP"
    assert isinstance(title, str)==True, "Title must be a string"
    assert len(train_p.T) >= q, "Train_p must have at least q columns"
    assert isinstance(Bo_iter,int) == True or Bo_iter == None, "Bo_iter must be an integer or None"
    assert isinstance(obj,str)==True, "Objective function name must be a string" 
    
    #Set plot details
    plt.figure(figsize=(6,6))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(direction="in",top=True, right=True)
    
    #   cmap defines the overall color within the heatmap 
    #   levels: determines the number and positions of the contour lines / regions.
    cs = plt.contourf(xx, yy,z, levels = 100, cmap = "autumn")
    
    # plot color bar
    if np.amax(z) < 1e-1:
        cbar = plt.colorbar(cs, format='%.2e')
    else:
        cbar = plt.colorbar(cs, format = '%2.2f')
    
    # plot title in color bar
    cbar.ax.set_ylabel(r'$\mathbf{' + title +'}$', fontsize=16, fontweight='bold')
#     print(p_GP_opt[0],p_GP_opt[1])

    # set font size in color bar
    cbar.ax.tick_params(labelsize=16)
    
    # Plot equipotential line
    cs2 = plt.contour(cs, levels=cs.levels[::20], colors='k', alpha=0.7, linestyles='dashed', linewidths=3)
    
    #Plot heatmap label
    
    if np.amax(z) < 1e-1:
        plt.clabel(cs2, fmt='%.2e', colors='k', fontsize=16)
    else:
        plt.clabel(cs2, fmt='%2.2f', colors='k', fontsize=16)
    
    #Can only plot np arrays
    if torch.is_tensor(train_p) == True:
        train_p = train_p.numpy()
        
    #Plots axes such that they are scaled the same way (eg. circles look like circles)
    plt.axis('square')    
    
    #Plots the true optimal value, the best EI value, the GP value, and all training points
    plt.scatter(p_true[0],p_true[1], color="blue", label = "True argmin" + r'$(e(\theta))$', s=100, marker = (5,1))
        
    plt.scatter(train_p[:,0],train_p[:,1], color="green",s=25, label = "Training Data", marker = "x")
    
    plt.scatter(p_GP_opt[0],p_GP_opt[1], color="white", s=50, label = "GP argmin" + r'$(e(\theta))$', marker = ".", edgecolors= "k", linewidth=0.3)
    
    plt.scatter(p_GP_best[0],p_GP_best[1], color="black", s=25, label = "GP argmax" + r'$(E(I(\theta)))$', marker = ".")
    
    #Plots grid and legend
#     plt.grid()
#     plt.legend(loc = 'upper right')
    plt.legend(fontsize=10,bbox_to_anchor=(0, 1.05, 1, 0.2),borderaxespad=0)

    #Creates axis labels and title
    plt.xlabel(r'$\mathbf{\theta_1}$',fontsize=16,fontweight='bold')
    plt.ylabel(r'$\mathbf{\theta_2}$',fontsize=16,fontweight='bold')
    plt.xlim((np.amin(xx), np.amax(xx)))
    plt.ylim((np.amin(yy),np.amax(yy)))   
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.minorticks_on() # turn on minor ticks
    plt.tick_params(which="minor",direction="in",top=True, right=True)
#     plt.gca().axes.xaxis.set_ticklabels([]) # remove tick labels
#     plt.gca().axes.yaxis.set_ticklabels([])
    
    #Back out number of original training points for saving figures and CSVs   
    df_list = [z, p_GP_opt, p_GP_best]
    df_list_ends = [str(title_save), "GP_Min_SSE_Pred", "GP_Best_EI_Pred"]
    
    for i in range(len(df_list)):
        array_df = pd.DataFrame(df_list[i])
        path_csv = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, Bo_iter, title_save, run, tot_iter=tot_iter, tot_runs=tot_runs, DateTime=DateTime, sep_fact = sep_fact, is_figure = False, csv_end = "/" + df_list_ends[i])
        save_csv(array_df, path_csv, ext = "csv")
    
    if tot_iter > 1:
#         plt.title(title+" BO iter "+str(Bo_iter+1), weight='bold',fontsize=16)
        
#         t = t = str(t - Bo_iter )
#         if emulator == True:
#             t = str(t - 5*(Bo_iter) )
#         else:
#             t = str(t - Bo_iter )
        
        #Generate path and save figures     
        if save_figure == True:
            path = path_name(emulator, ep, sparse_grid, fxn, set_lengthscale, t, obj, Bo_iter, title_save, run, tot_iter=tot_iter, tot_runs=tot_runs, DateTime=DateTime, sep_fact = sep_fact)
            save_fig(path, ext='png', close=True, verbose=False)
    #Don't save if there's only 1 BO iteration
#     else:
#         plt.title("Heat Map of "+title, weight='bold',fontsize=16)     
           
    return plt.show()