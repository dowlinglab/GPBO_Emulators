# project.py
import signac
from pathlib import Path
import os
import json
import flow
from flow import FlowProject, directives

#Import dependencies
import numpy as np
import bo_methods_lib
import templates.ndcrc
from bo_methods_lib.bo_methods_lib.GPBO_Classes_New import Method_name_enum, Ep_enum, Kernel_enum, Gen_meth_enum
from bo_methods_lib.bo_methods_lib.GPBO_Classes_New import GPBO_Methods, Exploration_Bias, CaseStudyParameters, GPBO_Driver
from bo_methods_lib.bo_methods_lib.GPBO_Class_fxns import simulator_helper_test_fxns
import pickle
import gzip

#Ignore warnings caused by "nan" values
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)

#Ignore warning from scikit learn hp tuning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

class Project(FlowProject):
    def __init__(self):
        super().__init__()
        current_path = Path(os.getcwd()).absolute()

@Project.label
def results_computed(job):
    #Write script that checks whether the .pickle file is there
    return job.isfile("BO_Results.gz")

@Project.post(results_computed)
@Project.operation(with_job = True)
def run_ep_or_sf_exp(job):
    #Define method, ep_enum classes, indecies to consider, and kernel
    meth_name = Method_name_enum(job.sp.meth_name_val)
    method = GPBO_Methods(meth_name)
    ep_enum = Ep_enum(job.sp.ep_enum_val)
    kernel = Kernel_enum(job.sp.kernel_enum_val)
    lenscl = job.sp.lenscl
    try:
        lenscl = json.loads(lenscl)
    except:
        lenscl = job.sp.lenscl
    
    #Define Simulator Class (Export your Simulator Object Here)
    simulator = simulator_helper_test_fxns(job.sp.cs_name_val, job.sp.noise_mean, job.sp.noise_std, job.sp.seed)

    #Generate Exp Data (OR Add your own experimental data as a Data class object)
    set_seed = 1 #Set set_seed to 1 for data generation
    gen_meth_x = Gen_meth_enum(job.sp.gen_meth_x)
    exp_data = simulator.gen_exp_data(job.sp.num_x_data, gen_meth_x, set_seed)

    #Set simulator noise_std artifically as 5% of y_exp mean (So that noise for gp will be set rather than trained)
    simulator.noise_std = np.mean(exp_data.y_vals)*0.05
    
    #Create Exploration Bias Class
    if ep_enum.value == 1:
        #Constant value stays constant
        ep_bias = Exploration_Bias(job.sp.ep0, None, ep_enum, None, None, None, None, None, None, None)
    elif ep_enum.value == 2:
        #For decay method, decay from mixed to full exploitation (alpha of 0.5) for this example
        ep_bias = Exploration_Bias(job.sp.ep0, None, ep_enum, None, job.sp.bo_iter_tot, None, 0.5, None, None, None)
    elif ep_enum.value == 3:
        #Set ep multiplier to 1.5 as recommended in Boyle
        ep_bias = Exploration_Bias(job.sp.ep0, None, ep_enum, None, None, 1.5, None, None, None, None)
    elif ep_enum.value == 4:
        #Jasrasaria method will take care of itself
        ep_bias = Exploration_Bias(None, None, ep_enum, None, None, None, None, None, None, None)
    else:
        raise Warning("Ep_enum value must be between 1 and 4!")
        
    #Generate Sim (Training) Data (OR Add your own training data here as a Data class object)
    num_theta_data = len(simulator.indeces_to_consider)*job.sp.num_theta_multiplier
    gen_meth_theta = Gen_meth_enum(job.sp.gen_meth_theta)
    #Note at present, training data is always the same between jobs since we set the data generation seed to 1
    sim_data = simulator.gen_sim_data(num_theta_data, job.sp.num_x_data, gen_meth_theta, gen_meth_x, job.sp.sep_fact, set_seed, False)
    
    #Gen sse_sim_data and sse_sim_val_data
    sim_sse_data = simulator.sim_data_to_sse_sim_data(method, sim_data, exp_data, job.sp.sep_fact, False)
    
    #Generate validation data if applicable. This is only useful for small (<5 Params + State Points). Otherwise this takes up too much memory
    if job.sp.num_val_pts > 0:
        gen_meth_theta_val = Gen_meth_enum(job.sp.gen_meth_theta_val) #input is an integer (1 or 2)
        val_data = simulator.gen_sim_data(job.sp.num_val_pts, job.sp.num_x_data, gen_meth_theta_val, gen_meth_x, job.sp.sep_fact, set_seed, True)
        val_sse_data = simulator.sim_data_to_sse_sim_data(method, val_data, exp_data, job.sp.sep_fact, True)        
    #Set validation data to None if not generating it
    else:
        val_data = None
        val_sse_data = None
        gen_meth_theta_val = job.sp.gen_meth_theta_val #Value is None
                       
    #Define cs_name and cs_params class
    #Signac saves all BO_Results in different folders, so they can have the same name
    cs_name = "BO_Results"
    cs_params = CaseStudyParameters(cs_name, job.sp.ep0, job.sp.sep_fact, job.sp.normalize, kernel, lenscl, job.sp.outputscl,
                                    job.sp.retrain_GP, job.sp.reoptimize_obj, job.sp.gen_heat_map_data, job.sp.bo_iter_tot,
                                    job.sp.bo_runs_in_job, job.sp.save_data, None, job.sp.seed, job.sp.ei_tol, job.sp.obj_tol)
    #Initialize driver class
    driver = GPBO_Driver(cs_params, method, simulator, exp_data, sim_data, sim_sse_data, val_data, val_sse_data, None, ep_bias, gen_meth_theta)
    #Get results
    restart_bo_results = driver.run_bo_restarts()
    
    #Save results in a .gz file in the job directory
    #Set path
    savepath = job.fn(cs_name + ".gz")
    #Open the file
    fileObj = gzip.open(savepath, 'wb', compresslevel = 1)
    #Turn this class into a pickled object and save to the file
    pickled_results = pickle.dump(restart_bo_results, fileObj)
    # Close the file
    fileObj.close()

if __name__ == "__main__":
    Project().main()