import numpy as np
from bo_methods_lib.bo_methods_lib.GPBO_Classes_New import * #Fix this later
from bo_methods_lib.bo_methods_lib.GPBO_Class_fxns import * #Fix this later
from bo_methods_lib.bo_methods_lib.analyze_data import * #Fix this later
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import * #Fix this later
from matplotlib import pyplot as plt

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

#Ignore warning from scikit learn hp tuning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

noise_mean = 0
seed = 1
retrain_GP = 25
normalize = True

cs_list = [1,2,3,10,11,12,13,14,15,16,17]
num_x =   [5,5,5, 5,10,10,10, 5,10, 6,12]

num_stds_ax = np.linspace(0,10,101)
norm_vals = (norm.cdf(num_stds_ax) - norm.cdf(-num_stds_ax))*100

fig = plt.figure()
cmap = plt.get_cmap('rainbow', len(cs_list))
#Make Pd df for hyperparameters and pct_ins

for i in range(len(cs_list)):
    #Make Simulator and Training Data
    cs_name_val = cs_list[i]
    
    #Define method, ep_enum classes, indecies to consider, and kernel
    meth_name = Method_name_enum(3)
    method = GPBO_Methods(meth_name)
    gen_meth_theta = Gen_meth_enum(1)
    gen_meth_x = Gen_meth_enum(2)
    num_x_data = num_x[i]

    #Define Simulator Class (Export your Simulator Object Here)
    simulator = simulator_helper_test_fxns(cs_name_val, noise_mean, None, seed)
    num_theta_data = len(simulator.indeces_to_consider)*10
    #Generate Exp Data (OR Add your own experimental data as a Data class object)
    set_seed = 1 #Set set_seed to 1 for data generation
    gen_meth_x = Gen_meth_enum(gen_meth_x)
    exp_data = simulator.gen_exp_data(num_x_data, gen_meth_x, set_seed)
    #Set simulator noise_std artifically as 5% of y_exp mean (So that noise will be set rather than trained)
    simulator.noise_std = np.abs(np.mean(exp_data.y_vals))*0.05
    #Note at present, training data is always the same between jobs since we set the data generation seed to 1
    sim_data = simulator.gen_sim_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, 1.0, seed, False)
    val_data = simulator.gen_sim_data(15, 15, Gen_meth_enum(1), Gen_meth_enum(1), 1.0, seed + 1, False)
    
    all_gp_data = sim_data
    all_val_data = val_data
    
    #Make Emulator
    noise_std = simulator.noise_std #Yexp_std is exactly the noise_std of the GP Kernel
    gp_object = Type_2_GP_Emulator(all_gp_data, all_val_data, None, None, None, Kernel_enum(1), None, noise_std, None, 
                                    retrain_GP, seed, normalize, None, None, None, None)
    #Choose training data
    train_data, test_data = gp_object.set_train_test_data(1.0, seed)
    #Train GP
    new_gp_model = gp_object.set_gp_model()
    gp_object.train_gp(new_gp_model)
    misc_gp_mean, misc_var_return = gp_object.eval_gp_mean_var_val(covar = False)

    #For Each Point, calculate the std multiplier that puts us in the prediction interval
    #Calculate num_stds = (y-mu)/sigma
    num_stds = sorted(abs((all_val_data.y_vals-misc_gp_mean)/np.sqrt(abs(misc_var_return))).tolist(), reverse = True)
    
    #Calculate % in side prediction interval at each point
    pct_ins = np.array([np.sum(num_stds < num_std) for num_std in num_stds_ax])/len(num_stds)*100
    plt.plot(num_stds_ax, pct_ins, color = cmap(i), alpha=0.7, label = "CS" + str(cs_name_val))
    
plt.plot(num_stds_ax, norm_vals, color='black', alpha=0.7, label = "Std Norm")
fig.legend(loc= "upper right", bbox_to_anchor=(-0.02, 1), 
                       borderaxespad=0)
plt.xlabel('Number of Std. Deviations')
plt.ylabel('% Inside Prediction Interval')
plt.title('PIPI With ' + str(retrain_GP) + " GP Retrains")
plt.grid(True)
plt.tight_layout()
plt.savefig("Results/PIPI_Retrain_"+str(retrain_GP) +".png",  bbox_inches='tight')
plt.close()