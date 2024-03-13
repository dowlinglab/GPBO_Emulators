import numpy as np
from bo_methods_lib.bo_methods_lib.GPBO_Classes_New import * #Fix this later
from bo_methods_lib.bo_methods_lib.GPBO_Class_fxns import * #Fix this later
from bo_methods_lib.bo_methods_lib.analyze_data import * #Fix this later
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import * #Fix this later
from matplotlib import pyplot as plt
from textwrap import fill

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

#Ignore warning from scikit learn hp tuning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def kernel_info(kernel):
    outputscl_final = float(kernel.kernels[0].variance.numpy())
    lenscl_final = kernel.kernels[0].lengthscales.numpy()
    noise_final = float(kernel.kernels[1].variance.numpy())

    if isinstance(kernel.kernels[0], gpflow.kernels.RBF):
        kern_type = "RBF"
    elif isinstance(kernel.kernels[0], gpflow.kernels.Matern32):
        kern_type = "Mat32"
    elif isinstance(kernel.kernels[0], gpflow.kernels.Matern52):
        kern_type = "Mat52"
    else:
        print(type(kernel))

    if isinstance(lenscl_final, np.ndarray):
        lenscl_str = '[' + ', '.join('{:.2g}'.format(x) for x in lenscl_final) + ']'
    else:
        lenscl_str = f"{lenscl_final:.2g}"
    
    info_str = "tau=" +'{:.3g}'.format(outputscl_final) + ", "+ str(kern_type) +"="+ lenscl_str + ", noise="'{:.2g}'.format(noise_final)
    return info_str

def perform_LOO(all_data, gp_object):
    n_samples = len(all_data.y_vals)

    predict_mean = []
    predict_std = []
    train_mean = []
    train_std = []
    seed = gp_object.seed
    hps_org = gp_object.trained_hyperparams

    #For each sample in the full set
    for i in range(n_samples):
        #Get sample Data
        t_leave_one = np.atleast_2d(all_data.theta_vals[i])
        X_leave_one = np.atleast_2d(all_data.x_vals[i])
        y_leave_one = np.atleast_1d(all_data.y_vals[i])
        t_rest = np.delete(all_data.theta_vals,i,axis=0)
        X_rest = np.delete(all_data.x_vals,i,axis=0)
        y_rest = np.delete(all_data.y_vals,i,axis=0)
        loo_data = Data(t_leave_one, X_leave_one, y_leave_one, None, None, None, None, None, 
                                all_data.bounds_theta, all_data.bounds_x, 1.0, seed)
        
        loo_data_train = Data(t_rest, X_rest, y_rest, None, None, None, None, None, 
                                all_data.bounds_theta, all_data.bounds_x, 1.0, seed)
        
        #Create GP object based on the above
        gp_new = Type_2_GP_Emulator(loo_data_train, loo_data, None, loo_data_train, None, Kernel_enum(1), None, noise_std, None, 
                                    1, seed, normalize, None, None, None, None)
        gp_new.scalerX = gp_object.scalerX
        gp_new.scalerY = gp_object.scalerY
        
        #Create GP Model Based On Past Hyperparamaters
        loo_ft = gp_new.featurize_data(loo_data)
        loo_ft_trn = gp_new.featurize_data(loo_data_train)
        looft_scl = gp_new.scalerX.transform(loo_ft_trn)
        scl_looy = gp_new.scalerY.transform(y_rest.reshape(-1,1))
        gp_new.train_data_init = loo_ft_trn
        mat_kern = gpflow.kernels.Matern52(variance = hps_org[2], lengthscales=hps_org[0])
        noise_kern = gpflow.kernels.White(variance=hps_org[1])
        lik_noise_var = np.maximum(1.000001e-6, float(gp_object.fit_gp_model.likelihood.variance.numpy()))
        kernel = mat_kern + noise_kern
        new_gp_model = gpflow.models.GPR((looft_scl, scl_looy), kernel=kernel, noise_variance=lik_noise_var)
        for param in new_gp_model.trainable_parameters:
            gpflow.set_trainable(param, False)
        gp_new.fit_gp_model = new_gp_model
        gp_new.feature_train_data = loo_ft_trn

        #Predict With New model
        #Get data in vector form into array form
        if len(loo_ft.shape) < 2:
            loo_ft.reshape(1,-1)
        #scale eval_point if necessary
        if gp_new.normalize == True:
            eval_points = gp_new.scalerX.transform(loo_ft)
        else:
            eval_points = loo_ft
        
        #Evaluate GP given parameter set theta and state point value
        gp_mean_scl, gp_covar_scl = gp_new.fit_gp_model.predict_f(eval_points, full_cov=True)
        #Remove dimensions of 1
        gp_mean_scl = gp_mean_scl.numpy()
        gp_covar_scl = np.squeeze(gp_covar_scl, axis = 0)

        #Unscale gp_mean and gp_covariance
        if gp_new.normalize == True:
            gp_mean = gp_new.scalerY.inverse_transform(gp_mean_scl.reshape(-1,1)).flatten()
            gp_covar = float(gp_new.scalerY.scale_**2) * gp_covar_scl  
        else:
            gp_mean = gp_mean_scl
            gp_covar = gp_covar_scl
        
        y_loo_mean = gp_mean
        y_loo_var = np.diag(gp_covar)
        y_loo_std =  np.sqrt(y_loo_var)

        #Check for feat_loo in train and test feat
        is_row_train = np.any(np.all(gp_object.feature_train_data == loo_ft, axis=1))
        if is_row_train:
            train_mean.append(float(y_loo_mean))
            train_std.append(float(y_loo_std))
        else:
            predict_mean.append(float(y_loo_mean))
            predict_std.append(float(y_loo_std))

    return np.array(predict_mean), np.array(predict_std), np.array(train_mean), np.array(train_std)

noise_mean = 0
set_seed = 1
retrain_GP = 25
normalize = True

cs_list = [1,2,3,10,11,12,13,14,15,16,17]
num_x =   [5,5,5, 5,10,10,10, 5,10, 6,12]

num_stds_ax = np.linspace(0,10,101)
norm_vals = (norm.cdf(num_stds_ax) - norm.cdf(-num_stds_ax))*100
idx_3 = np.argmin(np.abs(num_stds_ax - 3.0))+1

fig, ax = plt.subplots()
ax2 = fig.add_axes([0.6, 0.25, 0.3, 0.3]) # left, bottom, width, height
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
    simulator = simulator_helper_test_fxns(cs_name_val, noise_mean, None, set_seed)
    num_theta_data = len(simulator.indeces_to_consider)*10
    #Generate Exp Data (OR Add your own experimental data as a Data class object)
    gen_meth_x = Gen_meth_enum(gen_meth_x)
    exp_data = simulator.gen_exp_data(num_x_data, gen_meth_x, set_seed)
    #Set simulator noise_std artifically as 5% of y_exp mean (So that noise will be set rather than trained)
    simulator.noise_std = np.abs(np.mean(exp_data.y_vals))*0.05
    #Note at present, training data is always the same between jobs since we set the data generation seed to 1
    sim_data = simulator.gen_sim_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, 1.0, set_seed, False)
    val_data = simulator.gen_sim_data(15, 15, Gen_meth_enum(1), Gen_meth_enum(1), 1.0, set_seed + 1, False)

    #Make Emulator
    noise_std = simulator.noise_std #Yexp_std is exactly the noise_std of the GP Kernel
    #Get GP object
    gp_object = Type_2_GP_Emulator(sim_data, val_data, None, None, None, Kernel_enum(1), None, noise_std, None, 
                                    retrain_GP, set_seed, normalize, None, None, None, None)
    #Train on all sets
    train_data, test_data = gp_object.set_train_test_data(1.0, set_seed)
    gp_object.train_gp()

    #Create a Data Class which is Sim and Val Data together
    t_train_val = np.concatenate((sim_data.theta_vals, val_data.theta_vals))
    x_train_val = np.concatenate((sim_data.x_vals, val_data.x_vals))
    y_train_val = np.concatenate((sim_data.y_vals, val_data.y_vals))
    all_data = Data(t_train_val, x_train_val, y_train_val, None, None, None, None, None, 
    sim_data.bounds_theta, sim_data.bounds_x, 1.0, set_seed)
    
    predict_mean, predict_std, train_mean, train_std = perform_LOO(all_data, gp_object)

    #Save LOO Plots
    fig_loo, ax_loo = plt.subplots()
    ax_loo.errorbar(sim_data.y_vals, train_mean, 1.96*np.array(train_std), c='g', fmt = ' ', alpha = 0.3, zorder =1)
    ax_loo.errorbar(val_data.y_vals, predict_mean, 1.96*np.array(predict_std), c='b', fmt = ' ', alpha = 0.3, zorder=2)
    ax_loo.scatter(sim_data.y_vals, train_mean, 30,c='g', marker='D', label='LOO-Train', zorder= 3)
    ax_loo.scatter(val_data.y_vals, predict_mean, 30,c='b', marker='D', label='LOO-Test', zorder = 4)
    ax_loo.plot(all_data.y_vals,all_data.y_vals,'k--', label='parity line')
    plt.xlabel('Experimental y')
    plt.ylabel('Predicted y')
    plt.title(fill("CS " + str(cs_name_val) + ": " + kernel_info(gp_object.fit_gp_model.kernel), 60), fontdict={'size':15})
    plt.grid(True)
    ax_loo.legend(loc = 'best', fontsize='x-large')
    fig_loo.tight_layout()
    dir_LOO = "Results/LOO_Retrain_"+str(retrain_GP)
    os.makedirs(dir_LOO, exist_ok = True)
    plt.savefig(dir_LOO +"/" + "cs" + str(cs_name_val) +".png",  bbox_inches='tight')
    plt.close()

    #For Each Point, calculate the std multiplier that puts us in the prediction interval
    #Calculate num_stds = (y-mu)/sigma
    num_stds_pred = sorted(abs((val_data.y_vals-predict_mean)/predict_std).tolist(), reverse = True)
    num_stds_train = sorted(abs((sim_data.y_vals-train_mean)/train_std).tolist(), reverse = True)
    num_stds = num_stds_pred + num_stds_train
    
    #Calculate % in side prediction interval at each point
    pct_ins = np.array([np.sum(num_stds < num_std) for num_std in num_stds_ax])/len(num_stds)*100
    ax.plot(num_stds_ax, pct_ins, color = cmap(i), alpha=0.7, label = "CS" + str(cs_name_val))

    #Find the index closest to 3
    sub_x = num_stds_ax[:idx_3]
    sub_y = pct_ins[:idx_3]

    # Create a subplot for the subsection
    ax2.plot(sub_x, sub_y, color = cmap(i), alpha=0.7)
    
sub_yn = norm_vals[:idx_3]
ax2.plot(sub_x, sub_yn, color = "k", alpha=0.7)
ax.plot(num_stds_ax, norm_vals, color='black', alpha=0.7, label = "Std Norm")
fig.legend(loc= "upper right", bbox_to_anchor=(-0.02, 1), 
                       borderaxespad=0)
ax.set_xlabel('Number of Std. Deviations')
ax.set_ylabel('% Inside Prediction Interval')
ax.set_title('PIPI With ' + str(retrain_GP) + " GP Retrains")
ax.grid(True)
ax2.grid(True)
fig.tight_layout()
dir_PIPI = "Results/PIPI"
os.makedirs(dir_PIPI, exist_ok = True)
plt.savefig(dir_PIPI + "/LOO_Retrain_"+str(retrain_GP) +".png",  bbox_inches='tight')
plt.close()