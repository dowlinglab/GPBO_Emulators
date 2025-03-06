import numpy as np
from bo_methods_lib.bo_methods_lib.GPBO_Classes_New import * #Fix this later
from bo_methods_lib.bo_methods_lib.GPBO_Class_fxns import * #Fix this later
from bo_methods_lib.bo_methods_lib.analyze_data import * #Fix this later
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import * #Fix this later
from sklearn.preprocessing import RobustScaler
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

#Ignore inconcistent version warning
import warnings
warnings.filterwarnings('ignore')

#Make Simulator and Training Data
cs_name_val = 3
noise_mean = 0
noise_std = None
seed = 1
#Define method, ep_enum classes, indecies to consider, and kernel
meth_name = Method_name_enum(3)
method = GPBO_Methods(meth_name)
gen_meth_theta = Gen_meth_enum(1)
gen_meth_x = Gen_meth_enum(2)
num_x_data = 5

#Define Simulator Class (Export your Simulator Object Here)
simulator = simulator_helper_test_fxns(cs_name_val, noise_mean, noise_std, seed)
num_theta_data = len(simulator.indices_to_consider)*10
set_seed = 1 #Set set_seed to 1 for data generation
gen_meth_x = Gen_meth_enum(gen_meth_x)
exp_data = simulator.gen_exp_data(num_x_data, gen_meth_x, None, 0.01)
#Set simulator noise_std artifically as 1% of y_exp median (So that noise will be set rather than trained)
simulator.noise_std = np.abs(np.median(exp_data.y_vals))*0.01
#Note at present, training data is always the same between jobs since we set the data generation seed to 1
all_gp_data = simulator.gen_sim_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, 1.0, seed, False, None)
all_val_data = simulator.gen_sim_data(10, 10, Gen_meth_enum(1), Gen_meth_enum(1), 1.0, seed + 1, False, None)


#Scale training data if necessary
scalerY = RobustScaler(unit_variance = True)
y = scalerY.fit_transform(all_gp_data.y_vals.reshape(-1,1))
y_test = scalerY.transform(all_val_data.y_vals.reshape(-1,1))

plt.hist(y, bins=50, cumulative = True, density=True, stacked=True, edgecolor = 'k', label = "Train") 
plt.hist(y_test, bins=50, cumulative = True, stacked=True, density=True, label = "Test", edgecolor = 'k', alpha = 0.5) # Adjust the number of bins as needed
xmin, xmax = plt.xlim() 
mu, std = norm.fit(y[~np.isnan(y)])
x = np.linspace(xmin, xmax, 100) 
p = norm.cdf(x, mu, std) 
# plt.plot(x, p, 'g', linewidth=2, label = "Normal CDF") 

print(f"train min: {min(y)}, train max: {max(y)}, test min: {min(y_test)}, test max: {max(y_test)}")

percentile_95_train = np.percentile(y[~np.isnan(y)], 95)
percentile_95_test = np.percentile(y_test[~np.isnan(y_test)], 95)
print(f"train 95%: {percentile_95_train}, test 95%: {percentile_95_test}")

val1 = -1.2
val2 = 2
train_between = np.sum((y >= val1) & (y <= val2))
train_proportion = train_between / len(y)

# Calculate the proportion for the testing set
test_between = np.sum((y_test >= val1) & (y_test <= val2))
test_proportion = test_between / len(y_test)

print("Percent train/test data which fall in interval [-1.2,2]")
print(train_proportion, test_proportion)

# Add labels and title
plt.xlabel(r'$y^{sim}$' +' Value')
plt.ylabel('Cumulative Probability Density')
plt.title('Histogram of Müller ' + r'$y_0$' ' Data')
plt.legend(loc = "best")

plt.savefig("cs3_hist.png")