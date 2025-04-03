import numpy as np
import signac
from itertools import combinations
from bo_methods_lib.bo_methods_lib.analyze_data import General_Analysis, LS_Analysis
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import Plotters
from bo_methods_lib.bo_methods_lib.GPBO_Class_fxns import simulator_helper_test_fxns

# Ignore warnings
import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)

# From signac
import signac

meth_name_val_list = [1, 2, 3, 4, 5, 6, 7]
tot_runs_nls = 1000
save_csv = True  # Set to False if you don't want to save/resave csvs
save_figs = True
modes = ["act", "gp", "acq"]
project = signac.get_project("GPBO_nonoise")

for mode in modes:
    for val in [1, 11, 15, 17]:
        criteria_dict = {
            "cs_name_val": val,
            "ep_enum_val": 1,
            "gp_package": "gpflow",
            "meth_name_val": {"$in": meth_name_val_list},
        }

        # simulator = simulator_helper_test_fxns(val, 0, None, 1) #This is a dummy simulator object
        # analyzer = General_Analysis(criteria_dict, project, mode, save_csv)
        analyzer_NLS = LS_Analysis(criteria_dict, project, save_csv)
        plotters_NLS = Plotters(analyzer_NLS, save_figs)

        analyzer = General_Analysis(criteria_dict, project, mode, save_csv)
        plotters = Plotters(analyzer, save_figs)

        ### Get Best Data from ep experiment
        df_best, job_list_best = analyzer.get_best_data()

        # Set z_choices and levels
        z_choices = ["sse_sim", "sse_mean", "sse_var", "acq"]
        levels = 100

        # Back out number of parameters
        string_val = df_best["Theta Min Obj"].iloc[0]
        try:
            numbers = [
                float(num)
                for num in string_val.replace("[", "").replace("]", "").split()
            ]
        except:
            numbers = [float(num) for num in string_val]
        # Create list of parameter pair combinations
        dim_theta = len(np.array(numbers).reshape(-1, 1))
        dim_list = np.linspace(0, dim_theta - 1, dim_theta)
        pairs = len((list(combinations(dim_list, 2))))

        # Loop over parameter pairs
        if pairs < 3:
            for pair in range(pairs):
                # Plot local minima for each method
                if mode == "act":
                    plotters_NLS.plot_local_min_hms(pair, tot_runs_nls)
                # And all objectives
                for z_choice in z_choices:
                    # Plot heat maps for one objective with each method in a different subplot
                    plotters.plot_hms_all_methods(pair, z_choice, log_data=False)
