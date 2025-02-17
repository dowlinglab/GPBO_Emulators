
import signac
import os
from bo_methods_lib.bo_methods_lib.analyze_data import LS_Analysis
from bo_methods_lib.bo_methods_lib.GPBO_Class_fxns import simulator_helper_test_fxns

#Ignore warnings
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)

#Set Stuff
meth_name_val_list = [1,2,3,4,5,6,7]
save_csv = True #Set to False if you don't want to save/resave csvs
save_figs = True
project = signac.get_project("GPBO_Fix")
seed = 1

for val in [15,16,17]:
    criteria_dict = {
        "cs_name_val": val,
        "ep_enum_val": 1,
        "gp_package": "gpflow",
        "meth_name_val": {"$in": meth_name_val_list},
    }

    analyzer = LS_Analysis(criteria_dict, project, save_csv)
    #Get Simulator Object
    simulator = simulator_helper_test_fxns(val, 0, None, 1) #This is a dummy simulator object
    #Get all least squares solutions
    tot_runs = 1000
    ls_analyzer = LS_Analysis(criteria_dict, project, save_csv, simulator=simulator)
    local_mins = ls_analyzer.categ_min(tot_runs)

    #Get best runs
    ls_results = ls_analyzer.least_squares_analysis()
    ls_results_sort = ls_results.sort_values(by=['Min Obj Cum.', 'Iter'], ascending=[True, True])
    ls_runs = ls_results_sort.drop_duplicates(subset="Run", keep='first')
    ls_best_path = os.path.join(
            ls_analyzer.study_results_dir,
            "ls_best_run.csv"
        )
    ls_analyzer.save_data(ls_runs, ls_best_path)