import signac
from bo_methods_lib.bo_methods_lib.analyze_data import All_CS_Analysis
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import All_CS_Plotter

#Ignore warnings
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)


#Set Stuff
meth_list = [1, 2, 3, 4, 5, 6, 7]
cs_list = [11,14,2,1,12,13,3,10]
save_csv = True #Set to False if you don't want to save/resave csvs
save_figs = True
modes = ["act"]
bar_modes = ["objs", "time"] #time and/or objs
project = signac.get_project("GPBO_Fix")

for mode in modes:
    analyzer = All_CS_Analysis(cs_list, meth_list, project, mode, save_csv)
    plotters = All_CS_Plotter(analyzer, save_figs)

    #Make Parity Plots
    for bmode in bar_modes:
        #Get % true found
        analyzer.get_percent_true_found(cs_list)
        df_average = plotters.make_bar_charts(bmode)