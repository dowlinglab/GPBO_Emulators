import signac
from bo_methods_lib.bo_methods_lib.analyze_data import All_CS_Analysis
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import All_CS_Plotter

#Ignore warnings
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)

#Set parameters
meth_list = [1, 2, 3, 4, 5, 6, 7]
cs_list = [11,14,2,1,12,13,3,10]
save_csv = True #Set to False if you don't want to save/resave csvs
save_figs = True
bar_modes = ["objs", "time"] #time and/or objs
project = signac.get_project("GPBO_Fix")

analyzer = All_CS_Analysis(cs_list, meth_list, project, "act", save_csv)
plotters = All_CS_Plotter(analyzer, save_figs)

#Get % true found
#Change cs_list here to get averages over select case studies
analyzer.get_percent_true_found(cs_list)

#Make Overall GPBO bar charts
for bmode in bar_modes:
    df_average = plotters.make_bar_charts(bmode)

#Make Derivative Free Bar Charts
df_med_derivfree = plotters.make_derivfree_bar(s_meths = ["NLS", "SHGO-Sob", "NM", "GA"], ver = "med")