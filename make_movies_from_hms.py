import numpy as np
import pandas as pd
import signac
import os
import imageio
import glob
import itertools
from itertools import combinations
import signac

import bo_methods_lib
from bo_methods_lib.bo_methods_lib.GPBO_Classes_New import CS_name_enum, Method_name_enum
from bo_methods_lib.bo_methods_lib.analyze_data import get_best_data, make_plot_dict, get_df_all_jobs, make_dir_name_from_criteria, analyze_hypers, analyze_thetas 
from bo_methods_lib.bo_methods_lib.GPBO_Classes_plotters import plot_objs_all_methods, plot_one_obj_all_methods, plot_2D_Data_w_BO_Iter
from skimage.transform import resize

#Ignore warnings
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
# from sklearn.exceptions import InconsistentVersionWarning
# warnings.filterwarnings(action='ignore', category=InconsistentVersionWarning)

#Create mp4/gif files from pngs
#Initialize filename list
filenames = []
#Add all Heat map data files to list (All will be the same so just take files from 1st job)
#Create directory to store Heat Map Movies
job = job_list_best[0]
heat_map_files = glob.glob(job.fn("Heat_Maps/"+ z_choice + "_all_methods" + "/*.png"))
filenames += heat_map_files

for job in job_list_best:
    if save_fig is True:
        dir_name = job.fn("")
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        gif_path = dir_name + z_choice + "_all_methods" + ".mp4"

        #Create .mp4 file
        with imageio.get_writer(gif_path, mode='I', fps=0.3) as writer: #Note. For gif use duration instead of fps
            #For each file
            for filename in filenames: 
                #Get image
                image = imageio.imread(filename, pilmode = "RGBA")
                #Get the correct shape for the pngs based on the 1st file
                if filename == filenames[0]: 
                    shape = image.shape
                    #Force image to have XY dims divisible by 16
                    new_shape = (np.ceil(shape[0] / 16) * 16, np.ceil(shape[1] / 16) * 16, shape[2])
                #If item shapes not the same force them to be the same. Fixes issues where pixels are off
                if image.shape is not shape: 
                    image = resize(image, (new_shape))
                #Add file to movie as a uint8 type and multiply array by 255 to get correct coloring
                writer.append_data((image*255).astype(np.uint8))