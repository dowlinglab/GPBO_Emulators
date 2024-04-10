import numpy as np
import pandas as pd
import signac
import os
import imageio
import glob
import itertools
from itertools import combinations

import bo_methods_lib
from bo_methods_lib.bo_methods_lib.analyze_data import General_Analysis
from skimage.transform import resize

#Ignore warnings
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
# from sklearn.exceptions import InconsistentVersionWarning
# warnings.filterwarnings(action='ignore', category=InconsistentVersionWarning)

#Create mp4/gif files from pngs
#Set Stuff
meth_name_val_list = [1, 2, 3, 4, 5, 6, 7]
save_csv = True
save_figs = True
project = signac.get_project()

for val in [3, 10, 14]:
    criteria_dict = {"cs_name_val" : val,
                    "ep_enum_val": 1,
                    "gp_package":"gpflow",
                    "meth_name_val": {"$in": meth_name_val_list}}

    #Initialize filename list
    filenames = []
    #Add all Heat map data files to list
    analyzer = General_Analysis(criteria_dict, project, save_csv)
    dir_base = analyzer.make_dir_name_from_criteria(criteria_dict)
    dir_hms = dir_base+"/heat_maps/all_methods"

    z_choices = ["sse_sim", "sse_mean", "sse_var", "acq"]
    for z_choice in z_choices:
        #Create directory to store Heat Map Movies
        heat_map_files = glob.glob(dir_hms + "/*/" + z_choice + ".png")
        filenames += heat_map_files

    if save_figs is True:
        dir_name = dir_base + "/movies/"
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        z_choices_sort = sorted(z_choices, key=lambda x: ('sse_sim', 'sse_mean', 'sse_var','acq').index(x))
        z_choices_str = '_'.join(map(str, z_choices_sort))
        gif_path = dir_name + z_choices_str + ".mp4"

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