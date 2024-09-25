# Bayesian Optimization Methods for Nonlinear Model Calibration
Authors: Montana N. Carlozo, Ke Wang, and Alexander W. Dowling
<!-- Introduction: Provide a brief introduction to the project, including its purpose, goals, and any key features or benefits. -->
## Introduction
**GPBO_Emulators** is a repository used to calibrate computationally expensive models given experimental data. The key feature of this work is using machine learning tools in the form of Gaussian processes (GPs) and Bayesian Optimization (BO) which allow us to smartly sample parameter space to decrease the objective. This work features comparing standard GPBO in which a GP models an expensive objective function to emulator GPBO in which an expensive function is emulated directly by the GP.

## Citation
This work was submitted to Industrial & Engineering Chemistry Research (I&ECR). Please cite as:

Montana N. Carlozo, Ke Wang, Alexander W. Dowling, “Bayesian Optimization Methods for Nonlinear Model Calibration”, 2024
   
## Available Data

### Repository Organization
The repository is organized as follows: <br />
GPBO_Emulators/ is the top level directory. It contains: <br />
1. .gitignore prevents large files from the signac workflow and plots from being tracked by git and prevents  tracking of other unimportant files. <br />
2. init_gpbofix.py generated the setup for running the workflow with signac <br />
3. make_1meth_hms.py makes contour plots of Simulated SSE, GP SSE, GP SSE variance, and acqusition function for a given GP model. <br />
4. make_1obj_hms.py makes contour plots of either Simulated SSE, GP SSE, GP SSE variance, or acqusition function for a given GP model for each method. <br />
5. make_bar_charts.py makes Figures 2 and 7 from the main text (bar charts of relevant data). <br />
6. make_cond_num_data.py generates condition number data for the best GP models. <br />
7. makes_least_squares_data.py generates data related to nonlinear least squares (NLS) including categorizing number of local minima and best performing NLS runs. <br />
8. make_line_plots.py generates all line plots in the main text and SI including parity plots and plots of BO iteration for SSE and acquisition function for all modes of evaluation. <br />
9. make_movies_from_hms.py makes movies (.mp4) from the contour plots generated with make_1obj_hms.py. <br />
10. make_muly0_hist.py makes the histogram for M\"uller y0 case study data. <br />
11. gpbo-emul.yml is the environment for running this workflow. <br />

Directory bo_methods_lib/ contains the package for running the workflow. <br />
bo_methods_lib/bo_methods_lib/ contains the following files: <br />
1. GPBO_Classes_New.py is the main script for the algorithm. <br />
2. GPBO_Class_fxns.py are helper functions which define parameters for the multiple case studies. This function is also useful for mapping the numerical markers of case studies to their formal names from the manuscript. <br />
3. analyze_data.py and GPBO_Classes_plotters.py are used for analyzing and plotting the results. <br />
4. tests/*.py contains test functions for public methods of all classes in GPBO_Classes_New.py. <br />

Directory GPBO_Fix/ is initially created via init_gpbofix.py in the top directory through signac. <br /> 
It contains the following files/subdirectories: <br />
1. delete_jobs.py is a script for quickly deleting targeted jobs/results in signac. <br />
2. view_unfinished_jobs.py is a script for viewing individual case study job progress. <br />
3. project_GPBO_Fix.py is the script for running the workflow using signac. <br />
4. templates/ are the templates required to run this workflow in signac on the crc. <br />
5. workspace/ will appear to save all raw results generated during the workflow after running init_gpbofix.py. This file is not tracked by git due to its size. the workspace/ folder for this study can be downloaded on Google Drive (see section 'Workflow Files and Results') <br />
    
Running the analysis (steps 6 -11 below) will cause results directories to appear with relevant human readable data and plots. Subdirectories further categorize the results by case study and methods analyzed. <br />
1. Results_acq/ shows data where we analyze the best results based on how efficiently the acquisition function was optimized. <br />
2. Results_act/ shows data where we analyze the best results based on how efficiently the actual SSE was optimized. <br />
3. Results_gp/ shows data where we analyze the best results based on how efficiently the GP predicted SSE was optimized. <br />

We note that this repository is based on the branch ``update_and_merge`` in the ``dowlinglab/Toy_Problem`` repository, which is private.

### Workflow Files and Results
All workflow iterations were performed inside ``GPBO_Emulators/GPBO_Fix`` where it exists.
Each iteration was managed with ``signac-flow``. Inside ``GPBO_Fix``, you will find all the necessary files to
run the workflow. Note that you may not get the exact same simulation
results due to differences in software versions, random seeds, etc.
The raw results from this study are available in [Google Drive](https://drive.google.com/drive/u/0/folders/16etwfqNa7Hxe4r9RNrWnSzjIscO2ZsFU>). These files are necessary to download and add to ``GPBO_Emulators/GPBO_Fix`` to reproduce all figures and tables identified in the paper. As such, the directory ``GPBO_Emulators/GPBO_Fix/workspace`` must exist and contain all files from the google drive before analysis scripts are run. The analysis scripts work by parsing the signac workflow data and manipulating the data into a form conducive for analysis. The files are saved on google drive because the files in which the GP models reside are often on the order of gigabytes. For other plots, we also use Google Drive because the workspace folder has hundreds of subdirectories (many of which are also large) which makes storage on GitHub impractical.
When reproducing the results of this workflow, more practical, human readable csv files and figures are stored under ``Results_xxx/cs_name_val_y/``, where ``xxx`` represents a different analysis scheme and ``y`` represent a different (set of) case study. For example data pertaining to the BOD Curve case study for all methods where the actual SSE values were analyzed would be generated in the ``Results_act/cs_name_val_11/ep_enum_val_1/gp_package_gpflow/meth_name_val_in_1_2_3_4_5_6_7`` directory. We specifically note that the best runs for each case study and method are provided under ``Results_xxx/cs_name_val_y/best_results.csv``. These results for each case study make up results.xlsx in the supporting information (SI). We also specifically note nonlinear least squares results categorizing the number of minima are found in ``Results_act/cs_name_val_y``.

### Workflow Code
All of the scripts for running the workflow are provided in
``bo_methods_lib/bo_methods_lib/GPBO_Classes_New``. All tests for public methods in this file are found in ``bo_methods_lib/bo_methods_lib/tests``
All scipts for analyzing the data in ``GPBO_Fix/workflow`` are provided in ``GPBO_Emulators/make_*.py``.

### Figures
All scripts required to generate the primary figures in the
manuscript and SI are reported under ``GPBO_Emulators/make_*.py``. When running analysis scripts, these figures are saved under ``Results_xxx/cs_name_val_y/*``.

## Installation
To run this software, you must have access to all packages in the gpbo-emul environment (gpbo-emul.yml) which can be installed using the instructions in the next section.
<!-- Installation: Provide instructions on how to install and set up the project, including any dependencies that need to be installed. -->
This package has a number of requirements that can be installed in
different ways. We recommend using a conda environment to manage
most of the installation and dependencies. However, some items will
need to be installed from source or pip. <br />

Running the simulations will also require an installation of TASMANIAN.
This can be installed separately (see installation instructions
[here](https://github.com/ORNL/TASMANIAN) ). <br />

An example of the procedure is provided below:

    # Install pip/conda available dependencies
    # with a new conda environment named gpbo-emul
    conda env create -f gpbo-emul.yml
    conda activate gpbo-emul
    python -m pip install Tasmanian

<!-- Usage: Provide instructions on how to use the project, including any configuration or customization options. Examples of usage scenarios can also be added. -->
## Usage

### GPBO Workflow Execution

**NOTE**: We use Signac and signac flow (`<https://signac.io/>`)
to manage the setup and execution of the workflow. These
instructions assume a working knowledge of that software. <br />

**WARNING**: Running these scripts will overwrite your local copy of our data (``GPBO_Fix/workflow/*``) with the data from your workflow runs. <br />

To run the GPBO workflow, follow the following steps:
1. Use init_gpbofix.py to initialize files for simulation use
   ```
     python init_gpbofix.py
     cd GPBO_Fix
   ```    
2. Do the following in GPBO_Fix directory:
3. Check status a few times throughout the process
   ```
     python project_GPBO_Fix.py status 
   ```       
4. Modify project_GPBO_Fix.py if desired. This file is currently set to reproduce the paper results
5. Run the simulations
   ```
     python project_GPBO_Fix.py run -o run_ep_or_sf_exp
   ```     

**Note: rm -r workspace/ signac_project_document.json signac.rc will remove everything and allow you to start fresh if you mess up**

### Final Analysis
The final processing and figure generation steps can be run using the following once all signac jobs have finished.

**WARNING**: Running these scripts will overwrite your local copy of our workflow results (``Results_xxx/*``) with the results from your workflow runs. <br />

1. Make bar charts for the objective and time data (Figures 1 and 7) and get all data shown in full-results.xlsx <br />
   When this method is run, the data for Table 3, results.xlsx, and full-results.xlsx are generated.
   Note: results_xlsx is not automatically assembled. It is comprised of Results_act/*/*/*/*/best_results.csv for each method.
   ```
     python make_bar_charts.py
   ```

2. Gather nonlinear least squares data including number of local minima (in Table 3) and the best NLS solution for each run
   ```
     python make_least_squares_data.py
   ```

3. Make objective line plots (Figures 2, 5, 6, and S9-S32), parity plots (Figures 4 and S1-S8) for all case studies
   
   Hyperparameter and parameter value plots over BO iteration will also appear in GPBO_Fix/workspace/
   ```
     cd GPBO_Emulators
     python make_line_plots.py
   ```
4. Make objective contour plots (Figures 3 and S33-38) for the BOD Curve and Simple Linear case studies. This function also prints the MAE for each method.
   ```
     python make_1obj_hms.py
   ```
5. (Optional) Make movies of the plots generated via python make_1obj_hms.py
   ```
     python make_movies_from_hms.py
   ```
6. Gather condition number data for the best runs of each case study
   Condition number data is generated in gpflow_condition_numbers_raw.csv (raw) and gpflow_condition_numbers.csv (analyzed)
   ```
     python make_cond_num_data.py
   ```

7. Make histogram of M\"uller y0 data (Figure S39). This function also prints out the statistics mentioned in the paper.
   ```
     python make_muly0_hist.py
   ```

### Known Issues
The instructions outlined above seem to be system-dependent. In some cases, users have the following error:
```
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```
If you observe this, please try the following in the terminal
```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
which should fix the problem. This is not an optimal solution and is something we would like to address. We found that related projects [1](https://github.com/openmm/openmm/issues/3943), [2](https://github.com/conda/conda/issues/12410) have similar issues.
If you are aware of a robust solution to this issue, please let us know by raising an issue or sending an email!

## Credits
This work is funded by the Graduate Assistance in Areas of National Need fellowship from the Department of Education via grant number P200A210048, the National Science Foundation via award number CBET-1917474, the University of Notre Dame College of Engineering and Graduate School, and uses the computing resources provided by the Center for Research Computing (CRC) at the University of Notre Dame. Ke Wang also acknowledges the Patrick and Jana Eilers Graduate Student Fellowship for Energy-Related Research for providing financial support to advance this research. Montana Carlozo also acknowledges Dr. Ryan Smith and Dr. Bridgette Befort who offered technical guidance in the development of this work. 

## Contact
Please contact Montana Carlozo (mcarlozo@nd.edu) or Dr. Alex Dowling (adowling@nd.edu) with any questions, suggestions, or issues.

## Software Versions
This section lists software versions for the most important packages. <br />
gpflow==2.9.1 <br />
numpy==1.24.4 <br />
pandas==1.4.2 <br />
Pyomo==6.6.2 <br />
pytest==7.2.0 <br />
Python==3.9.12 <br />
scipy==1.8.0 <br />
signac==1.8.0 <br />
signac-flow==0.21.0 <br />
Tasmanian==7.7.1 <br />
torch==1.11.0 <br />
torchaudio==0.13.1 <br />
torchvision==0.14.1 <br />

