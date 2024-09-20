### Repository Organization
The repository is organized as follows:
GPBO_Emulators/ is the top level directory. It contains
.gitignore prevents large files from the signac workflow and plots from being tracked by git and prevents  tracking of other unimportant files. 
init_gpbofix.py generated the setup for running the workflow with signac.io
make_1meth_hms.py makes contour plots of Simulated SSE, GP SSE, GP SSE variance, and acqusition function for a given GP model
make_1obj_hms.py makes contour plots of either Simulated SSE, GP SSE, GP SSE variance, or acqusition function for a given GP model for each method
make_bar_charts.py makes Figures 2 and 7 from the main text (bar charts of relevant data)
make_cond_num_data.py generates condition number data for the best GP models
makes_least_squares_data.py generates data related to nonlinear least squares (NLS) including categorizing number of local minima and best performing NLS runs.
make_line_plots.py generates all line plots in the main text and SI including parity plots and plots of BO iteration for SSE and acquisition function for all modes of evaluation.
make_movies_from_hms.py makes movies (.mp4) from the contour plots generated with make_1obj_hms.py
make_muly0_hist.py makes the histogram for M\"uller y0 case study data.
Toy_Problem.yml is the environment for running this workflow.



Directory bo_methods_lib/ contains the package for running the workflow.
bo_methods_lib/bo_methods_lib/ contains the following files:
    GPBO_Classes_New.py is the main script for the algorithm.
    GPBO_Class_fxns.py are helper functions which define parameters for the multiple case studies. This function is also useful for mapping the numerical markers of case studies to their formal names from the manuscript.
    analyze_data.py and GPBO_Classes_plotters.py are used for analyzing and plotting the results.
bo_methods_lib/tests contains test functions for public methods of all classes in GPBO_Classes_New.py. 

Directory GPBO_Fix/ is initially created via init_gpbofix.py in the top directory through signac.io. 
It contains the following files/subdirectories:
    delete_jobs.py is a script for quickly deleting targeted jobs/results in signac.
    view_unfinished_jobs.py is a script for viewing individual case study job progress. 
    project_GPBO_Fix.py is the script for running the workflow using signac.io. 
    templates/ are the templates required to run this workflow in signac on the crc.
    workspace/ will appear to save all raw results generated during the workflow after running init_gpbofix.py. This file is not tracked by git due to its size.
    
Running the analysis (steps 6 -11 below) will cause results directories to appear with relevant human readable data and plots. Subdirectories further categorize the results by case study and methods analyzed.
Results_acq/ shows data where we analyze the best results based on how efficiently the acquisition function was optimized
Results_act/ shows data where we analyze the best results based on how efficiently the actual SSE was optimized
Results_gp/ shows data where we analyze the best results based on how efficiently the GP predicted SSE was optimized



### GPBO Optimization
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
When all simulations are completed

6. Make bar charts for the objective and time data (Figures 1 and 7)
   ```
     When this method is run, the data for Table 3, results.xlsx, and full-results.xlsx are generated.
     Note: results_xlsx is not automatically assembled. It is comprised of Results_act/*/*/*/*/best_results.csv for each method.
     python make_bar_charts.py
   ```

7. Gather nonlinear least squares data including number of local minima (in Table 3) and the best NLS solution for each run.
   ```
     python make_least_squares_data.py
   ```

8. Make objective line plots (Figures 2, 5, 6, and S9-S32), parity plots (Figures 4 and S1-S8) for all case studies
   
   Hyperparameter and parameter value plots over BO iteration will also appear in GPBO_Fix/workspace/
   ```
     cd GPBO_Emulators
     python make_line_plots.py
   ```
9. Make objective contour plots (Figures 3 and S33-38) for the BOD Curve and Simple Linear case studies. This function also prints the MAE for each method.
   ```
     python make_1obj_hms.py
   ```
10. (Optional) Make movies of the plots generated via python make_1obj_hms.py
   ```
     python make_movies_from_hms.py
   ```
11. Gather condition number data for the best runs of each case study
   Condition number data is generated in gpflow_condition_numbers_raw.csv (raw) and gpflow_condition_numbers.csv (analyzed)
   ```
     python make_cond_num_data.py
   ```

12. Make histogram of M\"uller y0 data (Figure S39). This function also prints out the statistics mentioned in the paper.
   ```
     python make_muly0_hist.py
   ```

