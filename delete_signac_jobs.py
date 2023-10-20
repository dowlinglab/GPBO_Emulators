import signac

project = signac.get_project()

for job in project.find_jobs({'cs_name_val': 2, 'param_name_str':"y0", 'retrain_GP': 10, 'num_x_data': 10}):
    job.remove() 