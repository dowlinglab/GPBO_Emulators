import signac

project = signac.get_project()

for job in project.find_jobs({"param_name_str":"y0", "bo_iter_tot":100}):
    job.remove() 