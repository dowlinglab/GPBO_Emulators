import signac

project = signac.get_project()

for job in project.find_jobs({'cs_name_val': 9, 'bo_iter_tot': 50}):
    job.remove() 