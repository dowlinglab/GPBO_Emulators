import signac

project = signac.get_project()

for job in project.find_jobs({'cs_name_val': {'$lt': 2.0}, 'retrain_GP': 10}):
    job.remove() 