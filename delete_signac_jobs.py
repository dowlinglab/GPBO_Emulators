import signac

project = signac.get_project()

for job in project.find_jobs({"cs_name_val":10,  'param_name_str': {'$exists': False}}):
    job.remove() 
