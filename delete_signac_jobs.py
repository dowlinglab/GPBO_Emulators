import signac

project = signac.get_project()

for job in project.find_jobs({"cs_name_val":1}):
    job.remove() 