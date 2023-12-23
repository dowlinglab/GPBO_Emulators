import signac

project = signac.get_project()

for job in project:
    job.remove() 