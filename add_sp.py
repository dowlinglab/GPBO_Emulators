import signac

project = signac.get_project()

for job in project:
    if 'gp_package' not in job.statepoint:
        job.sp.gp_package = "sklearn"