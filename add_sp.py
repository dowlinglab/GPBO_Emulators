import signac

project = signac.get_project()

for job in project.find_jobs({"cs_name_val":3, "bo_iter_tot": 75}):
    assert "bo_run_num" not in job.sp
    job.update_statepoint({"bo_run_num":int((job.sp.seed-1)/2 + 1)})