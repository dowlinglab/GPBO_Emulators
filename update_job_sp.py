import signac
project = signac.get_project()

for job in project:
    assert "bo_runs_in_job" not in job.sp
    job.sp.bo_runs_in_job = job.statepoint.pop("bo_run_tot")
    if job.sp.bo_runs_in_job == 1:
        if "bo_run_num" not in job.sp:
            job.sp.setdefault("bo_run_num", int((job.sp.seed-1/2)+1))
        if job.sp.cs_name_val in [2,3] and job.sp.bo_iter_tot == 75:
            job.sp.setdefault("bo_run_tot", 10)
        else:
            job.sp.setdefault("bo_run_tot", 5)
    else:
        job.sp.setdefault("bo_run_tot", job.sp.bo_runs_in_job)
        if "bo_run_num" not in job.sp:
            job.sp.setdefault("bo_run_num", 1)