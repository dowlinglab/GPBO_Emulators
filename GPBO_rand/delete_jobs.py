import signac
import os

project = signac.get_project()

count = 0
#Add categories of jobs you want to remove in ()
for job in project: 
    # if "get_y_sse" in job.sp:
    #     del job.sp.get_y_sse
    # if "w_noise" in job.sp:
    #     del job.sp.w_noise

    job.update_statepoint({"get_y_sse": True, "w_noise": True})
    # job.update_statepoint({"get_y_sse": True, "w_noise": True})
# print(count)
