import signac
import os

project = signac.get_project()

for job in project.find_jobs({"meth_name_val":{"$in": [5,6]}}):
    # print(job.fn("signac_statepoint.json"))

    if os.path.isfile(job.fn("BO_Results.gz")):
        # print(job.fn("signac_statepoint.json"))
    # # Delete the file
        os.remove(job.fn("BO_Results.gz"))
        os.remove(job.fn("BO_Results_GPs.gz"))
    # job.remove() 
