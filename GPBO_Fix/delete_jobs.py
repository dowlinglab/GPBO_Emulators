import signac
import os

project = signac.get_project()

for job in project.find_jobs({"cs_name_val":{"$in": [4,5,6,7,8,9,15,16,17]}}):
    print(job.fn("signac_statepoint.json"))

    # if os.path.isfile(job.fn("BO_Results.gz")):
    # # Delete the file
    #     os.remove(job.fn("BO_Results.gz"))
    job.remove() 
