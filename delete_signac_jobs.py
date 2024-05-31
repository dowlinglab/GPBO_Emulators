import signac
import os

project = signac.get_project()

for job in project.find_jobs({"cs_name_val":{"$in": [11]}, "gp_package":"gpflow", "meth_name_val":{"$in": [2]} }):
    print(job.id)

    if os.path.isfile(job.fn("BO_Results.gz")):
    # Delete the file
        os.remove(job.fn("BO_Results.gz"))
    # job.remove() 
