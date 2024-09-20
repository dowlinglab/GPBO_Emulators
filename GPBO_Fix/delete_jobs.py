import signac
import os

project = signac.get_project()

#Add categories of jobs you want to remove in ()
for job in project.find_jobs(): 
    #Remove job files
    if os.path.isfile(job.fn("BO_Results.gz")):
        os.remove(job.fn("BO_Results.gz"))
        os.remove(job.fn("BO_Results_GPs.gz"))
    # job.remove() #Uncomment if you want to remove the job from the project
