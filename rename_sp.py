import signac

project = signac.get_project()

# Define the recursive function to update state points
def recursively_update_state_points(job):
    #Check if the job is a Muller Potential job
    if 2 <= job.statepoint()['cs_name_val'] <= 4:
        #If the parameter is x0, the case study is the same
        if not job.statepoint()['param_name_str'] == "x0":
            #Otherwise add 1 to the current cs_name_val state point
            new_state_point_value = job.statepoint()['cs_name_val'] + 1
    
            # Update the state point
            job.document['cs_name_val'] = new_state_point_value
            job.init()

            # Recursively call the function on child jobs
            for child_job in job.childs():
                recursively_update_state_points(child_job)
    #otherwise do nothing
    else:
        pass

# Iterate over all jobs in the project
for job in project:
    recursively_update_state_points(job)