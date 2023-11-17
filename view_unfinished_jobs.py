import signac

def check_post_conditions(job):
    # Implement your post-condition checking logic here.
    # Return True if the post-conditions are met, otherwise, return False.
    # You can access job data using job.statepoint().
    # Example: Checking if a 'status' key in the job state point is 'completed'.
    return job.isfile("BO_Results.gz")

def find_jobs_with_unmet_conditions(project, cs_name_val):
    unmet_conditions_jobs = []
    for job in project.find_jobs({"cs_name_val":cs_name_val}):
        if not check_post_conditions(job):
            unmet_conditions_jobs.append(job)
    return unmet_conditions_jobs

def get_unfinished_ep_jobs(project, cs_name_val):
    # Initialize the Signac project
    project = signac.get_project()

    # Find jobs that have not met post-conditions
    unmet_conditions_jobs = find_jobs_with_unmet_conditions(project, cs_name_val)
    
    if len(unmet_conditions_jobs) == 0:
        return print("No unfinished jobs.")
    else:
        # Get the state point for the first job
        state_point_first_job = unmet_conditions_jobs[0].statepoint()

        # Initialize a list to store keys with different values
        different_keys = []

        # Iterate through keys and compare their values
        for key, value_first_job in state_point_first_job.items():
            is_different = any(job.statepoint().get(key) != value_first_job for job in unmet_conditions_jobs[1:])
            if is_different:
                different_keys.append(key)

        # Print the values
        for job in unmet_conditions_jobs:
            # Get the values of the specified keys from the job's state point
            state_point = job.statepoint()
            subset_state_point = {key: state_point[key] for key in different_keys}

            # Print the subset state point
            print("Job ", job.id, subset_state_point)
        return

project = signac.get_project()
get_unfinished_ep_jobs(project, 2)
