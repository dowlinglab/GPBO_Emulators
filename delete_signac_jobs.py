import signac

project = signac.get_project()

for job in project.find_jobs({"cs_name_val":{"$in": [15]},  'gp_package': {'$exists': True}, "ep_enum_val":{"$in": [1]} }):
    job.remove() 
