import signac

project = signac.get_project()

for job in project.find_jobs({"cs_name_val":1,  'param_name_str': {'$exists': False}, "ep_enum_val":{"$in": [2, 3, 4]} }):
    job.remove() 
