{% extends "base_script.sh" %}

#!/bin/bash

#$ -N {{ id }}
#$ -pe smp 1
#$ -r n
#$ -q long
#$ -m ae
#$ -M mcarlozo@nd.edu

export PATH=/afs/crc.nd.edu/user/m/mcarlozo/.conda/envs/Toy_Problem_env/bin:$PATH
conda activate Toy_Problem_env

{% block tasks %}
{% endblock %}
{% endblock %}
