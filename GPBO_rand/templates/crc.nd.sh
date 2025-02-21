{% extends "base_script.sh" %}
{% block header %}

#!/bin/bash

#$ -N {{ id }}
#$ -pe smp {{ np_global }}
#$ -r n
#$ -q long
#$ -m ae
#$ -M mcarlozo@nd.edu

export PATH=/afs/crc.nd.edu/user/m/mcarlozo/.conda/envs/Toy_Problem_env/bin:$PATH

{% block tasks %}
{% endblock %}
{% endblock %}