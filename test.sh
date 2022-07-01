#!/bin/bash
# required in inputfile for smp 4:

#$ -pe smp 1
#$ -m ae
#$ -r n
#$ -M mcarlozo@nd.edu
#$ -q long
#$ -N GP2_runs_15_100

export PATH=/afs/crc.nd.edu/user/m/mcarlozo/.conda/envs/Toy_Problem_env/bin:$PATH

set INPUT = "$1"

python3  2_Input_GPBO.py

