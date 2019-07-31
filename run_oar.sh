#! /bin/bash

set -e

for FLAGS in "ppo_pyco --env warehouse_manager-v0 --num_cpu 1 --tensorboard_path 'home/travis/build/EmpereurCC/spinningup_instadeep2'" "ppo_pyco --env warehouse_manager-v0 --num_cpu 4 --tensorboard_path 'home/travis/build/EmpereurCC/spinningup_instadeep2'" "ppo_pyco --env fluvial_natation-v1 --num_cpu 2 --tensorboard_path 'home/travis/build/EmpereurCC/spinningup_instadeep2'" ;
do
      oarsub -l "nodes=1/core=5,walltime=0:120:0"  --notify "mail:clement.collgon@lip6.fr" "cd /home/ccollgon/git/spinningup_instadeep2/ && python spinup/run.py $FLAGS ; exit"     
done

# walltime : hour:mn:sec 

# exemple 
# "{host='big6' or host='big7' or host='big8' or host='big9' or host='big10' or host='big11' or host='big12' or host='big13' or host='big14'}/nodes=1/core=1,walltime=0:12:0"

