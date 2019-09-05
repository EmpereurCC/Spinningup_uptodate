#! /bin/bash

set -e


for FLAGS in "ppo_pyco --env warehouse_manager-v0 --num_cpu 1 --tensorboard_path 'home/travis/build/EmpereurCC/spinningup_instadeep2'" "ppo_pyco --env warehouse_manager-v0 --num_cpu 4 --tensorboard_path 'home/travis/build/EmpereurCC/spinningup_instadeep2'" "ppo_pyco --env fluvial_natation-v1 --num_cpu 2 --tensorboard_path 'home/travis/build/EmpereurCC/spinningup_instadeep2'" ;
      oarsub -l "{host='big14'}/nodes=1/core=1,walltime=0:120:0" --notify "mail:clement.collgon@lip6.fr" "source /home/ccollgon/anaconda3/etc/profile.d/conda.sh; cd /home/ccollgon/git/spinningup_instadeep2/ && conda activate cc && python spinup/run.py $FLAGS "
      #oarsub -l "nodes=1/core=5,walltime=0:120:0"  --notify "mail:clement.collgon@lip6.fr" "cd /home/ccollgon/git/spinningup_instadeep2/ && conda activate cc && python spinup/run.py $FLAGS ; exit"     
      #oarsub -l "{host='big6' or host='big7' or host='big8' or host='big9' or host='big10' or host='big11' or host='big12' or host='big13' or host='big14'}/nodes=1/core=1,walltime=0:120:0"   --notify "mail:clement.collgon@lip6.fr" "cd /home/ccollgon/git/spinningup_instadeep2/ && python spinup/run.py $FLAGS ; exit"      
done

# walltime : hour:mn:sec 

# exemple 
# "{host='big6' or host='big7' or host='big8' or host='big9' or host='big10' or host='big11' or host='big12' or host='big13' or host='big14'}/nodes=1/core=1,walltime=0:12:0"
PARAMSV="3e-4 6e-4 9e-4 1e-3 3e-3 5e-3 8e-3 1e-2 3e-2"
PARAMSPI="3e-4 6e-4 9e-4 1e-3 3e-3 5e-3 8e-3 1e-2 3e-2"
MODELS="fluvial_natation-v1 shockwave-v0 warehouse_manager-v0 apprehend-v0"
for i in $PARAMSV ; for j in $PARAMSPI ; for m in $MODELS ; do  FLAGS="ppo_pyco --env $m --max_ep_len 500 --num_cpu 1 --pi_lr $PARAMSPI --vf_lr $PARAMSV --tensorboard_path /home/ccollgon/git/spinningup_instadeep2/Tensorboard_place/$m-1cpu" ; oarsub -l "{host='big12' or host='big8' or host='big9' or host='big10'}/nodes=1/core=1,walltime=0:7200:0" --notify "mail:clement.collgon@lip6.fr" "source /home/ccollgon/anaconda3/etc/profile.d/conda.sh; cd /home/ccollgon/git/spinningup_instadeep2/ && conda activate cc && python spinup/run.py $FLAGS " ; done