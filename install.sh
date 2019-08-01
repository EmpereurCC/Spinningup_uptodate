#! /bin/bash

cd
if [ ! -d ~/anaconda3/bin ] ; then rm -rf ~/anaconda3/ ; fi ;
if [ ! -d ~/anaconda3/bin ] ; then wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh -O ~/anaconda.sh ; fi ;
if [ ! -d ~/anaconda3/bin ] ; then bash ~/anaconda.sh -b -p $HOME/anaconda3 ; fi ;
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
conda init
conda update -y conda
conda create -y -n cc python=3.6 anaconda || true
conda activate cc
conda install -y tensorflow setuptools seaborn mpi4py cloudpickle joblib tqdm
cd -
    
conda activate cc
pip install -e .
cd gym
pip install -e .
cd ../PycoGYM/gym_pyco
pip install -e .
cd ../..

   

