#!/bin/bash
#SBATCH -p serial
#SBATCH -q normal
#SBATCH -t 0-21
#SBATCH -n 7
#SBATCH -o log/slurm.%j.out
#SBATCH -e log/slurm.%j.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aheida@asu.edu
case="${1:?ERROR--need a case number}"
module purge
module load anaconda/py3
source activate watertorch
which -a python
python FullWH_v2.py $case
