#!/bin/bash
#SBATCH -J diag-h16
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:10:00
module purge; module load nvhpc/26.5
source /nfs/home/koying/mantissa-optimize/apps/pyscf-venv/bin/activate
python /nfs/home/koying/mantissa-optimize/gh200/logs/diag/h16.py
