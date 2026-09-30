#!/bin/bash
#SBATCH -J cp2k-pull
#SBATCH -p compute
#SBATCH -w work1
#SBATCH -N 1 -c 72 --exclusive
#SBATCH -t 1:00:00
#SBATCH -o /nfs/home/koying/mantissa-optimize/apps/logs/cp2k_pull_%j.log
set -ex
cd /nfs/home/koying/mantissa-optimize/apps/cp2k
export APPTAINER_TMPDIR=/tmp/$USER-apt-$SLURM_JOB_ID APPTAINER_CACHEDIR=/tmp/$USER-aptc-$SLURM_JOB_ID
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR
time apptainer pull --force cp2k_v2024.3.sif docker://nvcr.io/hpc/cp2k:v2024.3
rm -rf $APPTAINER_TMPDIR $APPTAINER_CACHEDIR
ls -lh
apptainer exec --nv cp2k_v2024.3.sif bash -c 'which cp2k.psmp cp2k.psmp 2>/dev/null; ls /opt 2>/dev/null; env | grep -i -E "^PATH|LD_LIBRARY"'
