#!/bin/bash

#SBATCH --job-name=stigjb-train-model
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9447k
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G

# Increase this number when you really need parallel computing 
# (don't set it to more than 6 or 8 cores):
#SBATCH --ntasks-per-node 4

if [ -n "${SLURM_JOB_NODELIST}" ]; then
    source /cluster/bin/jobsetup

    cp -vr $SUBMITDIR/masterthesis $SCRATCH
    mkdir $SCRATCH/ASK
    cp -v $SUBMITDIR/ASK/metadata.csv $SCRATCH/ASK/metadata.csv
    cp -vr $SUBMITDIR/ASK/txt $SCRATCH/ASK

    echo $SUBMITDIR
    cd $SCRATCH
fi
set +o errexit

module purge
module use -a /projects/nlpl/software/modulefiles/
module add nlpl-inf5820

chkfile "results/"

python -m masterthesis.models.taghipour_ng "$@"
