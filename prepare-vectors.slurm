#!/bin/bash

#SBATCH --job-name=stigjb-train-model
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9447k
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G

# Increase this number when you really need parallel computing 
# (don't set it to more than 6 or 8 cores):
#SBATCH --ntasks-per-node 4

if [ -n "${SLURM_JOB_NODELIST}" ]; then
    source /cluster/bin/jobsetup

    cp -r $SUBMITDIR/masterthesis $SCRATCH
    mkdir $SCRATCH/ASK
    cp $SUBMITDIR/ASK/metadata.csv $SCRATCH/ASK/metadata.csv
    ln -s $SUBMITDIR/ASK/txt $SCRATCH/ASK
    ln -s $SUBMITDIR/ASK/conll $SCRATCH/ASK/conll
    mkdir -p $SCRATCH/models/vectors

    cd $SCRATCH
fi
set +o errexit

module purge
module use -a /projects/nlpl/software/modulefiles/
module load \
	nlpl-python-candy/201902/3.5 \
	nlpl-gensim/3.7.0/3.5 \
	nlpl-tensorflow/1.11

chkfile "models"

python -m masterthesis.data.prepare_vectors $1 $SCRATCH/models/vectors

