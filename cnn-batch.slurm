#!/bin/bash

#SBATCH --job-name=stigjb-train-model
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9447k
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G

# Increase this number when you really need parallel computing 
# (don't set it to more than 6 or 8 cores):
#SBATCH --ntasks-per-node 1

export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0

if [ -n "${SLURM_JOB_NODELIST}" ]; then
    source /cluster/bin/jobsetup

    cp -r $SUBMITDIR/masterthesis $SCRATCH
    mkdir $SCRATCH/ASK
    cp $SUBMITDIR/ASK/metadata.csv $SCRATCH/ASK/metadata.csv
    ln -s $SUBMITDIR/ASK/txt $SCRATCH/ASK
    ln -s $SUBMITDIR/ASK/conll $SCRATCH/ASK/conll
    mkdir -p $SCRATCH/models/stopwords
    cp $SUBMITDIR/models/stopwords/* $SCRATCH/models/stopwords/

    cd $SCRATCH
fi
set +o errexit

module purge
module use -a /projects/nlpl/software/modulefiles/
module load \
	nlpl-python-candy/201902/3.5 \
	nlpl-gensim/3.7.0/3.5 \
	nlpl-tensorflow/1.11

chkfile "results" "models"
cmd="python -m masterthesis.models.cnn --constraint 3"
embdim100="--vectors models/vectors/120-small.pkl"

case $SLURM_ARRAY_TASK_ID in
	1) args="--method classification" ;;
	2) args="--method classification --include-pos" ;;
	3) args="--method classification --mixed-pos" ;;

	4) args="--method regression" ;;
	5) args="--method regression --include-pos" ;;
	6) args="--method regression --mixed-pos" ;;

	7) args="--method ranked" ;;
	8) args="--method ranked --include-pos" ;;
	9) args="--method ranked --mixed-pos" ;;

	10) args="--method classification $embdim100" ;;
	11) args="--method classification --include-pos $embdim100" ;;

	12) args="--method regression $embdim100" ;;
	13) args="--method regression --include-pos $embdim100" ;;

	14) args="--method ranked $embdim100" ;;
	15) args="--method ranked --include-pos $embdim100" ;;

	# Run script with arguments if not an array job
	'') args="" ;;
	*)
		echo "Array ID out of range"
		exit 1
		;;
esac

$cmd $args "$@"
