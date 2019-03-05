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

    cp -r $SUBMITDIR/masterthesis $SCRATCH
    mkdir $SCRATCH/ASK
    cp $SUBMITDIR/ASK/metadata.csv $SCRATCH/ASK/metadata.csv
    ln -s $SUBMITDIR/ASK/txt $SCRATCH/ASK
    ln -s $SUBMITDIR/ASK/conll $SCRATCH/ASK/conll
    mkdir $SCRATCH/models
    mkdir $SCRATCH/models/stopwords
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
embdim50="--vectors models/vectors/127-small.pkl"

case $SLURM_ARRAY_TASK_ID in
	1) args="" ;;
	2) args="--round-cefr" ;;
	3) args="$embdim50" ;;
	4) args="$embdim50 --round-cefr" ;;
	5) args="--static-embs $embdim50" ;;
	6) args="--static-embs $embdim50 --round-cefr" ;;

	7) args="--include-pos" ;;
	8) args="--include-pos --round-cefr" ;;
	9) args="--include-pos $embdim50" ;;
	10) args="--include-pos $embdim50 --round-cefr" ;;
	11) args="--include-pos --static-embs $embdim50" ;;
	12) args="--include-pos --static-embs $embdim50 --round-cefr" ;;

	13) args="--mixed-pos" ;;
	14) args="--mixed-pos --round-cefr" ;;
	15) args="--mixed-pos --windows 2,3,4" ;;
	16) args="--mixed-pos --windows 2,3,4 --round-cefr" ;;

	# Run script with arguments if not an array job
	'') args="" ;;
	*)
		echo "Array ID out of range"
		exit 1
		;;
esac

$cmd $args "$@"
