#!/bin/bash

#SBATCH --job-name=stigjb-train-rnn-model
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
cmd="python -m masterthesis.models.rnn"

embdim50="--vectors models/vectors/127-small.pkl"
embdim100="--vectors models/vectors/120-small.pkl"

case $SLURM_ARRAY_TASK_ID in
	1) args="--pool-method mean" ;;
	2) args="--pool-method max" ;;
	3) args="--pool-method attention" ;;
	4) args="--include-pos --pool-method mean" ;;
	5) args="--include-pos --pool-method max" ;;
	6) args="--include-pos --pool-method attention" ;;
	7) args="--mixed-pos --pool-method mean" ;;
	8) args="--mixed-pos --pool-method max" ;;
	9) args="--mixed-pos --pool-method attention" ;;

	10) args="--bidirectional --pool-method mean" ;;
	11) args="--bidirectional --pool-method max" ;;
	12) args="--bidirectional --pool-method attention" ;;
	13) args="--bidirectional --include-pos --pool-method mean" ;;
	14) args="--bidirectional --include-pos --pool-method max" ;;
	15) args="--bidirectional --include-pos --pool-method attention" ;;
	16) args="--bidirectional --mixed-pos --pool-method mean" ;;
	17) args="--bidirectional --mixed-pos --pool-method max" ;;
	18) args="--bidirectional --mixed-pos --pool-method attention" ;;

	19) args="$embdim100 --pool-method mean" ;;
	20) args="$embdim100 --pool-method max" ;;
	21) args="$embdim100 --pool-method attention" ;;
	22) args="$embdim100 --include-pos --pool-method mean" ;;
	23) args="$embdim100 --include-pos --pool-method max" ;;
	24) args="$embdim100 --include-pos --pool-method attention" ;;

	25) args="$embdim100 --bidirectional --pool-method mean" ;;
	26) args="$embdim100 --bidirectional --pool-method max" ;;
	27) args="$embdim100 --bidirectional --pool-method attention" ;;
	28) args="$embdim100 --bidirectional --include-pos --pool-method mean" ;;
	29) args="$embdim100 --bidirectional --include-pos --pool-method max" ;;
	30) args="$embdim100 --bidirectional --include-pos --pool-method attention" ;;

	# Run script with arguments if not an array job
	'') args="" ;;
	*)
		echo "Array ID out of range"
		exit 1
		;;
esac

$cmd $args "$@"
