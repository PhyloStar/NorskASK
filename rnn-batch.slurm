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
embdim100="--vectors models/vectors/127-small.pkl"

case $SLURM_ARRAY_TASK_ID in
	1) args="" ;;
	2) args="--round-cefr" ;;
	3) args="--bidirectional" ;;
	4) args="--bidirectional --round-cefr" ;;
	5) args="--attention" ;;
	6) args="--attention --round-cefr" ;;

	7) args="$embdim50" ;;
	8) args="$embdim50 --round-cefr" ;;
	9) args="$embdim50 --bidirectional" ;;
	10) args="$embdim50 --bidirectional --round-cefr" ;;
	11) args="$embdim50 --attention" ;;
	12) args="$embdim50 --attention --round-cefr" ;;

	13) args="$embdim100" ;;
	14) args="$embdim100 --round-cefr" ;;
	15) args="$embdim100 --bidirectional" ;;
	16) args="$embdim100 --bidirectional --round-cefr" ;;
	17) args="$embdim100 --attention" ;;
	18) args="$embdim100 --attention --round-cefr" ;;

	19) args="--attention --rnn-cell gru" ;;
	20) args="--attention --rnn-cell gru --round-cefr" ;;
	21) args="$embdim50 --attention --rnn-cell gru" ;;
	22) args="$embdim50 --attention --rnn-cell gru --round-cefr" ;;
 
	23) args="$embdim50 --static-embs --attention" ;;
	24) args="$embdim50 --static-embs --attention --round-cefr" ;;
	25) args="$embdim100 --static-embs --attention" ;;
	26) args="$embdim100 --static-embs --attention --round-cefr" ;;

	27) args="--include-pos --attention" ;;
	28) args="--include-pos --attention --round-cefr" ;;
	29) args="$embdim50 --include-pos --attention" ;;
	30) args="$embdim50 --include-pos --attention --round-cefr" ;;

	# Run script with arguments if not an array job
	'') args="" ;;
	*)
		echo "Array ID out of range"
		exit 1
		;;
esac

$cmd $args "$@"
