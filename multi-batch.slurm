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

export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0

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

rnn="python -m masterthesis.models.rnn --multi"
cnn="python -m masterthesis.models.cnn --multi"

embdim50="--vectors models/vectors/127-small.pkl"
embdim100="--vectors models/vectors/127-small.pkl"

case $SLURM_ARRAY_TASK_ID in
	1) cmd="$cnn --static-embs $embdim50" ;;
	2) cmd="$cnn --static-embs $embdim50 --round-cefr" ;;
	3) cmd="$cnn --static-embs $embdim50 --aux-loss-weight 0.25" ;;
	4) cmd="$cnn --static-embs $embdim50 --aux-loss-weight 0.25 --round-cefr" ;;
	5) cmd="$cnn --static-embs $embdim50 --aux-loss-weight 0.75" ;;
	6) cmd="$cnn --static-embs $embdim50 --aux-loss-weight 0.75 --round-cefr" ;;

	7) cmd="$cnn --include-pos --static-embs $embdim50 --aux-loss-weight 0.25" ;;
	8) cmd="$cnn --include-pos --static-embs $embdim50 --aux-loss-weight 0.25 --round-cefr" ;;
	9) cmd="$cnn --include-pos --static-embs $embdim50" ;;
	10) cmd="$cnn --include-pos --static-embs $embdim50 --round-cefr" ;;
	11) cmd="$cnn --include-pos --static-embs $embdim50 --aux-loss-weight 0.75" ;;
	12) cmd="$cnn --include-pos --static-embs $embdim50 --aux-loss-weight 0.75 --round-cefr" ;;

	13) cmd="$rnn $embdim50 --attention --rnn-cell gru --aux-loss-weight 0.25" ;;
	14) cmd="$rnn $embdim50 --attention --rnn-cell gru --aux-loss-weight 0.25 --round-cefr" ;;
	15) cmd="$rnn $embdim50 --attention --rnn-cell gru" ;;
	16) cmd="$rnn $embdim50 --attention --rnn-cell gru --round-cefr" ;;
	17) cmd="$rnn $embdim50 --attention --rnn-cell gru --aux-loss-weight 0.75" ;;
	18) cmd="$rnn $embdim50 --attention --rnn-cell gru --aux-loss-weight 0.75 --round-cefr" ;;
 
	19) cmd="$rnn $embdim50 --include-pos --attention --aux-loss-weight 0.25" ;;
	20) cmd="$rnn $embdim50 --include-pos --attention --aux-loss-weight 0.25 --round-cefr" ;;
	21) cmd="$rnn $embdim50 --include-pos --attention" ;;
	22) cmd="$rnn $embdim50 --include-pos --attention --round-cefr" ;;
	23) cmd="$rnn $embdim50 --include-pos --attention --aux-loss-weight 0.75" ;;
	24) cmd="$rnn $embdim50 --include-pos --attention --aux-loss-weight 0.75 --round-cefr" ;;

	# Run script with arguments if not an array job
	*)
		echo "Array ID out of range"
		exit 1
		;;
esac

$cmd "$@"
