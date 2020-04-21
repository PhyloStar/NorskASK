#!/bin/bash

#SBATCH --job-name=stigjb-train-rnn-model
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9447k
#SBATCH --time=07:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G

# Increase this number when you really need parallel computing 
# (don't set it to more than 6 or 8 cores):
#SBATCH --ntasks-per-node 1

export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0

if [ -n "${SLURM_JOB_NODELIST}" ]; then
    source /cluster/bin/jobsetup

    cp -r "$SUBMITDIR"/masterthesis "$SCRATCH"
    mkdir "$SCRATCH"/ASK
    cp "$SUBMITDIR"/ASK/metadata.csv "$SCRATCH"/ASK/metadata.csv
    ln -s "$SUBMITDIR"/ASK/txt "$SCRATCH"/ASK
    ln -s "$SUBMITDIR"/ASK/conll "$SCRATCH"/ASK/conll
    mkdir "$SCRATCH"/models
    mkdir "$SCRATCH"/models/stopwords
    cp "$SUBMITDIR"/models/stopwords/* "$SCRATCH"/models/stopwords/

    cd "$SCRATCH"
fi
set +o errexit

module purge
module use -a /projects/nlpl/software/modulefiles/
module load \
	nlpl-python-candy/201902/3.5 \
	nlpl-gensim/3.7.0/3.5 \
	nlpl-tensorflow/1.11

chkfile "results" "models"

embdim100="--vectors models/vectors/120-small.pkl"

rnnbase="python -m masterthesis.models.rnn --method regression --embed-dim 100"
rnn1="$rnnbase --rnn-cell gru --bidirectional --include-pos --pool-method attention"
rnn2="$rnnbase --rnn-cell gru --bidirectional $embdim100 --pool-method attention"

case $SLURM_ARRAY_TASK_ID in
	0) cmd="$rnn1" ;;
	1) cmd="$rnn2" ;;
    2) cmd="$rnn1 --aux-loss-weight 0.1" ;;
    3) cmd="$rnn2 --aux-loss-weight 0.1" ;;

	*)
		echo "Array ID out of range"
		exit 1
		;;
esac

cmd="$cmd --save-model"
echo "SLURM BATCH RUNNING COMMAND: $cmd"
$cmd "$@"

