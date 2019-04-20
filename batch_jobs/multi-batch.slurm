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

cnn1="python -m masterthesis.models.cnn --method regression --mixed-pos --constraint 3.0"
cnn2="python -m masterthesis.models.cnn --method regression $embdim100 --constraint 3.0"

rnnbase="python -m masterthesis.models.rnn --method regression --embed-dim 100"
rnn1="$rnnbase --rnn-cell gru --bidirectional --include-pos --pool-method attention"
rnn2="$rnnbase --rnn-cell gru --bidirectional $embdim100 --pool-method attention"

model_select=$(bc -l <<< "scale=0; $SLURM_ARRAY_TASK_ID/9")
case $model_select in
	0) cmd=$cnn1 ;;
	1) cmd=$cnn2 ;;
	2) cmd=$rnn1 ;;
	3) cmd=$rnn2 ;;
esac

weight_select=$(bc -l <<< "scale=0; $SLURM_ARRAY_TASK_ID%9")
case $weight_select in
	0) lossweight="0.1" ;;
	1) lossweight="0.2" ;;
	2) lossweight="0.3" ;;
	3) lossweight="0.4" ;;
	4) lossweight="0.5" ;;
	5) lossweight="0.6" ;;
	6) lossweight="0.7" ;;
	7) lossweight="0.8" ;;
	8) lossweight="0.9" ;;

	# Run script with arguments if not an array job
	*)
		echo "Array ID out of range"
		exit 1
		;;
esac

echo "MODEL SELECT $model_select AUX LOSS WEIGHT $lossweight"

cmd="$cmd --aux-loss-weight $lossweight"
echo "SLURM BATCH RUNNING COMMAND: $cmd"
$cmd "$@"

