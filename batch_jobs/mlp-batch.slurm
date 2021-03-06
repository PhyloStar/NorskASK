#!/bin/bash

#SBATCH --job-name=stigjb-train-rnn-model
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9447k
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G

# Increase this number when you really need parallel computing 
# (don't set it to more than 6 or 8 cores):
#SBATCH --ntasks-per-node 4

export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0

if [ -n "${SLURM_JOB_NODELIST}" ]; then
    source /cluster/bin/jobsetup

    cp -r "$SUBMITDIR"/masterthesis "$SCRATCH"
    mkdir "$SCRATCH"/ASK
    cp "$SUBMITDIR"/ASK/metadata.csv "$SCRATCH"/ASK/metadata.csv
    ln -s "$SUBMITDIR"/ASK/txt "$SCRATCH"/ASK
    ln -s "$SUBMITDIR"/ASK/conll "$SCRATCH"/ASK/conll
    mkdir -p "$SCRATCH"/models/stopwords
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
cmd="python -m masterthesis.models.mlp"

case $SLURM_ARRAY_TASK_ID in
    1) args="bow" ;;
    2) args="bow --round-cefr" ;;
    3) args="char" ;;
    4) args="char --round-cefr" ;;
    5) args="pos" ;;
    6) args="pos --round-cefr" ;;
    7) args="mixed" ;;
    8) args="mixed --round-cefr" ;;

    9) args="bow --classification" ;;
    10) args="bow --round-cefr --classification" ;;
    11) args="char --classification" ;;
    12) args="char --round-cefr --classification" ;;
    13) args="pos --classification" ;;
    14) args="pos --round-cefr --classification" ;;
    15) args="mixed --classification" ;;
    16) args="mixed --round-cefr --classification" ;;

    # Run script with arguments if not an array job
    '') args="" ;;
    *)
        echo "Array ID out of range"
        exit 1
        ;;
esac

$cmd $args "$@"
