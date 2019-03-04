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
	1) $cmd ;;
	2) $cmd --round-cefr ;;
	3) $cmd --bidirectional ;;
	4) $cmd --bidirectional --round-cefr ;;
	5) $cmd --attention ;;
	6) $cmd --attention --round-cefr ;;

	7) $cmd $embdim50 ;;
	8) $cmd $embdim50 --round-cefr ;;
	9) $cmd $embdim50 --bidirectional ;;
	10) $cmd $embdim50 --bidirectional --round-cefr ;;
	11) $cmd $embdim50 --attention ;;
	12) $cmd $embdim50 --attention --round-cefr ;;

	13) $cmd $embdim100 ;;
	14) $cmd $embdim100 --round-cefr ;;
	15) $cmd $embdim100 --bidirectional ;;
	16) $cmd $embdim100 --bidirectional --round-cefr ;;
	17) $cmd $embdim100 --attention ;;
	18) $cmd $embdim100 --attention --round-cefr ;;

	19) $cmd --attention --rnn-cell gru ;;
	20) $cmd --attention --rnn-cell gru --round-cefr ;;
	21) $cmd $embdim50 --attention --rnn-cell gru ;;
	22) $cmd $embdim50 --attention --rnn-cell gru --round-cefr ;;
 
	23) $cmd $embdim50 --freeze-embeddings --attention ;;
	24) $cmd $embdim50 --freeze-embeddings --attention --round-cefr ;;
	25) $cmd $embdim100 --freeze-embeddings --attention ;;
	26) $cmd $embdim100 --freeze-embeddings --attention --round-cefr ;;

	27) $cmd --include-pos --attention ;;
	28) $cmd --include-pos --attention --round-cefr ;;
	29) $cmd $embdim50 --include-pos --attention ;;
	30) $cmd $embdim50 --include-pos --attention --round-cefr ;;
	
	# Run command with arguments if not array job
	*) $cmd "$@" ;;
esac
