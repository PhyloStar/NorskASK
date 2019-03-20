#!/bin/bash

#SBATCH --job-name=stigjb-train-model
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9447k
#SBATCH --time=01:00:00
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

mlp="python -m masterthesis.models.mlp"
linear="python -m masterthesis.models.linear_baseline"
logreg="$linear logreg"
svc="$linear svc"
svr="$linear svr"

case $SLURM_ARRAY_TASK_ID in
	1) cmd="$logreg bow" ;;
	2) cmd="$logreg char" ;;
	3) cmd="$logreg pos" ;;
	4) cmd="$logreg mix" ;;
    
	5) cmd="$svc bow" ;;
	6) cmd="$svc char" ;;
	7) cmd="$svc pos" ;;
	8) cmd="$svc mix" ;;

	9) cmd="$svr bow" ;;
	10) cmd="$svr char" ;;
	11) cmd="$svr pos" ;;
	12) cmd="$svr mix" ;;

	13) cmd="$mlp bow" ;;
	14) cmd="$mlp char" ;;
	15) cmd="$mlp pos" ;;
	16) cmd="$mlp mix" ;;

	# Run script with arguments if not an array job
	'') cmd=$mlp ;;
	*)
		echo "Array ID out of range"
		exit 1
		;;
esac

$cmd "$@"
