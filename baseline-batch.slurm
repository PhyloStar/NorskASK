#!/bin/bash

#SBATCH --job-name=stigjb-train-model
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9447k
#SBATCH --time=01:00:00
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

	module purge
	module use -a /projects/nlpl/software/modulefiles/
	module load \
		nlpl-python-candy/201902/3.5 \
		nlpl-gensim/3.7.0/3.5 \
		nlpl-tensorflow/1.11

	chkfile "results" "models"
fi
set +o errexit

mlp="python -m masterthesis.models.mlp"
linear="python -m masterthesis.models.linear_baseline"
logreg="$linear --algorithm logreg"
svc="$linear --algorithm svc"
svr="$linear --algorithm svr"

echo $SLURM_ARRAY_TASK_ID $SLURM_JOB_ID

case $SLURM_ARRAY_TASK_ID in
	1) cmd="$logreg --kind bow" ;;
	2) cmd="$logreg --kind char" ;;
	3) cmd="$logreg --kind pos" ;;
	4) cmd="$logreg --kind mix" ;;
    
	5) cmd="$svc --kind bow" ;;
	6) cmd="$svc --kind char" ;;
	7) cmd="$svc --kind pos" ;;
	8) cmd="$svc --kind mix" ;;

	9) cmd="$svr --kind bow" ;;
	10) cmd="$svr --kind char" ;;
	11) cmd="$svr --kind pos" ;;
	12) cmd="$svr --kind mix" ;;

	13) cmd="$mlp bow --method classification" ;;
	14) cmd="$mlp char --method classification" ;;
	15) cmd="$mlp pos --method classification" ;;
	16) cmd="$mlp mix --method classification" ;;

	17) cmd="$mlp bow --method regression" ;;
	18) cmd="$mlp char --method regression" ;;
	19) cmd="$mlp pos --method regression" ;;
	20) cmd="$mlp mix --method regression" ;;

	21) cmd="$mlp bow --method ranked" ;;
	22) cmd="$mlp char --method ranked" ;;
	23) cmd="$mlp pos --method ranked" ;;
	24) cmd="$mlp mix --method ranked" ;;

	# Run script with arguments if not an array job
	'') cmd=$mlp ;;
	*)
		echo "Array ID out of range"
		exit 1
		;;
esac

$cmd "$@"
