#! /bin/bash
#SBATCH --job-name=3d_downsample
#SBATCH --output=out_%j_%u.log
#SBATCH --nodes=1
#SBATCH --mem=32G

export E_BADARGS=65

# Check for correct number of arguments
if [ ! -n "$5" ]
then
    echo "Usage: `basename $0` <input-path> <file-type> <outdir> <resolution> <channel>"
    exit $E_BADARGS
fi

module load singularity

# Set singularity alias
shopt -s expand_aliases
alias 3dDownsample='singularity exec -B /panfs 3d_downsampling.sif /code/downsample.py'

file=$1
type=$2
outdir=$3
resolution=$4
channel=$5

# Start downsample
date
echo "Running '3dDownsample $file $type $outdir -res $resolution -c $channel'"
3dDownsample $file $type $outdir -res $resolution -c $channel
echo "done..."
