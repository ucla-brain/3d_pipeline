#! /bin/bash
#SBATCH --job-name=3d_downsample
#SBATCH --output=out_%j_%u.log
#SBATCH --nodes=1
#SBATCH --mem=32G

export E_BADARGS=65

# Check for correct number of arguments
if [ ! -n "$4" ]
then
    echo "Usage: `basename $0` <input-path> <channel> <file-type> <resolution>"
    exit $E_BADARGS
fi

module load singularity

# Set singularity alias
shopt -s expand_aliases
alias 3dDownsample='singularity exec -B /panfs 3d_downsampling.sif /code/3d_downsample.py'

file=$1
channel=$2
type=$3
resolution=$4

# Start downsample
date
echo "Running '3dDownsample $file -c $channel -t $type -res $resolution'"
3dDownsample $file -c $channel -t $type -res $resolution
echo "done..."
