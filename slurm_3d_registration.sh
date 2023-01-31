#! /bin/bash
#SBATCH --job-name=3d_registration
#SBATCH --output=out_%j_%u.log
#SBATCH --nodes=1
#SBATCH --mem=16G

export E_BADARGS=65

# Check for correct number of arguments
if [ ! -n "$6" ]
then
    echo "Usage: `basename $0` <target-name> <output-dir> <seg_name> <output-file> <atlas> <template>"
    exit $E_BADARGS
fi

module load singularity

# Set singularity alias
shopt -s expand_aliases
alias 3dRegistration='singularity exec -B /panfs 3d_registration.sif /code/registration.py'

target_name$1
output_dir=$2
seg_name=$3
output_file=$4
atlas=$5
template=$6

# Start registration
date
echo "Running '3dRegistration $target_name $output_dir $seg_name $output_file $atlas $template'"
3dRegistration $target_name $output_dir $seg_name $output_file $atlas $template
echo "done..."
