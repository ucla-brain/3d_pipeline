#! /bin/bash
#SBATCH --job-name=3d_registration
#SBATCH --output=out_%j_%u.log
#SBATCH --nodes=1
#SBATCH --mem=32G

export E_BADARGS=65

# Check for correct number of arguments
if [ ! -n "$5" ]
then
    echo "Usage: `basename $0` <target-name> <output-dir> <output-file> <config-file> <atlas>"
    exit $E_BADARGS
fi

module load singularity

# Set singularity alias
shopt -s expand_aliases
alias 3dRegistration='singularity exec -B /panfs 3d_registration.sif /code/3d_registration.py'

target_name$1
output_dir=$2
output_file=$3
config_file=$4
atlas=$5

# Start registration
date
echo "Running '3dRegistration $target_name -c -od $output_dir -of $output_file -cf $config_file -a $atlas'"
3dRegistration $target_name -c -od $output_dir -of $output_file -cf $config_file -a $atlas
echo "done..."
