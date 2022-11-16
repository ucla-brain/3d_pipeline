#! /bin/bash
#SBATCH --job-name=3d_transformation
#SBATCH --output=out_%j_%u.log
#SBATCH --nodes=1
#SBATCH --mem=32G

export E_BADARGS=65

# Check for correct number of arguments
if [ ! -n "$5" ]
then
    echo "Usage: `basename $0` <image-name> <transformation-name> <atlas> <output-path> <output_file>"
    exit $E_BADARGS
fi

module load singularity

# Set singularity alias
shopt -s expand_aliases
alias 3dTransformation='singularity exec -B /panfs 3d_transformation.sif /code/3d_transformation.py'

image_name=$1
transformation_name=$2
atlas=$3
output_path=$4
output_file=$5

# Start transformation
date
echo "Running '3dTransformation $image_name $transformation_name -a $atlas -d $output_path -o $output_file'"
3dTransformation $image_name $transformation_name -a $atlas -d $output_path -o $output_file
echo "done..."
