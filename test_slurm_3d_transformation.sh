#! /bin/bash

# Uses test registered image for testing slurm_3d_transformation

image_name=/path/to/file
transformation_name=/path/to/file
atlas=/path/to/file
output_path=/path/to/directory
output_file=/path/to/file

sbatch slurm_3d_transformation.sh $image_name $transformation_name $output_path $output_file
