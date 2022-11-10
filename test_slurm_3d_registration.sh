#! /bin/bash

# Uses test downsampled image and parameters for testing slurm_3d_registration

target_name=/path/to/file
output_dir=/path/to/directory
output_file=/path/to/file
config_file=/path/to/file
atlas=/path/to/file

sbatch slurm_3d_registration.sh $target_name $output_dir $output_file $config_file $atlas
