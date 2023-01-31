#! /bin/bash

target_name$1
output_dir=$2
output_file=$3
config_file=$4
atlas=$5

# launch
sbatch slurm_3d_transformation.sh $target_name $output_dir $output_file $config_file $atlas
